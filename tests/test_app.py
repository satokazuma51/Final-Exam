"""Real Streamlit rendering and state/persistence tests; no fabricated ML metrics."""
from __future__ import annotations

import json
import os
import base64
from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import load_config
from src import analytics, committee, db, ledger
from src.ui import today
from src.exports import csv_bytes, report_metadata, xlsx_bytes, pdf_bytes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ui_config(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    paths = {key: str(tmp_path / key) for key in ("processed_dir", "artifact_dir", "reports_dir")}
    paths.update(database=str(tmp_path / "operational.sqlite"), analytics_database=str(tmp_path / "source.sqlite"),
                 data_dir=str(ROOT / "home-credit-default-risk"))
    path.write_text(json.dumps({"paths": paths}), encoding="utf-8")
    monkeypatch.setenv("LENDING_CONFIG", str(path))
    return load_config(path)


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def test_two_main_sections_and_all_tabs_render(ui_config):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.sidebar.radio[0].options == ["Кредитная деятельность", "Аналитика"]
    assert [tab.label for tab in app.tabs] == ["Новая заявка", "Клиенты", "Заявки", "Кредиты и платежи"]
    app.sidebar.radio[0].set_value("Аналитика").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Портфель", "Продукты", "Активность заявок", "Качество модели", "Отчёты"]
    assert any("ещё не рассчитаны" in item.value for item in app.info)


def test_invalid_form_never_creates_credit_decision(ui_config):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    widget(app.button, "Рассчитать и сохранить решение").click().run()
    assert not app.exception
    assert any("Исправьте исходные данные" in error.value for error in app.error)
    assert db.list_applications(ui_config) == []
    assert db.list_clients(ui_config) == []


def test_missing_model_is_honest_and_does_not_save_decision(ui_config):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    widget(app.text_input, "ФИО").set_value("Тестовый новый клиент")
    widget(app.text_input, "Цель кредита").set_value("Учебный пример")
    widget(app.button, "Рассчитать и сохранить решение").click().run()
    assert not app.exception
    assert any("модель отсутствует" in item.value for item in app.warning)
    assert db.list_applications(ui_config) == []
    assert db.list_clients(ui_config) == []


def test_client_creation_survives_reruns_and_duplicate_clicks(ui_config):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    widget(app.text_input, "ФИО нового клиента").set_value("Тестовый клиент")
    widget(app.button, "Создать клиента").click().run()
    assert not app.exception
    assert len(db.list_clients(ui_config)) == 1
    widget(app.button, "Создать клиента").click().run()
    assert len(db.list_clients(ui_config)) == 1
    widget(app.button, "Подготовить создание другого клиента").click().run()
    widget(app.button, "Создать клиента").click().run()
    assert len(db.list_clients(ui_config)) == 2
    assert len({row["full_name"] for row in db.list_clients(ui_config)}) == 1
    restarted = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not restarted.exception
    assert len(db.list_clients(ui_config)) == 2


def test_ui_separate_disbursement_payment_and_early_settlement(ui_config):
    """A persisted test-only decision isolates UI accounting from model training."""
    db.init_db(ui_config)
    day = today().isoformat()
    client_id = db.create_client(ui_config, "Клиент UI-теста")
    inputs = {"monthly_income": 100000, "amount": 10000, "term_months": 12, "annual_rate": 18,
              "product": "cash", "age": 35, "employment_type": "Working", "employment_years": 5,
              "existing_monthly_debt": 0, "essential_expenses": 1000, "initial_draw": 10000,
              "application_date": day, "intended_disbursement_date": day}
    application_id = db.submit_application(ui_config, client_id, inputs, "ui-test-application", day)
    # This is an explicitly labelled test fixture, never a reported model result.
    score_fixture = {"probability": .01, "threshold": .5, "model_version": "test-fixture-only", "history_used": False}
    db.save_decision(ui_config, application_id, committee.evaluate_application(ui_config, inputs, score_fixture))
    assert analytics.snapshot(ui_config, day)["kpis"]["active_portfolio_balance"] == 0
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    widget(app.button, "Оформить выдачу").click().run()
    assert not app.exception
    agreements = ledger.list_agreements(ui_config, as_of=day)
    assert len(agreements) == 1
    assert agreements[0]["outstanding_principal"] == 10000
    widget(app.button, "Зарегистрировать платёж").click().run()
    assert not app.exception
    state = ledger.agreement_snapshot(ui_config, agreements[0]["agreement_id"], day)
    assert state["outstanding_principal"] == 9000
    widget(app.button, "Зарегистрировать платёж").click().run()
    assert len(ledger.agreement_snapshot(ui_config, agreements[0]["agreement_id"], day)["payments"]) == 1
    widget(app.button, "Погасить полностью по расчёту").click().run()
    assert not app.exception
    final = ledger.agreement_snapshot(ui_config, agreements[0]["agreement_id"], day)
    assert final["status"] == "Closed"
    assert final["outstanding_principal"] == 0
    app.sidebar.radio[0].set_value("Аналитика").run()
    assert not app.exception
    assert any(metric.label == "Основной долг на отчётную дату" and metric.value == "0.00 у.е." for metric in app.metric)


def test_report_exports_preserve_filtered_rows_and_metadata():
    records = [{"agreement_id": 7, "outstanding_principal": 123.45, "source_kind": "local"}]
    metadata = report_metadata("2026-10-31", "2026-01-01", "2026-10-31", {"product": "cash"})
    from io import BytesIO
    frame = pd.read_csv(BytesIO(csv_bytes(records, metadata)))
    assert frame["agreement_id"].tolist() == [7]
    assert frame["outstanding_principal"].sum() == 123.45
    assert frame["report_reporting_date"].iloc[0] == "2026-10-31"
    assert "cash" in frame["report_filters"].iloc[0]
    assert xlsx_bytes({"Договоры": records}, metadata).startswith(b"PK")
    assert pdf_bytes({"outstanding_principal": 123.45}, metadata).startswith(b"%PDF")


@pytest.mark.skipif(os.environ.get("RUN_REAL_MODEL_UI") != "1", reason="Run explicitly after real source preparation and training")
def test_genuine_model_existing_new_rejection_route(tmp_path, monkeypatch):
    """Acceptance uses untouched fitted models and real source history, no score overrides."""
    base = load_config(ROOT / "config.example.json")
    assert (Path(base["paths"]["artifact_dir"]) / "model_bundle.joblib").exists()
    base["paths"]["database"] = str(tmp_path / "real_model_acceptance.sqlite")
    path = tmp_path / "real_config.json"
    path.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("LENDING_CONFIG", str(path))
    config = load_config(path)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    widget(app.button, "Загрузить демо-профиль").click().run()
    assert not app.exception
    imported = db.list_clients(config)
    assert len(imported) == 1 and imported[0]["source_application_id"] == 100002
    widget(app.radio, "Клиент").set_value("Существующий клиент").run()
    app.selectbox(key="new_application_client").set_value(imported[0]["client_id"]).run()
    widget(app.number_input, "Запрашиваемая сумма, у.е.").set_value(10000.0)
    widget(app.number_input, "Необходимые расходы семьи, у.е./мес.").set_value(1000.0)
    widget(app.text_input, "Цель кредита").set_value("Приёмочная проверка: исходный профиль и новая заявка")
    widget(app.button, "Рассчитать и сохранить решение").click().run()
    assert not app.exception
    application = db.list_applications(config)[0]
    assert application["decision"] == "Approved", application["decision_details"]
    assert application["decision_details"]["history_used"] is True
    assert application["source_kind"] == "local" and not application["is_synthetic"]
    assert ledger.list_agreements(config) == []
    widget(app.button, "Оформить выдачу").click().run()
    assert not app.exception
    loan = ledger.list_agreements(config)[0]
    assert loan["outstanding_principal"] == 10000
    widget(app.button, "Зарегистрировать платёж").click().run()
    assert not app.exception
    assert ledger.agreement_snapshot(config, loan["agreement_id"])["outstanding_principal"] == 9000
    widget(app.button, "Зарегистрировать платёж").click().run()
    assert len(ledger.agreement_snapshot(config, loan["agreement_id"])["payments"]) == 1
    app.sidebar.radio[0].set_value("Аналитика").run()
    assert not app.exception
    assert any(metric.label == "Основной долг на отчётную дату" and metric.value == "9 000.00 у.е." for metric in app.metric)
    app.sidebar.radio[0].set_value("Кредитная деятельность").run()
    widget(app.radio, "Клиент").set_value("Новый клиент").run()
    widget(app.text_input, "ФИО").set_value("Приёмочный новый клиент")
    widget(app.number_input, "Запрашиваемая сумма, у.е.").set_value(10000.0)
    widget(app.number_input, "Необходимые расходы семьи, у.е./мес.").set_value(1000.0)
    widget(app.text_input, "Цель кредита").set_value("Новая анкета без вымышленной истории")
    widget(app.button, "Рассчитать и сохранить решение").click().run()
    assert not app.exception
    newest = db.list_applications(config)[0]
    assert newest["decision_details"]["history_used"] is False
    assert newest["decision"] in ("Approved", "Rejected")
    assert len(ledger.list_agreements(config)) == 1
    widget(app.number_input, "Месячный доход, у.е.").set_value(10000.0)
    widget(app.number_input, "Все существующие платежи по долгам, у.е./мес.").set_value(20000.0)
    widget(app.text_input, "Цель кредита").set_value("Проверка реального отказа по долговой нагрузке")
    widget(app.button, "Рассчитать и сохранить решение").click().run()
    assert not app.exception
    rejected = db.list_applications(config)[0]
    assert rejected["decision"] == "Rejected"
    assert rejected["decision_details"]["dti"] > config["policy"]["max_dti"]
    assert len(ledger.list_agreements(config)) == 1
    app.selectbox(key="application_details").set_value(rejected["application_id"]).run()
    widget(app.button, "Повторно оценить: сохранить новую версию решения").click().run()
    assert not app.exception
    assert len(db.list_decisions(config, rejected["application_id"])) == 2


def pie_sum(app, chart_key):
    chart = next(element for element in app.get("plotly_chart") if chart_key in element.proto.id)
    values = json.loads(chart.proto.spec)["data"][0]["values"]
    if isinstance(values, dict) and "bdata" in values:
        values = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return float(sum(values))


@pytest.mark.skipif(os.environ.get("RUN_REAL_MODEL_UI") != "1", reason="Run explicitly after reproducible real-model demo generation")
def test_full_demo_visuals_respect_active_portfolio_and_filters(tmp_path, monkeypatch):
    """Read-only backup of real generated records verifies chart/KPI populations."""
    base = load_config(ROOT / "config.example.json")
    clone = tmp_path / "visual_basis.sqlite"
    with closing(sqlite3.connect("file:" + Path(base["paths"]["database"]).as_posix() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(clone)) as target:
            source.backup(target)
    base["paths"]["database"] = str(clone)
    path = tmp_path / "visual_config.json"
    path.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("LENDING_CONFIG", str(path))
    config = load_config(path)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Аналитика").run()
    assert not app.exception and not app.error
    assert any(expander.label.startswith("EDA:") for expander in app.expander)
    as_of = app.date_input(key="analytics_asof").value.isoformat()
    expected = analytics.snapshot(config, as_of)
    assert expected["kpis"]["closed_agreements"] > 0
    assert pie_sum(app, "product_donut") == expected["kpis"]["active_agreements"]
    profile = json.loads((Path(config["paths"]["processed_dir"]) / "profile.json").read_text(encoding="utf-8"))
    assert pie_sum(app, "goods_donut") == profile["tables"]["previous_application"]["rows"]
    app.selectbox(key="analytics_product").set_value("cash")
    app.selectbox(key="analytics_source").set_value("synthetic")
    widget(app.button, "Применить фильтры").click().run()
    assert not app.exception and not app.error
    expected = analytics.snapshot(config, as_of, filters={"product": "cash", "source_kind": "synthetic"})
    assert pie_sum(app, "product_donut") == expected["kpis"]["active_agreements"]
    app.radio(key="portfolio_composition").set_value("Остаток основного долга").run()
    assert abs(pie_sum(app, "product_donut") - expected["kpis"]["active_portfolio_balance"]) < .01
    # Historical goods remain a separately labelled whole-source population.
    assert pie_sum(app, "goods_donut") == profile["tables"]["previous_application"]["rows"]
