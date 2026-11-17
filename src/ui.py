"""Russian operational interface backed by persisted accounting and real ML artifacts."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import logging
import sqlite3
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st

from src import analytics, committee, db, features, ledger, models
from src.config import load_config
from src.exports import KPI_LABELS, csv_bytes, pdf_bytes, readable_value, report_metadata, xlsx_bytes

LOGGER = logging.getLogger(__name__)
PRODUCTS = {"cash": "Денежный кредит", "revolving": "Возобновляемая кредитная линия"}
STATUSES = {"Approved": "Одобрено", "Rejected": "Отказано", "Pending": "Ожидает решения",
            "Active": "Активен", "Closed": "Закрыт", "Disbursed": "Выдано"}
SOURCES = {"local": "Введено пользователем", "synthetic": "Демонстрационные операции", "demo": "Демонстрационные операции",
           "home_credit": "Исторический Home Credit", "source": "Исторический Home Credit"}
LABELS = {"client_id": "ID клиента", "full_name": "ФИО", "source_application_id": "Исходная заявка",
          "application_id": "ID заявки", "agreement_id": "ID договора", "product": "Продукт", "status": "Статус",
          "amount": "Сумма, у.е.", "outstanding_principal": "Основной долг, у.е.",
          "accrued_interest": "Начисленные проценты, у.е.", "overdue_amount": "Просрочка, у.е.",
          "dpd": "Дней просрочки", "max_dpd": "Максимум дней просрочки", "source_kind": "Происхождение",
          "is_synthetic": "Синтетическая запись", "client_type": "Тип клиента", "annual_rate": "Номинальная ставка, %",
          "term_months": "Срок, мес.", "application_date": "Дата заявки", "disbursement_date": "Дата выдачи",
          "due_date": "Срок платежа", "installment_number": "№ платежа", "payment": "Платёж, у.е.",
          "principal": "Основной долг", "interest": "Проценты", "paid": "Оплачено", "unpaid": "Не оплачено",
          "credit_limit": "Кредитный лимит, у.е.", "available_credit": "Доступно, у.е.", "risk_probability": "Риск TARGET=1",
          "probability": "Риск TARGET=1", "payment_date": "Дата платежа", "transaction_date": "Дата операции",
          "installment_no": "№ платежа", "actual_paid": "Фактически оплачено", "unpaid_amount": "Осталось по обязательству",
          "remaining_principal": "Остаток основного долга", "principal_paid": "Погашено основного долга", "interest_paid": "Оплачено процентов",
          "kind": "Операция", "opened_date": "Дата открытия", "closed_date": "Дата закрытия", "decision": "Решение"}
LABELS.update({"variant": "Набор признаков", "model": "Модель", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC (AP)",
               "precision": "Точность (precision)", "recall": "Полнота (recall)", "f1": "F1", "threshold": "Порог риска",
               "training_seconds": "Обучение, с", "inference_seconds": "Прогноз, с", "model_bytes": "Размер, байт",
               "feature": "Признак", "contribution": "Вклад (логарифм шансов)", "direction": "Направление", "method": "Метод",
               "brier_score": "Brier", "log_loss": "Log-loss", "inference_rows": "Строк прогноза", "cv_roc_auc": "ROC-AUC CV",
               "cv_std": "Разброс ROC-AUC CV", "params": "Проверенные параметры", "working_set_mb": "Рабочая память, МиБ",
               "peak_working_set_mb": "Пиковая память, МиБ", "model_version": "Версия модели", "policy_version": "Версия политики",
               "decision_version": "Версия решения", "decision_date": "Дата решения", "created_at": "Создано (UTC)",
               "scheduled_end_date": "Договорный срок окончания", "total_disbursed": "Всего выдано, у.е.", "payment_total": "Всего оплачено, у.е."})
FEATURE_NAMES = {
    "monthly_income": "Месячный доход", "loan_amount": "Сумма кредита / лимит", "monthly_payment": "Месячный платёж",
    "employment_years": "Стаж", "loan_to_annual_income": "Кредит / годовой доход", "payment_to_income": "Платёж / доход",
    "bureau_count": "Число кредитов бюро", "bureau_active_count": "Активные кредиты бюро", "bureau_debt_sum": "Долг бюро",
    "bureau_max_dpd": "Максимальный DPD бюро", "bureau_overdue_sum": "Просрочка бюро", "bureau_months": "Месяцы истории бюро",
    "bureau_late_fraction": "Доля просроченных месяцев бюро", "bureau_max_severity": "Максимальная тяжесть просрочки бюро",
    "previous_count": "Предыдущие заявки", "previous_approval_rate": "Доля прежних одобрений", "previous_recent_count": "Заявки за предыдущий год",
    "previous_mean_credit": "Средняя сумма прежних заявок", "pos_months": "Месяцы POS-истории", "pos_max_dpd": "Максимальный DPD POS",
    "pos_late_fraction": "Доля просроченных POS-месяцев", "card_months": "Месяцы карточной истории", "card_max_dpd": "Максимальный DPD карт",
    "card_mean_utilization": "Среднее использование карточного лимита", "payment_events": "Исторические платёжные события",
    "payment_late_fraction": "Доля поздних платёжных событий", "payment_max_lateness": "Максимальное опоздание платежа",
    "payment_total_amount": "Историческая сумма платежей", "history_present": "Наличие исходной истории", "TARGET": "TARGET: трудности",
    "AMT_INCOME_TOTAL": "Годовой доход, у.е.", "AMT_CREDIT": "Сумма кредита, у.е.", "AMT_ANNUITY": "Месячный платёж, у.е.",
    "AMT_GOODS_PRICE": "Стоимость товара, у.е.", "DAYS_BIRTH": "Возраст в днях относительно заявки", "DAYS_EMPLOYED": "Стаж в днях относительно заявки",
    "NAME_CONTRACT_TYPE": "Продукт исходной заявки", "NAME_INCOME_TYPE": "Тип занятости"}
PALETTE = ["#1967D2", "#26A69A", "#F0AD4E", "#8C77D9", "#D95B68", "#78909C"]


def today() -> date:
    return datetime.now(timezone(timedelta(hours=5))).date()


def money(value) -> str:
    return f"{float(value or 0):,.2f}".replace(",", " ") + " у.е."


def percent(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.1%}"


def cfg_json(config) -> str:
    return json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)


def revision() -> int:
    return st.session_state.setdefault("data_revision", 0)


def invalidate(message: str | None = None) -> None:
    st.session_state["data_revision"] = revision() + 1
    st.cache_data.clear()
    if message:
        st.session_state["flash"] = message


@st.cache_data(ttl=60, show_spinner=False)
def cached_clients(serialized: str, search: str, rev: int):
    return db.list_clients(json.loads(serialized), search=search)


@st.cache_data(ttl=60, show_spinner=False)
def cached_snapshot(serialized: str, as_of: str, start: str, end: str, filters_json: str, rev: int):
    return analytics.snapshot(json.loads(serialized), as_of, filters=json.loads(filters_json), period_start=start, period_end=end)


@st.cache_resource(show_spinner=False)
def cached_model(artifact_dir: str, modified_ns: int, _config):
    # No SQLite connection is cached or shared between Streamlit threads.
    return models.load_model_bundle(_config)


@st.cache_data(show_spinner=False)
def cached_profile(serialized: str, source_id: int, modified_ns: int):
    return features.load_source_profile(json.loads(serialized), source_id)


@st.cache_data(show_spinner=False)
def cached_json(path: str, modified_ns: int):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def source_profile(config, source_id):
    path = Path(config["paths"]["processed_dir"]) / "application_features.parquet"
    return cached_profile(cfg_json(config), int(source_id), path.stat().st_mtime_ns if path.exists() else 0)


def model_bundle(config):
    path = Path(config["paths"]["artifact_dir"]) / "model_bundle.joblib"
    if not path.exists():
        raise FileNotFoundError("Обученная модель отсутствует. Сначала выполните команду обучения.")
    return cached_model(str(path.parent), path.stat().st_mtime_ns, config)


def train_command() -> str:
    return r".\.venv\Scripts\python.exe -m src.cli train"


def show_error(error: Exception, action="Операция не выполнена") -> None:
    LOGGER.exception(action, exc_info=error)
    st.error(f"{action}: {error}")


def table(records, *, key=None, limit=500):
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if frame.empty:
        st.info("Нет записей для выбранных условий.")
        return frame
    technical = {"schedule", "transactions", "payments", "inputs", "decision_details", "result", "operation_key", "feature_snapshot"}
    visible = frame.drop(columns=[column for column in frame if column in technical or str(column).endswith(("_json", "_cents"))], errors="ignore").copy()
    for column in visible.columns:
        if column == "product":
            visible[column] = visible[column].map(lambda value: PRODUCTS.get(value, value))
        elif column in ("status", "decision", "decision_status"):
            visible[column] = visible[column].map(lambda value: STATUSES.get(value, value))
        elif column == "source_kind":
            visible[column] = visible[column].map(lambda value: SOURCES.get(value, value))
        elif column == "client_type":
            visible[column] = visible[column].map(lambda value: {"New": "Новый", "Repeat": "Повторный"}.get(value, value))
        elif column == "kind":
            visible[column] = visible[column].map(lambda value: {"disbursement": "Выдача", "payment": "Платёж", "drawdown": "Выборка", "line_closure": "Закрытие линии"}.get(value, value))
        elif column == "variant":
            visible[column] = visible[column].map(lambda value: {"form": "Только анкета", "history": "Анкета + история"}.get(value, value))
        elif column == "direction":
            visible[column] = visible[column].map(lambda value: {"increases": "Повышает оценку риска", "decreases": "Понижает оценку риска"}.get(value, value))
        elif visible[column].map(lambda v: isinstance(v, (dict, list))).any():
            visible[column] = visible[column].map(lambda v: json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v)
    st.dataframe(visible.head(limit).rename(columns=LABELS), hide_index=True, width="stretch", key=key)
    if len(frame) > limit:
        st.caption(f"Показаны {limit:,} из {len(frame):,} строк. Экспорт содержит все отфильтрованные записи.")
    return frame


def client_label(client):
    mark = " · Демо-клиент" if client.get("is_synthetic") or client.get("source_application_id") else ""
    source = f" · источник {client['source_application_id']}" if client.get("source_application_id") else ""
    return f"#{client['client_id']} · {client['full_name']}{mark}{source}"


def choose_client(config, prefix, show_search=True):
    search = st.text_input("Поиск: ID, ФИО или исходная заявка", key=f"{prefix}_search") if show_search else ""
    clients = cached_clients(cfg_json(config), search, revision())
    if not clients:
        st.info("Клиентов не найдено. Создайте нового клиента или импортируйте демонстрационный профиль во вкладке «Клиенты».")
        return None
    by_id = {int(row["client_id"]): row for row in clients}
    selected = st.selectbox("Выберите клиента", list(by_id), index=None,
                            format_func=lambda key: client_label(by_id[key]),
                            placeholder="Явно выберите нужную запись", key=f"{prefix}_client")
    return by_id.get(selected)


def validate_inputs(config, inputs, full_name, selected_client):
    errors = []
    if selected_client is None and not full_name.strip():
        errors.append("Укажите ФИО нового клиента.")
    for key, label in (("monthly_income", "Месячный доход"), ("amount", "Сумма / лимит")):
        if inputs[key] <= 0:
            errors.append(f"{label} должен быть положительным.")
    product = config["products"][inputs["product"]]
    if not product["min_amount"] <= inputs["amount"] <= product["max_amount"]:
        errors.append(f"Допустимая сумма: {money(product['min_amount'])} — {money(product['max_amount'])}.")
    if not product["min_term_months"] <= inputs["term_months"] <= product["max_term_months"]:
        errors.append(f"Допустимый срок: {product['min_term_months']}—{product['max_term_months']} мес.")
    if not 0 <= inputs["annual_rate"] <= 100:
        errors.append("Номинальная ставка должна быть в пределах 0—100%.")
    if inputs["product"] == "revolving" and not 0 <= inputs["initial_draw"] <= inputs["amount"]:
        errors.append("Первоначальная выборка должна находиться между нулём и кредитным лимитом.")
    if inputs["employment_years"] is None:
        errors.append("Укажите стаж по анкете: в исходном профиле он неизвестен.")
    elif inputs["employment_years"] > max(0, inputs["age"] - 14):
        errors.append("Стаж не может превышать возраст за вычетом 14 лет.")
    if inputs["disbursement_date"] < inputs["application_date"]:
        errors.append("Предполагаемая выдача не может быть раньше заявки.")
    if not inputs.get("purpose", "").strip():
        errors.append("Укажите цель кредита.")
    return errors


def score_and_save(config, inputs, client, full_name):
    errors = validate_inputs(config, inputs, full_name, client)
    if errors:
        st.error("Исправьте исходные данные: " + " ".join(errors))
        return
    try:
        source_id = client.get("source_application_id") if client else None
        bundle = model_bundle(config)
        score = models.score_application(config, inputs, source_application_id=source_id, bundle=bundle)
        local_debt = ledger.local_monthly_debt(config, int(client["client_id"]), inputs["application_date"]) if client else 0.0
        decision = committee.evaluate_application(config, inputs, score, local_monthly_debt=local_debt)
        flow = st.session_state.setdefault("application_flow", uuid4().hex)
        if client is None:
            existing = st.session_state.get("flow_client")
            if existing and existing["flow"] == flow and existing["full_name"] == full_name.strip():
                client_id = existing["client_id"]
            else:
                client_id = db.create_client(config, full_name.strip(), source_kind="local", is_synthetic=False)
                st.session_state["flow_client"] = {"flow": flow, "full_name": full_name.strip(), "client_id": client_id}
        else:
            client_id = int(client["client_id"])
        fingerprint = hashlib.sha256(json.dumps({"inputs": inputs, "client_id": client_id}, sort_keys=True, default=str).encode()).hexdigest()
        operation = f"application:{flow}:{fingerprint}"
        application_id = db.submit_application(config, client_id, inputs, operation_key=operation, application_date=inputs["application_date"],
                                               source_kind="local", is_synthetic=False, source_application_id=source_id)
        # Repeated identical form submissions reuse the persisted decision; an explicit
        # re-evaluation in the register creates a new decision version.
        if not db.list_decisions(config, application_id):
            db.save_decision(config, application_id, decision)
        st.session_state["last_application_id"] = application_id
        invalidate()
        st.success(f"Заявка #{application_id} и решение сохранены. Выдача оформляется отдельным действием.")
    except FileNotFoundError as error:
        st.warning(str(error))
        st.code(train_command(), language="powershell")
    except Exception as error:
        show_error(error, "Заявка не оценена")


def render_application_form(config):
    st.subheader("Заявка в кредитный комитет")
    st.caption("Проверка платёжеспособности и модель TARGET=1. Модель не использует ФИО, идентификаторы, пол или возраст; возраст проверяется отдельным правилом допуска.")
    mode = st.radio("Клиент", ["Новый клиент", "Существующий клиент"], horizontal=True, key="app_client_mode")
    client = choose_client(config, "new_application") if mode == "Существующий клиент" else None
    if mode == "Существующий клиент" and client is None:
        return
    defaults = {}
    if client and client.get("source_application_id"):
        try:
            profile = source_profile(config, client["source_application_id"])
            defaults = profile.get("form_inputs", {})
            st.info("Демо-клиент: данные исходного профиля относятся к его исторической заявке. Новые условия в форме заменяют условия исходной заявки.")
        except Exception as error:
            st.warning(f"Исходный профиль не загружен: {error}. Доступна модель без истории.")
    elif client:
        st.info("Для местного клиента без сопоставимой исходной истории используется модель анкеты. Местные договоры учитываются в проверке долговой нагрузки; отсутствующие исторические признаки не выдумываются.")
    product = st.selectbox("Продукт", list(PRODUCTS), format_func=PRODUCTS.get, key="application_product")
    product_cfg = config["products"][product]
    selected_id = client["client_id"] if client else "new"
    with st.form(f"application_form_{selected_id}_{product}", clear_on_submit=False):
        identity = st.columns([2, 1, 1])
        full_name = identity[0].text_input("ФИО", value=client["full_name"] if client else "", disabled=bool(client))
        age = identity[1].number_input("Возраст, лет (только допуск)", min_value=0, max_value=110, value=int(defaults.get("age", 35)))
        application_date = identity[2].date_input("Дата заявки", value=today())
        left, middle, right = st.columns(3)
        monthly_income = left.number_input("Месячный доход, у.е.", min_value=0.0, value=float(defaults.get("monthly_income", 30000)), step=1000.0)
        existing_debt = middle.number_input("Все существующие платежи по долгам, у.е./мес.", min_value=0.0, value=0.0, step=100.0,
                                            help="Включите внешние и местные договоры. Используется максимум из заявленной общей суммы и известных платежей платформы, без повторного сложения. Архивное бюро не подменяет сегодняшние обязательства.")
        expenses = right.number_input("Необходимые расходы семьи, у.е./мес.", min_value=0.0, value=5000.0, step=100.0)
        left, middle, right = st.columns(3)
        employment_options = ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"]
        employment_names = {"Working": "Работа по найму", "Commercial associate": "Частный сектор", "Pensioner": "Пенсионер", "State servant": "Госслужащий",
                            "Unemployed": "Безработный", "Student": "Студент", "Businessman": "Предприниматель", "Maternity leave": "Отпуск по уходу"}
        default_employment = defaults.get("employment_type", "Working")
        employment = left.selectbox("Занятость", employment_options, index=employment_options.index(default_employment) if default_employment in employment_options else 0,
                                    format_func=lambda value: employment_names.get(value, value))
        known_tenure = defaults.get("employment_years", 5)
        employment_years = middle.number_input("Стаж на текущей работе, лет", min_value=0.0, max_value=80.0,
                                                value=None if known_tenure is None else min(80.0, max(0.0, float(known_tenure))), step=0.5,
                                                help="Если исходный стаж неизвестен, поле остаётся пустым. Укажите сведения из новой анкеты; пропуск не заменяется нулём.")
        term = right.number_input("Срок / горизонт сценария, мес.", min_value=1, max_value=120, value=24, step=1)
        left, middle, right = st.columns(3)
        amount = left.number_input("Запрашиваемая сумма, у.е." if product == "cash" else "Кредитный лимит, у.е.", min_value=0.0,
                                   value=min(200000.0, float(product_cfg["max_amount"])), step=1000.0)
        rate = middle.number_input("Номинальная годовая ставка, %", value=float(product_cfg["annual_rate"]), disabled=True,
                                   help="Берётся из конфигурационной таблицы продуктов; это не эффективная ставка.")
        intended_date = right.date_input("Предполагаемая дата выдачи", value=today())
        initial_draw = st.number_input("Первоначальная выборка, у.е.", min_value=0.0, value=50000.0, step=1000.0) if product == "revolving" else amount
        if product == "revolving":
            st.caption("Лимит и выбранный основной долг учитываются отдельно. Минимальный платёж и проценты рассчитываются по учебной политике кредитной линии.")
        purpose = st.text_input("Цель кредита", placeholder="Например: ремонт жилья")
        comments = st.text_area("Комментарий (не используется моделью)", height=70)
        st.caption("Месячный доход преобразуется в годовой AMT_INCOME_TOTAL ×12. Обязательства и новый платёж сравниваются в месячных единицах.")
        submitted = st.form_submit_button("Рассчитать и сохранить решение", type="primary", width="stretch")
    inputs = {"monthly_income": monthly_income, "amount": amount, "term_months": int(term), "annual_rate": rate,
              "product": product, "age": int(age), "employment_type": employment, "employment_years": employment_years,
              "existing_monthly_debt": existing_debt, "essential_expenses": expenses, "initial_draw": initial_draw,
              "application_date": application_date.isoformat(), "disbursement_date": intended_date.isoformat(),
              "intended_disbursement_date": intended_date.isoformat(), "purpose": purpose, "comments": comments}
    if submitted:
        with st.spinner("Рассчитываем условия и прогноз…"):
            score_and_save(config, inputs, client, full_name)
    if st.session_state.get("last_application_id"):
        decision_card(config, int(st.session_state["last_application_id"]), "latest")
        if st.button("Начать следующую заявку", key="reset_application"):
            st.session_state["application_flow"] = uuid4().hex
            st.session_state.pop("flow_client", None)
            st.session_state.pop("last_application_id", None)
            st.rerun()


def unpack(value, default=None):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default if default is not None else value
    return value if value is not None else default


def decision_payload(row):
    for key in ("result", "result_json", "decision_json", "payload_json", "details_json", "details", "decision_data"):
        if key in row:
            details = unpack(row[key], {})
            if isinstance(details, dict):
                return {**row, **details}
    return row


def decision_card(config, application_id: int, prefix: str):
    application = db.get_application(config, application_id)
    decisions = db.list_decisions(config, application_id)
    if not decisions:
        st.info(f"Заявка #{application_id}: окончательного решения пока нет.")
        return
    row = decision_payload(decisions[0])
    decision = row.get("decision", row.get("status", "Pending"))
    st.divider()
    st.subheader(f"{STATUSES.get(decision, decision)} · заявка #{application_id}")
    reasons = unpack(row.get("reasons", row.get("reasons_json")), [])
    if decision == "Approved":
        st.success("Комитет одобрил заявку. Портфель изменится только после отдельного оформления выдачи.")
    elif decision == "Rejected":
        st.error("Комитет отказал по условиям учебной политики.")
    probability = row.get("probability", row.get("risk_probability"))
    columns = st.columns(4)
    columns[0].metric("Вероятность платёжных трудностей", percent(probability))
    columns[1].metric("Новый месячный платёж", money(row.get("monthly_payment")))
    columns[2].metric("Доля долговых платежей / доход", percent(row.get("dti")))
    columns[3].metric("Остаток дохода после расходов", money(row.get("remaining_income")))
    st.write("Категория риска:", {"Low": "Низкий", "Elevated": "Повышенный", "High": "Высокий"}.get(row.get("risk_category"), row.get("risk_category", "не определена")))
    threshold = row.get("threshold", row.get("risk_threshold"))
    if threshold is not None:
        st.caption(f"Низкий риск < {percent(float(threshold) / 2)}; повышенный — от {percent(float(threshold) / 2)} до {percent(threshold)}; высокий — от {percent(threshold)}. Порог выбран на validation.")
    if isinstance(reasons, list):
        for reason in reasons:
            st.write("•", reason)
    elif reasons:
        st.write(reasons)
    inputs = unpack(application.get("inputs", application.get("inputs_json")), {})
    if inputs:
        st.caption(f"{PRODUCTS.get(inputs.get('product'), inputs.get('product'))} · {money(inputs.get('amount'))} · "
                   f"{inputs.get('term_months', '—')} мес. · номинальная ставка {inputs.get('annual_rate', '—')}%")
    st.caption(f"Модель: {row.get('model_version', '—')} · политика: {row.get('policy_version', '—')} · "
               f"оценено: {row.get('evaluation_date', row.get('evaluated_at', row.get('created_at', '—')))} · история: {'использована' if row.get('history_used') else 'не использована'}")
    contributions = unpack(row.get("contributions", row.get("contributions_json")), [])
    if contributions:
        with st.expander("Локальные вклады факторов в прогноз"):
            table(contributions if isinstance(contributions, list) else [{"Фактор": k, "Вклад": v} for k, v in contributions.items()])
            st.caption("Локальные вклады объясняют вычисление модели, а не доказанные причины поведения человека. Глобальная важность признаков здесь не подменяет локальное объяснение.")
    with st.expander("Аудит: версии решения и снимок признаков"):
        table([{k: v for k, v in item.items() if k not in ("feature_snapshot", "feature_snapshot_json", "inputs_json")} for item in decisions])
        st.json(unpack(row.get("feature_snapshot", row.get("feature_snapshot_json")), {}))
        st.json(row)
    if application.get("agreement_id"):
        st.info(f"По заявке оформлен договор #{application['agreement_id']}. Операции доступны во вкладке «Кредиты и платежи».")
    elif decision == "Approved":
        with st.form(f"disbursement_{prefix}_{application_id}"):
            disb_date = st.date_input("Дата фактической учебной выдачи", value=date.fromisoformat(inputs.get("disbursement_date", today().isoformat())), key=f"disb_date_{prefix}_{application_id}")
            submitted = st.form_submit_button("Оформить выдачу", type="primary")
        if submitted:
            try:
                agreement_id = ledger.disburse(config, application_id, disb_date.isoformat(), f"disbursement:{application_id}")
                invalidate(f"Выдача оформлена. Договор #{agreement_id}.")
                st.rerun()
            except Exception as error:
                show_error(error, "Выдача не оформлена")


def render_clients(config):
    st.subheader("Клиенты и происхождение истории")
    with st.expander("Добавить клиента без заявки"):
        with st.form("create_client"):
            name = st.text_input("ФИО нового клиента", key="create_client_name")
            create = st.form_submit_button("Создать клиента")
        if create:
            if not name.strip():
                st.error("Укажите ФИО.")
            else:
                try:
                    fingerprint = hashlib.sha256(name.strip().encode()).hexdigest()
                    token = st.session_state.setdefault("create_client_token", uuid4().hex)
                    done = st.session_state.setdefault("created_client_operations", {})
                    operation = f"{token}:{fingerprint}"
                    if operation not in done:
                        done[operation] = db.create_client(config, name.strip(), source_kind="local", is_synthetic=False)
                    invalidate(f"Создан клиент #{done[operation]}. Совпадение ФИО не объединяет разных людей.")
                    st.rerun()
                except Exception as error:
                    show_error(error)
        if st.button("Подготовить создание другого клиента", key="next_client_creation"):
            st.session_state["create_client_token"] = uuid4().hex
            st.info("Следующее создание будет отдельной записью, даже при совпадающем ФИО.")
    with st.expander("Импортировать исходный демонстрационный профиль"):
        st.caption("В Home Credit нет ФИО. Импортируемой записи присваивается вымышленное имя и пометка «Демо-клиент». Исходная заявка не является универсальным идентификатором человека.")
        with st.form("import_source_client"):
            source_id = st.number_input("SK_ID_CURR исходной заявки", min_value=1, value=100002, step=1)
            import_profile = st.form_submit_button("Загрузить демо-профиль")
        if import_profile:
            try:
                source_profile(config, int(source_id))
                linked = [row for row in db.list_clients(config, str(source_id)) if row.get("source_application_id") == int(source_id)]
                if linked:
                    st.info(f"Профиль уже связан с клиентом #{linked[0]['client_id']}; повторный импорт не выполнен.")
                else:
                    client_id = db.create_client(config, f"Демо-клиент {source_id}", source_application_id=int(source_id), source_kind="home_credit", is_synthetic=True)
                    invalidate(f"Импортирован демонстрационный клиент #{client_id}.")
                    st.rerun()
            except Exception as error:
                show_error(error, "Профиль не импортирован")
    client = choose_client(config, "clients")
    if client:
        st.write(client_label(client))
        as_of = st.date_input("Состояние местной истории на дату", value=today(), key="client_asof")
        render_client_history(config, client, as_of.isoformat())


@st.cache_data(ttl=300, show_spinner=False)
def historical_rows(database_path: str, source_id: int, source_table: str, modified_ns: int):
    allowed = {"bureau", "previous_application", "installments_payments", "POS_CASH_balance", "credit_card_balance"}
    if source_table not in allowed:
        raise ValueError("Неизвестная таблица источника")
    path = Path(database_path)
    if not path.exists():
        return pd.DataFrame()
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (source_table,)).fetchone()
        if not exists:
            return pd.DataFrame()
        cutoff = {"bureau": "DAYS_CREDIT<=0 AND DAYS_CREDIT_UPDATE<=0", "previous_application": "DAYS_DECISION<=0",
                  "installments_payments": "DAYS_ENTRY_PAYMENT<=0 AND DAYS_INSTALMENT<=0",
                  "POS_CASH_balance": "MONTHS_BALANCE<=0", "credit_card_balance": "MONTHS_BALANCE<=0"}[source_table]
        return pd.read_sql_query(f'SELECT * FROM "{source_table}" WHERE SK_ID_CURR=? AND {cutoff} LIMIT 200', connection, params=(source_id,))


def render_client_history(config, client, as_of):
    local = cached_snapshot(cfg_json(config), as_of, "1900-01-01", as_of,
                            json.dumps({"client_id": int(client["client_id"])}), revision())
    st.markdown("**Операции платформы**")
    st.caption(f"Состояние на {as_of}; реальные операции пользователя и синтетический демо-портфель имеют отдельные признаки происхождения.")
    if isinstance(local, dict):
        for name, label in (("applications", "Заявки"), ("agreements", "Договоры и долги"), ("payments", "Фактические платежи")):
            with st.expander(label, expanded=name == "agreements"):
                table(local.get(name, []))
    else:
        table(local)
    source_id = client.get("source_application_id")
    st.markdown("**Исходный исторический профиль Home Credit**")
    if not source_id:
        st.info("Исходная история отсутствует. Это не означает безупречные платежи; модель использует вариант без истории.")
        return
    st.warning("Архивный снимок на момент исходной заявки, абсолютная дата неизвестна. Это не актуальная проверка бюро. Внешние долги не входят в собственный портфель платформы.")
    try:
        profile = source_profile(config, source_id)
        with st.expander("Агрегированные признаки и момент отсечения"):
            st.write("Момент отсечения: день и месяц 0 относительно исходной заявки. Её абсолютная календарная дата неизвестна; фактические события после отсечения исключены из признаков модели.")
            st.json(profile.get("history", {}))
        path = Path(config["paths"]["analytics_database"])
        for source_table, label in (("bureau", "Внешние кредиты бюро"), ("previous_application", "Предыдущие заявки Home Credit"),
                                    ("installments_payments", "Исторические запланированные и фактические платежи"),
                                    ("POS_CASH_balance", "POS и денежные кредиты: месячные снимки"), ("credit_card_balance", "Карты: месячные снимки")):
            with st.expander(label):
                data = historical_rows(str(path), int(source_id), source_table, path.stat().st_mtime_ns if path.exists() else 0)
                table(data, limit=200)
                if data.empty:
                    st.caption("Нет совпавших записей либо аналитический импорт ещё не выполнен.")
                st.caption(f"Источник: {source_table}. Не более 200 исходных строк; платежи не позволяют точно восстановить разделение основного долга и процентов.")
    except Exception as error:
        show_error(error, "История источника не загружена")


def render_applications(config):
    st.subheader("Реестр заявок и решений")
    rows = db.list_applications(config)
    frame = pd.DataFrame(rows)
    status = st.selectbox("Решение комитета", ["Все", "Approved", "Rejected", "Pending"], format_func=lambda value: STATUSES.get(value, value), key="applications_status")
    if not frame.empty and status != "Все":
        column = "decision" if "decision" in frame else "status"
        frame = frame[frame[column] == status]
    table(frame)
    if frame.empty:
        return
    selected = st.selectbox("Заявка для просмотра", frame["application_id"].tolist(), key="application_details")
    decision_card(config, int(selected), "register")
    selected_application = db.get_application(config, int(selected))
    if st.button("Повторно оценить: сохранить новую версию решения", key="reevaluate_application", disabled=bool(selected_application.get("agreement_id"))):
        try:
            application = db.get_application(config, int(selected))
            inputs = unpack(application.get("inputs", application.get("inputs_json")), {})
            client = db.get_client(config, int(application["client_id"]))
            score = models.score_application(config, inputs, source_application_id=client.get("source_application_id"), bundle=model_bundle(config))
            local_debt = ledger.local_monthly_debt(config, int(client["client_id"]), inputs.get("application_date", application.get("application_date")))
            result = committee.evaluate_application(config, inputs, score, local_monthly_debt=local_debt)
            db.save_decision(config, int(selected), result, operation_key=f"reevaluation:{selected}:{uuid4().hex}")
            invalidate("Сохранена новая версия решения; прежние версии доступны в аудите.")
            st.rerun()
        except Exception as error:
            show_error(error, "Повторная оценка не выполнена")


def operation_key(kind, agreement_id, amount, operation_date):
    token = st.session_state.setdefault(f"operation_token_{kind}_{agreement_id}", uuid4().hex)
    return f"{kind}:{agreement_id}:{operation_date}:{float(amount):.2f}:{token}"


def render_loan_actions(config, agreement_id, snapshot, as_of):
    status = snapshot.get("status")
    if status == "Closed":
        st.success("Договор закрыт. Новые платежи и выборки не требуются.")
        return
    with st.form(f"payment_{agreement_id}"):
        st.markdown("**Зарегистрировать фактический платёж**")
        left, right = st.columns(2)
        amount = left.number_input("Сумма платежа, у.е.", min_value=0.0, value=1000.0, step=100.0, key=f"payment_amount_{agreement_id}")
        payment_date = right.date_input("Дата платежа", value=date.fromisoformat(as_of), key=f"payment_date_{agreement_id}")
        submit = st.form_submit_button("Зарегистрировать платёж", type="primary")
    if submit:
        try:
            if amount <= 0:
                raise ValueError("Сумма платежа должна быть положительной")
            result = ledger.record_payment(config, agreement_id, amount, payment_date.isoformat(), operation_key("payment", agreement_id, amount, payment_date))
            invalidate(f"Платёж {money(result.get('amount', amount))} зарегистрирован: основной долг {money(result.get('principal'))}, проценты {money(result.get('interest'))}.")
            st.rerun()
        except Exception as error:
            show_error(error, "Платёж не проведён")
    if st.button("Подготовить следующий платёж", key=f"new_payment_{agreement_id}"):
        st.session_state[f"operation_token_payment_{agreement_id}"] = uuid4().hex
        st.success("Подготовлена новая операция; одинаковые сумма и дата теперь могут быть зарегистрированы отдельно.")
    with st.expander("Досрочное полное погашение"):
        payoff_date = st.date_input("Дата полного расчёта", value=date.fromisoformat(as_of), key=f"settle_date_{agreement_id}")
        try:
            quote = ledger.settlement_quote(config, agreement_id, payoff_date.isoformat())
            if isinstance(quote, dict):
                columns = st.columns(3)
                columns[0].metric("Полный расчёт на дату", money(quote.get("amount")))
                columns[1].metric("Из них основной долг", money(quote.get("principal")))
                columns[2].metric("Из них начисленные проценты", money(quote.get("interest")))
            total = float(quote.get("total", quote.get("amount", quote.get("settlement_amount", 0)))) if isinstance(quote, dict) else float(quote)
            if st.button("Погасить полностью по расчёту", key=f"settle_{agreement_id}", disabled=total <= 0):
                ledger.record_payment(config, agreement_id, total, payoff_date.isoformat(), operation_key("settlement", agreement_id, total, payoff_date))
                invalidate("Полное погашение зарегистрировано. Для кредитной линии при нулевом долге требуется отдельное закрытие.")
                st.rerun()
        except Exception as error:
            show_error(error, "Сумма полного расчёта недоступна")
    if snapshot.get("product") == "revolving":
        with st.form(f"drawdown_{agreement_id}"):
            st.markdown("**Дополнительная выборка по кредитной линии**")
            amount_draw = st.number_input("Дополнительная сумма, у.е.", min_value=0.0, value=1000.0, key=f"draw_amount_{agreement_id}")
            draw_date = st.date_input("Дата выборки", value=date.fromisoformat(as_of), key=f"draw_date_{agreement_id}")
            draw = st.form_submit_button("Зарегистрировать выборку")
        if draw:
            try:
                ledger.drawdown(config, agreement_id, amount_draw, draw_date.isoformat(), operation_key("drawdown", agreement_id, amount_draw, draw_date))
                invalidate("Выборка зарегистрирована в пределах лимита.")
                st.rerun()
            except Exception as error:
                show_error(error, "Выборка не проведена")
        if st.button("Подготовить следующую выборку", key=f"new_drawdown_{agreement_id}"):
            st.session_state[f"operation_token_drawdown_{agreement_id}"] = uuid4().hex
            st.success("Подготовлена новая операция выборки.")
        if st.button("Закрыть кредитную линию при нулевых обязательствах", key=f"close_line_{agreement_id}"):
            try:
                ledger.close_line(config, agreement_id, as_of, f"close-line:{agreement_id}:{as_of}")
                invalidate("Кредитная линия закрыта.")
                st.rerun()
            except Exception as error:
                show_error(error, "Кредитная линия не закрыта")


def render_loans(config):
    st.subheader("Договоры, графики и фактические платежи")
    as_of = st.date_input("Состояние договоров на дату", value=today(), key="loans_asof").isoformat()
    snap = cached_snapshot(cfg_json(config), as_of, "1900-01-01", as_of, "{}", revision())
    rows = pd.DataFrame(snap.get("agreements", []))
    left, right = st.columns(2)
    product = left.selectbox("Продукт договора", ["Все"] + list(PRODUCTS), format_func=lambda value: PRODUCTS.get(value, value), key="loans_product")
    status = right.selectbox("Статус договора", ["Все", "Active", "Closed"], format_func=lambda value: STATUSES.get(value, value), key="loans_status")
    if not rows.empty:
        if product != "Все":
            rows = rows[rows["product"] == product]
        if status != "Все":
            rows = rows[rows["status"] == status]
    table(rows)
    if rows.empty:
        return
    selected = st.selectbox("Договор для просмотра и операций", rows["agreement_id"].tolist(), key="loan_details")
    snapshot = ledger.agreement_snapshot(config, int(selected), as_of)
    cards = st.columns(4)
    cards[0].metric("Основной долг", money(snapshot.get("outstanding_principal")))
    cards[1].metric("Неуплаченные проценты", money(snapshot.get("accrued_interest")))
    cards[2].metric("Просроченные обязательства", money(snapshot.get("overdue_amount")))
    cards[3].metric("Текущий DPD", f"{snapshot.get('dpd', 0)} дней")
    if snapshot.get("product") == "revolving":
        st.info(f"Лимит: {money(snapshot.get('credit_limit'))} · доступно: {money(snapshot.get('available_credit'))}. Нулевой остаток долга автоматически не закрывает линию.")
        st.caption("Проценты: простое ежедневное начисление ACT/365 на выбранный основной долг; месячные выписки. Минимальный платёж — доля основного долга плюс начисленные проценты. Ниже фактические сформированные обязательства, без вымышленных будущих выборок.")
    else:
        st.caption("Равные аннуитетные платежи; последний платёж корректируется на округление. Учебное начисление процентов — в месячную дату платежа на фактический основной долг, без ежедневных промежуточных процентов. Оплата: начисленные проценты → основной долг от старейших обязательств; будущие неначисленные проценты при досрочном погашении не взыскиваются.")
    st.markdown("**График обязательств и состояние оплаты**")
    schedule = snapshot.get("schedule", [])
    table(schedule)
    schedule_frame = pd.DataFrame(schedule)
    if not schedule_frame.empty:
        payment_column = next((key for key in ("payment", "total_payment", "scheduled_payment", "amount") if key in schedule_frame), None)
        if payment_column:
            total = float(schedule_frame[payment_column].sum())
            st.write("Всего по текущему графику:", money(total))
            if "principal" in schedule_frame:
                st.write("Сумма сверх основного долга по графику:", money(total - schedule_frame["principal"].sum()))
    st.download_button("Скачать график CSV", csv_bytes(schedule, report_metadata(as_of, "1900-01-01", as_of, {"agreement_id": int(selected)})),
                       file_name=f"schedule_{selected}_{as_of}.csv", mime="text/csv", key="download_schedule")
    with st.expander("Фактические платежи и операции", expanded=True):
        table(snapshot.get("payments", snapshot.get("transactions", [])))
    render_loan_actions(config, int(selected), snapshot, as_of)


def render_operations(config):
    tabs = st.tabs(["Новая заявка", "Клиенты", "Заявки", "Кредиты и платежи"])
    for tab, renderer in zip(tabs, (render_application_form, render_clients, render_applications, render_loans)):
        with tab:
            renderer(config)


def analytics_filters(config):
    st.subheader("Параметры отчёта")
    reference = date.fromisoformat(config["demo"]["reference_date"])
    with st.form("analytics_filters"):
        row = st.columns(3)
        as_of = row[0].date_input("Отчётная дата (снимок)", value=max(reference, today()), key="analytics_asof")
        start = row[1].date_input("Период операций: с", value=reference - timedelta(days=365), key="analytics_start")
        end = row[2].date_input("Период операций: по", value=max(reference, today()), key="analytics_end")
        row = st.columns(4)
        product = row[0].selectbox("Продукт", ["Все"] + list(PRODUCTS), format_func=lambda value: PRODUCTS.get(value, value), key="analytics_product")
        status = row[1].selectbox("Статус договора", ["Все", "Active", "Closed"], format_func=lambda value: STATUSES.get(value, value), key="analytics_status")
        client_type = row[2].selectbox("Клиент", ["Все", "New", "Repeat"], format_func=lambda value: {"New": "Новый", "Repeat": "Повторный"}.get(value, value), key="analytics_client_type")
        provenance = row[3].selectbox("Происхождение операций", ["Все", "local", "synthetic", "home_credit"], format_func=lambda value: SOURCES.get(value, value), key="analytics_source")
        submitted = st.form_submit_button("Применить фильтры", type="primary")
    if "analytics_selection" not in st.session_state or submitted:
        if start > end:
            st.error("Начало периода не может быть позже окончания.")
        elif end > as_of:
            st.error("Период операций не может заканчиваться после отчётной даты.")
        else:
            filters = {key: value for key, value in (("product", product), ("status", status), ("client_type", client_type), ("source_kind", provenance)) if value != "Все"}
            st.session_state["analytics_selection"] = {"as_of": as_of.isoformat(), "start": start.isoformat(), "end": end.isoformat(), "filters": filters}
    return st.session_state.get("analytics_selection")


def donut(frame, category, value, title, unit="договоров", key=None):
    st.markdown(f"**{title}**")
    if frame.empty or category not in frame or value not in frame:
        st.info("Нет данных для диаграммы.")
        return
    data = frame.groupby(category, dropna=False)[value].sum().reset_index()
    data[value] = pd.to_numeric(data[value], errors="coerce").fillna(0)
    data = data[data[value] > 0].sort_values(value, ascending=False)
    if data.empty:
        st.info("Сумма равна нулю; доли не определены.")
        return
    if len(data) > 8:
        other = pd.DataFrame([{category: "Прочее", value: data.iloc[7:][value].sum()}])
        data = pd.concat([data.head(7), other], ignore_index=True)
    figure = px.pie(data, names=category, values=value, hole=.64, color_discrete_sequence=PALETTE)
    figure.update_traces(textinfo="percent", hovertemplate=f"%{{label}}<br>%{{value:,.2f}} {unit}<br>%{{percent}}<extra></extra>")
    figure.update_layout(height=300, margin=dict(l=4, r=4, t=15, b=5), legend=dict(orientation="h", y=-.12), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, use_container_width=True, key=key)
    st.caption(f"Основа расчёта: {data[value].sum():,.2f} {unit}. Доли рассчитаны только в выбранной совокупности.")


def decisions_frame(applications):
    frame = pd.DataFrame(applications)
    if frame.empty:
        return frame
    column = next((key for key in ("decision", "decision_status", "status") if key in frame), None)
    if column is None:
        return pd.DataFrame()
    frame = frame[frame[column].isin(["Approved", "Rejected"])].copy()
    frame["Решение"] = frame[column].map(STATUSES)
    frame["count"] = 1
    return frame


def render_portfolio(snapshot, selection):
    kpis = snapshot.get("kpis", {})
    specifications = [("Клиенты", "clients", "client_count"), ("Заявки", "applications", "application_count"),
                      ("Одобрено", "approvals", "approved"), ("Отказано", "rejections", "rejected"),
                      ("Выдано за период", "disbursed_count", "loans_disbursed"), ("Активные договоры", "active_agreements", "active_count"),
                      ("Закрытые договоры", "closed_agreements", "closed_count"), ("Договоры с просрочкой", "agreements_in_arrears", "overdue_count")]
    for offset in range(0, len(specifications), 4):
        columns = st.columns(4)
        for column, (label, first, second) in zip(columns, specifications[offset:offset + 4]):
            column.metric(label, f"{kpis.get(first, kpis.get(second, 0)):,}")
    financial = [("Основной долг на отчётную дату", "active_portfolio_balance", "portfolio_balance"),
                 ("Фактически выдано за период", "disbursed_amount", "amount_disbursed"),
                 ("Получено платежей за период", "payments_total", "payments_received"), ("Просроченные обязательства", "overdue_amount", "arrears_amount")]
    columns = st.columns(4)
    for column, (label, first, second) in zip(columns, financial):
        column.metric(label, money(kpis.get(first, kpis.get(second, 0))))
    columns = st.columns(4)
    columns[0].metric("Погашено основного долга за период", money(kpis.get("principal_received")))
    columns[1].metric("Получено процентов за период", money(kpis.get("interest_received")))
    columns[2].metric("Средняя фактическая выдача за период", "—" if kpis.get("average_disbursed_amount") is None else money(kpis["average_disbursed_amount"]))
    columns[3].metric("Средний риск оценённых заявок", percent(kpis.get("average_risk")))
    st.caption(f"Завершённых решений: {kpis.get('decisions', 0)} · ожидают решения: {kpis.get('pending', 0)}. "
               "Средняя выдача рассчитана по фактическим выдачам периода; средний риск — по заявкам выбранной совокупности с сохранённым прогнозом.")
    st.caption(f"Остатки и статусы — на {selection['as_of']}. Операции — с {selection['start']} по {selection['end']}. "
               "Основной долг = выдачи и выборки минус погашенный основной долг. Внешние долги бюро исключены.")
    left, right = st.columns(2)
    with left:
        decided = decisions_frame(snapshot.get("applications", []))
        donut(decided, "Решение", "count", "Одобрения и отказы среди завершённых решений", "заявок", "decision_donut")
        approved = int((decided["Решение"] == "Одобрено").sum()) if not decided.empty else 0
        st.write("Доля одобрений:", percent(approved / len(decided) if len(decided) else None))
        st.caption("Одобрено / (одобрено + отказано); ожидающие решения не входят в знаменатель.")
    agreements = pd.DataFrame(snapshot.get("agreements", []))
    with right:
        if not agreements.empty:
            agreements = agreements.copy()
            agreements["Статус"] = agreements["status"].map(STATUSES).fillna(agreements["status"])
            agreements["count"] = 1
        donut(agreements, "Статус", "count", "Активные и закрытые договоры на дату", "договоров", "status_donut")
    left, right = st.columns(2)
    with left:
        client_types = pd.DataFrame(snapshot.get("client_type_summary", []))
        if not client_types.empty:
            client_types["Тип клиента"] = client_types["client_type"].map({"New": "Новый", "Repeat": "Повторный"}).fillna(client_types["client_type"])
        donut(client_types, "Тип клиента", "count", "Новые и повторные клиенты на отчётную дату", "клиентов", "repeat_donut")
        st.caption("Повторный: не менее двух фактически профинансированных местных договоров к отчётной дате. Новый: ноль или один. Каждый клиент учитывается один раз; связь с Home Credit не заменяет выдачу платформы.")
    with right:
        if not agreements.empty:
            dpd = pd.to_numeric(agreements["dpd"], errors="coerce").fillna(0)
            agreements["DPD"] = pd.cut(dpd, bins=[-1, 0, 30, 60, 90, float("inf")], labels=["0", "1–30", "31–60", "61–90", "91+"])
            grouped = agreements.groupby("DPD", observed=False).size().rename("Договоров").reset_index()
            st.plotly_chart(px.bar(grouped, x="DPD", y="Договоров", title="Текущая просрочка — DPD", color_discrete_sequence=PALETTE), use_container_width=True, key="dpd_distribution")
        else:
            st.info("Нет договоров для распределения DPD.")
    render_timeseries(snapshot)
    with st.expander("Определения и все рассчитанные показатели"):
        st.caption("Просрочка — неуплаченные обязательства с прошедшей датой платежа. DPD — от старейшего непогашенного срока, возвращается к нулю после погашения просрочки. Полученные проценты — доход до расходов, а не чистая прибыль.")
        table([{"Показатель": KPI_LABELS.get(key, key), "Значение": readable_value(key, value)} for key, value in kpis.items()])
    if snapshot.get("sql_results"):
        with st.expander("Проверяемые аналитические выборки SQL"):
            for name, result in snapshot["sql_results"].items():
                st.markdown(f"**{name}**")
                table(result)


def render_timeseries(snapshot):
    timeline = snapshot.get("activity_series", snapshot.get("timeseries", snapshot.get("time_series", [])))
    if timeline:
        frame = pd.DataFrame(timeline)
        x = next((key for key in ("date", "month", "period", "transaction_date") if key in frame), None)
        numeric = [column for column in frame.select_dtypes(include="number").columns if column not in ("agreement_id", "client_id")]
        if x and numeric:
            labels = {"month": "Месяц", "value": "Сумма, у.е.", "variable": "Показатель", "disbursed": "Выдано", "payments": "Платежи", "principal": "Погашение основного долга", "interest": "Полученные проценты", "outstanding_principal": "Остаток основного долга"}
            display_frame = frame.rename(columns={column: labels.get(column, column) for column in frame.columns})
            st.plotly_chart(px.line(display_frame, x=labels.get(x, x), y=[labels.get(column, column) for column in numeric], title="Выдачи, платежи и долг во времени", labels={"value": "Сумма, у.е.", "variable": "Показатель"}, color_discrete_sequence=PALETTE), use_container_width=True, key="portfolio_timeseries")
            st.caption("Выдачи и платежи — потоки за соответствующий месяц; основной долг — остаток на конец месяца (для последнего месяца — не позже отчётной даты). Учтён входящий остаток до начала выбранного периода.")
            return
    payments = pd.DataFrame(snapshot.get("payments", []))
    if not payments.empty:
        x = next((key for key in ("payment_date", "transaction_date", "date", "occurred_on") if key in payments), None)
        amount = next((key for key in ("amount", "payment_total", "total") if key in payments), None)
        if x and amount:
            payments["Месяц"] = pd.to_datetime(payments[x]).dt.to_period("M").astype(str)
            data = payments.groupby("Месяц")[amount].sum().reset_index()
            st.plotly_chart(px.bar(data, x="Месяц", y=amount, title="Фактические платежи за период, у.е.", color_discrete_sequence=PALETTE), use_container_width=True, key="payments_timeseries")


@st.cache_data(ttl=300, show_spinner=False)
def historical_goods(database_path, modified_ns):
    path = Path(database_path)
    if not path.exists():
        return pd.DataFrame()
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='previous_application'").fetchone()
        if not exists:
            return pd.DataFrame()
        return pd.read_sql_query("SELECT NAME_GOODS_CATEGORY AS category, COUNT(*) AS applications FROM previous_application GROUP BY NAME_GOODS_CATEGORY ORDER BY applications DESC", connection)


def render_products(config, snapshot):
    agreements = pd.DataFrame(snapshot.get("agreements", []))
    mode = st.radio("Состав активного собственного портфеля", ["Количество активных договоров", "Остаток основного долга"], horizontal=True, key="portfolio_composition")
    if not agreements.empty:
        agreements = agreements[agreements["status"] == "Active"].copy()
        agreements["Продукт"] = agreements["product"].map(PRODUCTS).fillna(agreements["product"])
        agreements["count"] = 1
    donut(agreements, "Продукт", "count" if mode == "Количество активных договоров" else "outstanding_principal", "Активный собственный портфель по продуктам", "активных договоров" if mode == "Количество активных договоров" else "у.е.", "product_donut")
    if not agreements.empty:
        revolving = agreements[agreements["product"] == "revolving"]
        cols = st.columns(3)
        cols[0].metric("Лимиты кредитных линий", money(revolving["credit_limit"].sum()))
        cols[1].metric("Выбранный основной долг", money(revolving["outstanding_principal"].sum()))
        cols[2].metric("Доступный кредит", money(revolving["available_credit"].sum()))
    st.caption("Основа: только активные договоры на отчётную дату в выбранных фильтрах; закрытые договоры исключены. Кредитный лимит не равен выдаче. Исторические POS, карты, ипотека и другие продукты показываются отдельно.")
    st.divider()
    path = Path(config["paths"]["analytics_database"])
    goods = historical_goods(str(path), path.stat().st_mtime_ns if path.exists() else 0)
    donut(goods, "category", "applications", "Категории финансируемых товаров — исходные предыдущие заявки", "исторических заявок", "goods_donut")
    st.caption("Источник: previous_application.NAME_GOODS_CATEGORY, весь исходный исторический снимок. Это заявки, включая неуспешные; они не означают состоявшиеся покупки или выдачи. Фильтры местного портфеля к этому отдельному источнику не применяются. x-sell / walk-in — тип предложения, а не категория товара.")


def render_application_activity(snapshot):
    applications = pd.DataFrame(snapshot.get("applications", []))
    st.caption("Реестр ограничен периодом заявок и выбранными фильтрами. Показывается последнее решение на отчётную дату.")
    table(applications)
    if not applications.empty:
        probability = next((key for key in ("risk_probability", "probability") if key in applications), None)
        if probability:
            frame = applications.dropna(subset=[probability])
            if not frame.empty:
                figure = px.histogram(frame, x=probability, nbins=20, title="Распределение оценённого риска среди заявок с прогнозом", color_discrete_sequence=PALETTE)
                figure.update_xaxes(tickformat=".0%")
                st.plotly_chart(figure, use_container_width=True, key="risk_histogram")


@st.cache_data(show_spinner=False)
def numeric_eda_sample(path: str, column: str, modified_ns: int):
    values = pd.to_numeric(pd.read_parquet(path, columns=[column])[column], errors="coerce").dropna()
    sentinel_count = int((values == 365243).sum()) if column == "DAYS_EMPLOYED" else 0
    if column == "DAYS_EMPLOYED":
        values = values[values != 365243]
    sample = values.sample(n=min(20000, len(values)), random_state=42)
    return {"records": pd.DataFrame({"Значение": sample}).reset_index(drop=True),
            "observed": len(values), "sentinel_count": sentinel_count}


@st.cache_data(show_spinner=False)
def cached_correlations(path: str, modified_ns: int):
    return pd.read_parquet(path)


def render_eda(config):
    folder = Path(config["paths"]["processed_dir"])
    profile_path = folder / "profile.json"
    if not profile_path.exists():
        st.info("Исследовательский профиль исходных данных появится после подготовки.")
        return
    profile = cached_json(str(profile_path), profile_path.stat().st_mtime_ns)
    with st.expander("EDA: баланс TARGET, пропуски, описательные статистики и распределения"):
        st.caption("Фактически рассчитанный профиль исходных CSV. Это исследовательский анализ; синтетический местный портфель сюда не входит.")
        balance = pd.DataFrame([{"Класс": "0 — все остальные случаи" if str(key) == "0" else "1 — платёжные трудности", "Количество": value}
                                for key, value in profile.get("target_balance", {}).items()])
        if not balance.empty:
            donut(balance, "Класс", "Количество", "Баланс целевой переменной application_train", "исходных заявок", "eda_target")
        train = profile.get("tables", {}).get("application_train", {})
        missing = pd.DataFrame([{"Признак": key, "Пропущено": value, "Доля пропусков": value / train["rows"] if train.get("rows") else None}
                                for key, value in train.get("missing", {}).items()]).sort_values("Пропущено", ascending=False)
        if not missing.empty:
            figure = px.bar(missing.head(15).sort_values("Пропущено"), x="Доля пропусков", y="Признак", orientation="h",
                            title="15 признаков с наибольшей долей пропусков", color_discrete_sequence=PALETTE)
            figure.update_xaxes(tickformat=".0%")
            st.plotly_chart(figure, use_container_width=True, key="eda_missing")
        numeric = train.get("numeric", {})
        candidates = [key for key in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED") if key in numeric]
        if candidates:
            selected = st.selectbox("Числовой признак для исследования", candidates, format_func=lambda value: FEATURE_NAMES.get(value, value), key="eda_numeric")
            stats = numeric[selected]
            table([{"Признак": FEATURE_NAMES.get(selected, selected), "Наблюдений": stats.get("count"), "Среднее": stats.get("mean"),
                    "Стандартное отклонение": stats.get("std"), "Минимум": stats.get("min"), "Максимум": stats.get("max"),
                    "Наблюдений вне границ IQR": stats.get("outliers")}])
            st.caption(f"Описательные статистики рассчитаны по всем известным значениям исходного столбца до очистки. "
                       f"Диагностические границы выбросов 1.5×IQR оценены по первым 5 000 строкам: {stats.get('outlier_bounds')}. "
                       "Это признак для проверки, а не основание автоматически удалять строки; первая часть файла может быть нерепрезентативной.")
            parquet = folder / "application_train.parquet"
            if parquet.exists():
                sample = numeric_eda_sample(str(parquet), selected, parquet.stat().st_mtime_ns)
                figure = px.histogram(sample["records"], x="Значение", nbins=50, title=FEATURE_NAMES.get(selected, selected), color_discrete_sequence=PALETTE)
                figure.update_yaxes(title="Наблюдений в выборке")
                st.plotly_chart(figure, use_container_width=True, key="eda_numeric_distribution")
                st.caption(f"Гистограмма: воспроизводимая случайная выборка {len(sample['records']):,} из {sample['observed']:,} известных значений, seed=42. "
                           f"Для DAYS_EMPLOYED служебное значение 365243 исключено ({sample['sentinel_count']:,} строк); модель трактует его как пропуск.")
        category = train.get("categories", {}).get("NAME_CONTRACT_TYPE", {})
        if category:
            table([{"Исходный продукт": {"Cash loans": "Денежный кредит", "Revolving loans": "Возобновляемый кредит"}.get(key, key), "Заявок": value} for key, value in category.items()])
        if not missing.empty:
            table(missing)
    with st.expander("Корреляции числовых признаков и TARGET"):
        path = folder / "correlations.parquet"
        if path.exists():
            correlations = cached_correlations(str(path), path.stat().st_mtime_ns)
            if "TARGET" in correlations:
                top = correlations["TARGET"].drop(labels="TARGET", errors="ignore").dropna().abs().sort_values(ascending=False).head(10).index.tolist()
                selected = correlations.loc[top + ["TARGET"], top + ["TARGET"]].rename(index=FEATURE_NAMES, columns=FEATURE_NAMES)
                figure = px.imshow(selected, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto", title="Корреляции: 10 наиболее связанных с TARGET признаков")
                figure.update_layout(height=600, xaxis_tickangle=-45, coloraxis_colorbar_title="Пирсон r")
                st.plotly_chart(figure, use_container_width=True, key="eda_correlations")
                table([{"Признак": FEATURE_NAMES.get(key, key), "Корреляция с TARGET": correlations.loc[key, "TARGET"]} for key in top])
            st.caption("Корреляции Пирсона рассчитаны на исходной размеченной выборке. Они не доказывают причинность. Полная матрица сохранена в correlations.parquet; идентификаторы и чувствительные признаки в неё не входят.")
        else:
            st.info("Матрица корреляций ещё не рассчитана.")
    with st.expander("Большие файлы: размеры, типы, память и контроль объединений"):
        rows = []
        for name, source in profile.get("tables", {}).items():
            rows.append({"Источник": name, "Строк": source.get("rows"), "Столбцов": source.get("columns"),
                         "CSV, МиБ": source.get("source_bytes", 0) / 1024 ** 2, "Parquet, МиБ": source.get("parquet_bytes", 0) / 1024 ** 2,
                         "Первый блок до оптимизации, МиБ": source.get("first_chunk_memory_before_mb"),
                         "Первый блок после оптимизации, МиБ": source.get("first_chunk_memory_after_mb"), "Подготовка, с": source.get("seconds")})
        table(rows)
        st.caption("Память блоков измерена фактически средствами Pandas, включая строки. Это память одного блока до/после оптимизации типов, а не пик всего процесса. Отсутствующее измерение показывается пустым.")
        st.write("Строк после агрегации:", profile.get("feature_rows"), "· заявителей без подходящей истории:", profile.get("applicants_without_history"))
        st.json({"Несовпавшие дочерние записи": profile.get("unmatched_joins", {}), "Контроль времени событий": profile.get("history_cutoff_audit", {})})


def render_calibration(metrics):
    curves = []
    comparisons = []
    for variant, result in metrics.get("calibration", {}).items():
        label = "Только анкета" if variant == "form" else "Анкета + история"
        curve = result.get("holdout_curve", {})
        curves.extend({"Средняя предсказанная вероятность": predicted, "Фактическая доля TARGET=1": observed, "Модель": label}
                      for predicted, observed in zip(curve.get("predicted", []), curve.get("observed", [])))
        comparisons.append({"Набор признаков": label, "Brier до калибровки": result.get("holdout_brier_before"), "Brier после калибровки": result.get("holdout_brier_after")})
    if curves:
        figure = px.line(pd.DataFrame(curves), x="Средняя предсказанная вероятность", y="Фактическая доля TARGET=1", color="Модель", markers=True, color_discrete_sequence=PALETTE)
        figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="#9AA5B5", dash="dash"))
        figure.update_xaxes(tickformat=".0%", range=[0, max(.3, max(row["Средняя предсказанная вероятность"] for row in curves) * 1.1)])
        figure.update_yaxes(tickformat=".0%", range=[0, max(.3, max(row["Фактическая доля TARGET=1"] for row in curves) * 1.1)])
        st.plotly_chart(figure, use_container_width=True, key="calibration_curves")
        table(comparisons)
        st.caption("Калибровка sigmoid обучена на отдельной calibration-выборке; график и Brier — на нетронутом holdout. Меньший Brier лучше; ухудшение после калибровки также показано без исправления результата.")


def render_model_quality(config):
    path = Path(config["paths"]["artifact_dir"]) / "metrics.json"
    st.caption("TARGET=1 — платёжные трудности по определению датасета; это не текущая просрочка договора и не доказательство банкротства. Синтетические операции не участвуют в оценке модели.")
    if not path.exists():
        st.info("Метрики ещё не рассчитаны. После реального обучения здесь появятся результаты отложенной выборки.")
        st.code(train_command(), language="powershell")
        return
    metrics = cached_json(str(path), path.stat().st_mtime_ns)
    st.caption(f"Версия эксперимента: {metrics.get('version', '—')} · рассчитано UTC: {metrics.get('created_utc', '—')}")
    records = metrics.get("models", [])
    frame = pd.DataFrame(records)
    if not frame.empty:
        columns = [key for key in ("variant", "model", "roc_auc", "pr_auc", "precision", "recall", "f1", "threshold", "brier_score", "log_loss", "training_seconds", "inference_seconds", "inference_rows", "model_bytes") if key in frame]
        table(frame[columns])
        st.caption("PR-AUC = sklearn average_precision_score (AP). Порог выбирается по максимуму F1 на validation; финальный holdout не используется для подбора порога. Время прогноза относится ко всей указанной holdout-выборке, а не одному клиенту.")
        if {"roc_auc", "pr_auc"}.issubset(frame.columns):
            frame["Эксперимент"] = frame.get("variant", pd.Series("", index=frame.index)).map({"form": "Анкета", "history": "Анкета + история"}).fillna("") + " · " + frame.get("model", pd.Series("", index=frame.index)).astype(str)
            st.plotly_chart(px.bar(frame, x="Эксперимент", y=["roc_auc", "pr_auc"], barmode="group", color_discrete_sequence=PALETTE, title="Фактически измеренное качество"), use_container_width=True, key="quality_chart")
        for row in records:
            matrix = row.get("confusion_matrix")
            if matrix:
                variant = "Анкета" if row.get("variant") == "form" else "Анкета + история"
                with st.expander(f"Матрица ошибок: {variant} · {row.get('model')}"):
                    table(pd.DataFrame(matrix, index=["Факт 0", "Факт 1"], columns=["Прогноз 0", "Прогноз 1"]).reset_index(names=""))
    else:
        st.warning("Файл метрик существует, но таблица результатов пуста.")
    st.markdown("**Калибровка и ограничения вероятностей**")
    st.caption("Высокий ROC-AUC не гарантирует точность индивидуальных вероятностей. Сдвиг между историческими данными и новыми анкетами ограничивает применимость оценок.")
    render_calibration(metrics)
    with st.expander("Подбор параметров, время, размер моделей и измеренная память"):
        table(metrics.get("search", []))
        table([{"Этап": "До обучения", **metrics.get("memory_before", {})}, {"Этап": "После обучения", **metrics.get("memory_after", {})}])
        table([{"Набор": row.get("variant"), "Модель": row.get("model"), **row.get("memory", {})} for row in records])
        split = metrics.get("split", {})
        st.caption(f"Поиск: {split.get('search_rows', '—'):,} строк и {split.get('cv_folds', '—')} CV-блока. Все преобразования обучаются внутри Pipeline в каждом блоке. "
                   f"Полный эксперимент занял {metrics.get('seconds', 0):.2f} с; сохранённый комплект моделей — {metrics.get('model_bundle_bytes', 0):,} байт. Размеры и времена отдельных моделей приведены выше.")
        st.json({"Разбиение": split, "Выбранные модели": metrics.get("selected", {}), "Анализ ошибок": metrics.get("error_analysis", {})})
    with st.expander("Разбиение, выбранные модели и воспроизводимость"):
        st.json({key: value for key, value in metrics.items() if key not in ("models", "feature_importance")})
    with st.expander("Глобальная важность признаков — исследовательский анализ"):
        for variant, rows in metrics.get("feature_importance", {}).items():
            data = pd.DataFrame(rows)
            if not data.empty:
                data["Признак"] = data["feature"].map(lambda value: FEATURE_NAMES.get(value, value))
                title = "Анкета" if variant == "form" else "Анкета + история"
                st.plotly_chart(px.bar(data.head(12).sort_values("importance"), x="importance", y="Признак", orientation="h", title=f"Перестановочная важность: {title}", labels={"importance": "Уменьшение ROC-AUC при перестановке"}, color_discrete_sequence=PALETTE), use_container_width=True, key=f"importance_{variant}")
        st.caption("Глобальная важность не является локальным объяснением решения конкретного клиента.")
    render_eda(config)


def render_reports(snapshot, selection):
    metadata = report_metadata(selection["as_of"], selection["start"], selection["end"], selection["filters"])
    tables = {"Заявки": snapshot.get("applications", []), "Договоры": snapshot.get("agreements", []),
              "Платежи": snapshot.get("payments", []), "Сводка": [{"metric": key, "value": value} for key, value in snapshot.get("kpis", {}).items()]}
    st.caption("Файлы используют те же записи и показатели, что таблицы и диаграммы. Метаданные включают время формирования UTC, отчётную дату, период, фильтры, валюту и происхождение.")
    columns = st.columns(4)
    for column, (name, records) in zip(columns, tables.items()):
        column.download_button(f"{name} · CSV", csv_bytes(records, metadata), file_name=f"{name}_{selection['as_of']}.csv", mime="text/csv", key=f"export_{name}")
    left, right = st.columns(2)
    try:
        left.download_button("Полный отчёт Excel", xlsx_bytes(tables, metadata), file_name=f"portfolio_{selection['as_of']}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="export_excel")
    except ImportError:
        left.info("Excel-экспорт требует openpyxl из requirements.txt.")
    try:
        right.download_button("Сводный отчёт PDF", pdf_bytes(snapshot.get("kpis", {}), metadata), file_name=f"portfolio_{selection['as_of']}.pdf", mime="application/pdf", key="export_pdf")
    except ImportError:
        right.info("PDF-экспорт требует matplotlib из requirements.txt.")
    with st.expander("Сводные данные отчёта", expanded=True):
        table([{"Показатель": KPI_LABELS.get(key, key), "Значение": readable_value(key, value)} for key, value in snapshot.get("kpis", {}).items()])
        st.json(metadata)


def render_analytics(config):
    selection = analytics_filters(config)
    if not selection:
        return
    snapshot = cached_snapshot(cfg_json(config), selection["as_of"], selection["start"], selection["end"], json.dumps(selection["filters"], sort_keys=True), revision())
    tabs = st.tabs(["Портфель", "Продукты", "Активность заявок", "Качество модели", "Отчёты"])
    with tabs[0]:
        render_portfolio(snapshot, selection)
    with tabs[1]:
        render_products(config, snapshot)
    with tabs[2]:
        render_application_activity(snapshot)
    with tabs[3]:
        render_model_quality(config)
    with tabs[4]:
        render_reports(snapshot, selection)


def main():
    st.set_page_config(page_title="Кредитная платформа · Home Credit", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>
      .block-container {padding-top:2rem;max-width:1500px}
      [data-testid="stMetric"] {background:white;border:1px solid #E4EAF3;border-radius:12px;padding:17px 18px;box-shadow:0 2px 7px #17243B05}
      [data-testid="stMetricLabel"] {font-size:.84rem;color:#62718A}
      [data-testid="stMetricValue"] {font-size:1.65rem;font-weight:650}
      [data-testid="stSidebar"] {background:#EDF2FB;border-right:1px solid #DDE5F1}
      .stTabs [data-baseweb="tab-list"] {gap:18px;border-bottom:1px solid #DDE5F1}
      .stTabs [data-baseweb="tab"] {font-weight:600}
      [data-testid="stForm"] {background:#FFFFFF;border:1px solid #E0E7F1;border-radius:14px;padding:22px}
      h1,h2,h3 {color:#17243B;letter-spacing:-.02em}
      div.stButton>button[kind="primary"] {border-radius:8px}
    </style>""", unsafe_allow_html=True)
    try:
        config = load_config()
        db.init_db(config)
    except Exception as error:
        show_error(error, "Платформа не запущена")
        st.stop()
    st.sidebar.markdown("## Кредитная платформа")
    st.sidebar.caption("HOME CREDIT · УЧЕБНАЯ СРЕДА")
    section = st.sidebar.radio("Раздел", ["Кредитная деятельность", "Аналитика"], label_visibility="collapsed")
    st.sidebar.divider()
    if st.sidebar.button("Обновить данные", key="refresh_data"):
        invalidate()
        st.rerun()
    st.sidebar.caption("Все суммы — условные единицы (у.е.). Операции локальные, переводы денег не выполняются.")
    model_path = Path(config["paths"]["artifact_dir"]) / "model_bundle.joblib"
    st.sidebar.markdown("**Модель:** " + ("артефакт доступен" if model_path.exists() else "не обучена"))
    if not model_path.exists():
        st.sidebar.caption("Для решений сначала подготовьте данные и обучите модель из командной строки.")
    st.title(section)
    st.caption("Единый процесс: анкета → решение → отдельная выдача → платежи → портфель. Источники и версии сохраняются.")
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    try:
        if section == "Кредитная деятельность":
            render_operations(config)
        else:
            render_analytics(config)
    except Exception as error:
        show_error(error, "Не удалось открыть данные раздела")
        st.caption("Исходные CSV не изменены. Проверьте подготовку данных и совместимость артефактов; подробность ошибки записана в журнал.")
