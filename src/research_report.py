"""Generate offline defense materials from computed artifacts, never invented metrics."""
from __future__ import annotations

from datetime import datetime, timezone
import base64
from html import escape
import json
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from .config import ensure_directories


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _table(rows):
    frame = pd.DataFrame(rows)
    return frame.to_html(index=False, border=0, float_format=lambda n: f"{n:,.4f}") if len(frame) else "<p>Нет записей для выбранной выборки.</p>"


def _save_figure(fig, folder, name):
    fig.tight_layout()
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _source_queries(config, report_dir):
    path = Path(config["paths"]["analytics_database"])
    if not path.exists():
        return {}
    text = (Path(__file__).resolve().parents[1] / "sql" / "source_analytics.sql").read_text(encoding="utf-8")
    result = {}
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        sample_id = connection.execute("SELECT MIN(SK_ID_CURR) FROM bureau").fetchone()[0]
        for chunk in text.split("-- name: ")[1:]:
            name, query = chunk.split("\n", 1)
            frame = pd.read_sql_query(query.split(";", 1)[0], connection, params={"application_id": sample_id})
            frame.to_csv(report_dir / f"sql_source_{name.strip()}.csv", index=False, encoding="utf-8-sig")
            result[name.strip()] = frame
    finally:
        connection.close()
    return result


def _slide(presentation, title, text=None, image=None):
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(248, 250, 252)
    box = slide.shapes.add_textbox(Inches(.55), Inches(.3), Inches(12.2), Inches(.8))
    paragraph = box.text_frame.paragraphs[0]
    paragraph.text = title
    paragraph.font.name, paragraph.font.size = "Arial", Pt(26)
    paragraph.font.bold = True
    paragraph.font.color.rgb = RGBColor(23, 65, 117)
    if image:
        # The fixed plotting canvas fits comfortably below the title.
        slide.shapes.add_picture(str(image), Inches(.7), Inches(1.2), width=Inches(11.7), height=Inches(5.5))
    if text:
        box = slide.shapes.add_textbox(Inches(.7), Inches(1.35), Inches(11.8), Inches(5.7))
        frame = box.text_frame
        frame.word_wrap = True
        for index, line in enumerate(text):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.text = str(line)
            paragraph.font.name, paragraph.font.size = "Arial", Pt(21)
            paragraph.space_after = Pt(18)
    return slide


def build_reports(config: dict) -> dict:
    ensure_directories(config)
    folder = Path(config["paths"]["reports_dir"])
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    profile = _load(Path(config["paths"]["processed_dir"]) / "profile.json")
    metrics = _load(Path(config["paths"]["artifact_dir"]) / "metrics.json")
    if not profile or not metrics:
        raise FileNotFoundError("Computed preparation/training artifacts are required: run prepare, then train")
    source_tables = [dict(file=name + ".csv", rows=data["rows"], columns=data["columns"] if isinstance(data["columns"], int) else len(data["columns"]),
                         source_mb=data["source_bytes"] / 2**20,
                         chunk_memory_before_mb=data.get("first_chunk_memory_before_mb"),
                         chunk_memory_after_mb=data.get("first_chunk_memory_after_mb"))
                     for name, data in profile["tables"].items()]
    table_rows = []
    for row in metrics["models"]:
        table_rows.append({key: row.get(key) for key in ["variant", "model", "roc_auc", "pr_auc", "precision", "recall", "f1", "brier_score", "threshold", "training_seconds", "inference_seconds", "model_bytes"]})
    comparison = pd.DataFrame(table_rows)
    comparison.to_csv(folder / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(source_tables).to_csv(folder / "data_inventory.csv", index=False, encoding="utf-8-sig")
    charts = {}
    target = profile["target_balance"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(["0: other cases", "1: payment difficulties"], [target.get("0", 0), target.get("1", 0)], color=["#2563eb", "#e26850"])
    ax.set_ylabel("Labeled applications")
    ax.set_title("Observed TARGET distribution")
    charts["target_balance"] = _save_figure(fig, figures, "target_balance")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    labels = comparison.variant + " / " + comparison.model
    for ax, metric in zip(axes, ["roc_auc", "pr_auc"]):
        ax.barh(labels, comparison[metric], color=["#60a5fa" if v == "form" else "#1d4ed8" for v in comparison.variant])
        ax.set_xlim(0, 1)
        ax.set_title("Holdout " + metric)
    charts["model_comparison"] = _save_figure(fig, figures, "model_comparison")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal calibration")
    for variant, values in metrics["calibration"].items():
        curve = values["holdout_curve"]
        ax.plot(curve["predicted"], curve["observed"], "o-", label=variant)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Untouched holdout calibration assessment")
    ax.legend()
    charts["calibration"] = _save_figure(fig, figures, "calibration")
    fig, ax = plt.subplots(figsize=(10, 5))
    imp = pd.DataFrame(metrics["feature_importance"]["history"]).head(12).iloc[::-1]
    ax.barh(imp.feature, imp.importance, xerr=imp["std"], color="#2563eb")
    ax.set(xlabel="Decrease in ROC-AUC after permutation", title="Global permutation importance (not an individual explanation)")
    charts["importance"] = _save_figure(fig, figures, "importance")
    train_info = profile["tables"]["application_train"]
    missing = pd.Series(train_info["missing"]).sort_values(ascending=False).head(12) / train_info["rows"] * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(missing.index[::-1], missing.values[::-1], color="#d99b23")
    ax.set(xlabel="Missing values (%)", title="Most incomplete application fields")
    charts["missing"] = _save_figure(fig, figures, "missing_values")
    data = pd.read_parquet(Path(config["paths"]["processed_dir"]) / "application_features.parquet", columns=["TARGET", "monthly_income", "loan_amount", "payment_to_income"])
    data = data[data.TARGET.notna()]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax, column in zip(axes, ["monthly_income", "loan_amount", "payment_to_income"]):
        values = data[column].dropna()
        cap = values.quantile(.99)
        ax.hist(values[values <= cap], bins=45, color="#2563eb")
        ax.set_title(column + " (up to p99)")
        ax.set_ylabel("Applications")
    charts["distributions"] = _save_figure(fig, figures, "distributions")
    corr = pd.Series(profile["target_correlations"]).head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(corr.index, corr.values, color=["#dc654b" if value > 0 else "#2563eb" for value in corr.values])
    ax.set(title="Largest absolute numeric correlations with TARGET", xlabel="Pearson correlation (association, not causation)")
    charts["correlations"] = _save_figure(fig, figures, "correlations")
    source_queries = _source_queries(config, folder)
    from .analytics import snapshot
    operational = snapshot(config, config["demo"]["reference_date"], period_start="2025-01-01", period_end=config["demo"]["reference_date"])
    for name, frame in operational["sql_results"].items():
        frame.to_csv(folder / f"sql_{name}.csv", index=False, encoding="utf-8-sig")
    (folder / "portfolio_summary.json").write_text(json.dumps(operational["kpis"], ensure_ascii=False, indent=2), encoding="utf-8")
    verification_path = folder / "verification.xml"
    verification = None
    if verification_path.exists():
        root = ET.parse(verification_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        verification = {key: sum(int(s.get(key, 0)) for s in suites) for key in ["tests", "failures", "errors", "skipped"]}
    history_auc, form_auc = metrics["selected"]["history"]["roc_auc"], metrics["selected"]["form"]["roc_auc"]
    sections = [
        ("Задача и границы", "<p>Локальная учебная платформа: заявка → оценка риска и платежеспособности → решение → отдельная выдача → платежи → портфель. Ровно два раздела: «Кредитная деятельность» и «Аналитика». Денежная единица — у.е.; исходная валюта не раскрыта.</p><p>TARGET=1 означает платежные трудности по определению набора. Это не банкротство и не текущий статус договора. Историческая заявка — не универсальный идентификатор человека.</p>"),
        ("Источники и качество данных", _table(source_tables) + f"<p>После агрегации: {profile['feature_rows']:,} уникальных исходных заявок. Положительных исходов: {target.get('1', 0):,}. Потоковая обработка, Parquet, SQLite с индексами; исходные CSV не изменяются.</p>" + "<pre>" + escape(json.dumps(profile.get("unmatched_joins", {}), ensure_ascii=False, indent=2)) + "</pre><p>Фактически рассчитанный аудит временного ограничения:</p><pre>" + escape(json.dumps(profile.get("history_cutoff_audit", {}), ensure_ascii=False, indent=2)) + "</pre><p>Измерения памяти в таблице относятся к первому реальному чанку до и после оптимизации типов. Память процесса первоначального полного импорта не была получена ранней версией помощника; отсутствующие значения не восстанавливаются предположением. Память финального обучения измерена отдельно.</p>"),
        ("EDA и пропуски", "<p>Показаны действительные частоты, пропуски, распределения и корреляции. Гистограммы ограничены p99 только для читаемости; строки не удалялись из обучения по этому правилу. Выбросы по IQR, числовые описания и все пропуски сохранены в profile.json. DAYS_EMPLOYED=365243 переводится в пропуск.</p>"),
        ("Признаки и защита от утечки", "<p>Форма: месячный доход (исходный AMT_INCOME_TOTAL / 12), кредит/лимит, месячный платеж, стаж, отношения кредита и платежа к доходу, продукт и вид занятости. Исторический вариант добавляет только агрегаты на исходную дату заявки. Имена, ID, пол, возраст, национальность, религия, TARGET и EXT_SOURCE исключены из предикторов. Возраст проверяется отдельно учебной политикой.</p><p>Фактические события после относительной даты 0 исключены. Локальные операции не подменяют исходные агрегаты: если надежного сопоставления нет, применяется модель формы. Исходная история не является проверкой бюро сегодня.</p>"),
        ("Методика исследования", "<pre>" + escape(json.dumps(metrics["split"], ensure_ascii=False, indent=2)) + "</pre><p>Pipeline/ColumnTransformer: обучаемая внутри CV импутация, индикаторы пропусков, безопасное кодирование категорий, масштабирование для логистической регрессии. Ограниченный поиск гиперпараметров на обучающей части. Отдельная калибровочная часть, отдельная валидация для выбора модели и порога максимального F1; финальная контрольная часть не используется для выбора. PR-AUC рассчитывается как average_precision_score.</p>"),
        ("Реальные результаты", _table(table_rows) + f"<p>Сравнение выбранных моделей на одинаковом holdout: форма ROC-AUC={form_auc:.4f}; форма + история={history_auc:.4f}; разница={history_auc - form_auc:+.4f}. Это наблюдаемый результат, без обещания заданного качества. Общее время обучения: {metrics['seconds']:.2f} с; размер итогового пакета: {metrics['model_bundle_bytes']:,} байт.</p><p>Измеренная память процесса обучения (МиБ):</p><pre>" + escape(json.dumps({"before": metrics["memory_before"], "after": metrics["memory_after"]}, ensure_ascii=False, indent=2)) + "</pre><p>Калибровка оценивается после фиксации всех решений; ее улучшение не гарантируется и не выбирается по holdout:</p>" + _table([dict(variant=key, **{k: v for k, v in value.items() if k != "holdout_curve"}) for key, value in metrics["calibration"].items()]) + "<p>Ошибки на контрольной выборке:</p><pre>" + escape(json.dumps(metrics.get("error_analysis", {}), ensure_ascii=False, indent=2)) + "</pre>"),
        ("Учет и учебная политика", "<p>Денежные суммы в SQLite — целые сотые у.е. Платеж сначала погашает начисленные проценты, затем основной долг. Денежный кредит: аннуитетный исходный график; проценты по месячной ставке на фактический остаток в дату платежа; раннее погашение уменьшает будущие проценты. Кредитная линия: ACT/365 по фактическому остатку, месячный минимум; возврат основного долга восстанавливает доступный лимит. Нулевой долг не закрывает линию автоматически.</p><p>Долговая нагрузка = (существующие месячные платежи + новый платеж) / месячный доход. Для известных обязательств используется max(заявленные, учтенные платформой), чтобы избежать повторного сложения. Бюро показывается отдельно, его исторические суммы не прибавляются к сегодняшним обязательствам без подтверждения.</p><pre>" + escape(json.dumps(config["policy"], ensure_ascii=False, indent=2)) + "</pre>"),
        ("Портфель и SQL", _table([operational["kpis"]]) + "<p>Все показатели получены из договоров и датированных проводок; отдельные случайные значения для карточек не создаются. Выдано — реальное событие учебного реестра, не одобрение. Активы платформы исключают внешние долги. Результаты 10 операционных и 3 исходных SQL-запросов экспортированы в CSV.</p>"),
        ("Проверки", "<pre>" + escape(json.dumps(verification, ensure_ascii=False, indent=2)) + "</pre><p>Подробности фактически выполненных проверок: verification.xml; без него результаты тестирования не заявляются.</p>"),
        ("Ограничения и выводы", "<ul>" + "".join("<li>" + escape(value) + "</li>" for value in metrics["limitations"]) + "<li>Демонстрационный календарь, имена, ставки и операции синтетические; исходные профили и TARGET остаются реальными и отдельными.</li><li>Калибровка проверена на holdout, но не доказывает пригодность вероятностей для реального кредитования.</li><li>Дату экзамена задает преподаватель; материалы следует подать в Moodle за 2–3 дня до нее.</li></ul>"),
        ("Официальные источники", '<p><a href="https://www.kaggle.com/competitions/home-credit-default-risk/data">Kaggle Home Credit</a>; <a href="https://docs.streamlit.io/develop/concepts/architecture/forms">Streamlit forms</a>; <a href="https://docs.streamlit.io/develop/concepts/architecture/caching">Streamlit caching</a>; <a href="https://scikit-learn.org/stable/common_pitfalls.html">scikit-learn leakage</a>; <a href="https://scikit-learn.org/stable/modules/compose.html">Pipeline</a>; <a href="https://lightgbm.readthedocs.io/">LightGBM</a>.</p>'),
    ]
    pictures = "".join(f'<figure><img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="{escape(name)}"><figcaption>{escape(name)}</figcaption></figure>' for name, path in charts.items())
    body = "".join(f"<section><h2>{escape(title)}</h2>{content}</section>" for title, content in sections)
    html = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Исследование Home Credit</title><style>body{font:16px Arial,sans-serif;color:#172c48;max-width:1250px;margin:40px auto;padding:0 24px;background:#f8fafc}h1,h2{color:#174175}section{background:white;padding:24px;margin:18px 0;border-radius:12px}table{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;font-size:12px}img{max-width:100%}figure{background:white;margin:24px 0;padding:12px}@media print{body{margin:0;background:white}section,figure{break-inside:avoid}}</style><h1>Учебная кредитная платформа: исследование и результаты</h1>'
    html += f"<p>Модель {escape(metrics['version'])}; сформировано {datetime.now(timezone.utc).isoformat()} UTC.</p>" + body + pictures + "</html>"
    report_path = folder / "research_report.html"
    report_path.write_text(html, encoding="utf-8")
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Inches(13.333), Inches(7.5)
    _slide(presentation, "Учебная кредитная платформа · Home Credit", ["Python · Streamlit · SQLite · scikit-learn · LightGBM", "Заявка → решение → отдельная выдача → платеж → изменение портфеля", "Два раздела: «Кредитная деятельность» и «Аналитика»", "Локальная демонстрация, у.е.; реальных переводов нет"])
    _slide(presentation, "Данные и единица наблюдения", [f"10 CSV; {profile['feature_rows']:,} исходных заявок после агрегации", f"Обучающих наблюдений: {sum(target.values()):,}; TARGET=1: {target.get('1',0):,}", "История агрегируется до одной строки на исходную заявку", "Клиент платформы имеет собственный client_id; исходная связь — source_application_id"])
    for key, title in [("target_balance", "Баланс целевой переменной"), ("missing", "Качество данных: пропуски"), ("distributions", "Распределения и выбросы"), ("correlations", "EDA: корреляции")]:
        _slide(presentation, title, image=charts[key])
    split = metrics["split"]
    _slide(presentation, "Эксперимент без утечки", [f"Train {split['train']:,}; calibration {split['calibration']:,}; validation {split['validation']:,}; holdout {split['holdout']:,}", f"CV: {split['cv_folds']} folds; поиск на {split['search_rows']:,} обучающих строках", "DummyClassifier, LogisticRegression, LightGBM × форма / форма+история", "Преобразования внутри CV; порог максимального F1 выбирается только на validation", "Возраст — правило допуска, не признак модели; имена, ID и пол исключены"])
    _slide(presentation, "Фактически измеренное качество", image=charts["model_comparison"])
    _slide(presentation, "Проверка калибровки", image=charts["calibration"])
    _slide(presentation, "Факторы прогноза", image=charts["importance"])
    _slide(presentation, "Учет и воспроизводимые операции", ["Одобрение не меняет портфель до отдельной выдачи", "Аннуитет: округление до сотых, месяц-конец, корректировка последнего платежа", "Линия: лимит ≠ выбранный основной долг; ACT/365; отдельное закрытие", "Платежи: проценты → основной долг; текущий DPD и исторический максимум отдельно", "Исторические срезы восстанавливаются по датам; уникальные ключи защищают от дубликатов"])
    kpis = operational["kpis"]
    _slide(presentation, "Демонстрационный портфель: данные реестра", [f"Клиентов: {kpis['clients']}; выданных договоров за период: {kpis['disbursed_count']}", f"Активных: {kpis['active_agreements']}; закрытых: {kpis['closed_agreements']}", f"Остаток основного долга: {kpis['active_portfolio_balance']:,.2f} у.е.", f"Получено платежей: {kpis['payments_received']:,.2f} у.е.; проценты: {kpis['interest_received']:,.2f} у.е.", "Синтетические операции явно помечены; метрики моделей обучены только на исходных TARGET"])
    _slide(presentation, "Проверки и ограничения", ["Выполненные тесты: " + json.dumps(verification, ensure_ascii=False), "Вероятность платежных трудностей — не доказательство невозврата", "Нет точной календарной истории и универсального ID человека", "Исходное бюро — исторический снимок; локальная история не подменяет его", "Материалы Moodle: за 2–3 дня до даты экзамена, которую задает преподаватель"])
    _slide(presentation, "Маршрут демонстрации", ["Существующий демо-клиент → новая заявка → фактическая оценка → решение", "При одобрении: отдельная выдача → частичный платеж → обновленный портфель", "Новый клиент без истории: модель формы и явный признак отсутствия исходной истории", "Пример отказа: высокий платежный бюджет; причины из расчетов", "Повторный клик не создает новую операцию; история и отчеты сохраняются в SQLite"])
    pptx_path = folder / "defense_presentation.pptx"
    presentation.save(pptx_path)
    return {"report": str(report_path), "presentation": str(pptx_path), "slides": len(presentation.slides),
            "figures": len(charts), "operational_sql_queries": len(operational["sql_results"]),
            "source_sql_queries": len(source_queries), "verification": verification,
            "selected_models": metrics["selected"]}


def export_submission(config: dict) -> dict:
    from .models import load_model_bundle
    bundle = load_model_bundle(config)
    frame = pd.read_parquet(Path(config["paths"]["processed_dir"]) / "application_features.parquet")
    frame = frame[frame.source_table == "application_test"].copy()
    probability = np.zeros(len(frame))
    for variant in ("form", "history"):
        mask = (frame.history_present == 1).to_numpy() if variant == "history" else (frame.history_present != 1).to_numpy()
        selected = bundle["variants"][variant]
        if mask.any():
            probability[mask] = selected["calibrated"].predict_proba(frame.loc[mask, selected["features"]])[:, 1]
    predicted = pd.DataFrame({"SK_ID_CURR": frame.SK_ID_CURR.to_numpy(), "TARGET": probability})
    sample = pd.read_csv(Path(config["paths"]["data_dir"]) / "sample_submission.csv", usecols=["SK_ID_CURR"])
    result = sample.merge(predicted, on="SK_ID_CURR", validate="one_to_one", how="left")
    if result.TARGET.isna().any() or not result.TARGET.between(0, 1).all():
        raise ValueError("Submission IDs or probabilities are incomplete")
    path = Path(config["paths"]["artifact_dir"]) / "submission.csv"
    result.to_csv(path, index=False)
    return {"path": str(path), "rows": len(result), "model_version": bundle["version"], "test_labels_available": False}
