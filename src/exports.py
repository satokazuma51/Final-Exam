"""Filter-preserving, self-describing report downloads (no external services)."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import json

import pandas as pd

KPI_LABELS = {
    "clients": "Клиенты, человек", "applications": "Заявки за период, шт.", "decisions": "Завершённые решения, шт.",
    "approvals": "Одобрено, шт.", "rejections": "Отказано, шт.", "pending": "Ожидают решения, шт.",
    "approval_rate": "Доля одобрений среди завершённых решений", "disbursed_count": "Договоры с фактической выдачей за период, шт.",
    "disbursed_amount": "Фактические выдачи и выборки за период, у.е.", "active_agreements": "Активные договоры на дату, шт.",
    "closed_agreements": "Закрытые договоры на дату, шт.", "active_portfolio_balance": "Основной долг активного портфеля на дату, у.е.",
    "outstanding_principal": "Непогашенный основной долг, у.е.", "agreements_in_arrears": "Договоры с просрочкой на дату, шт.",
    "overdue_amount": "Просроченные обязательства на дату, у.е.", "payments_received": "Полученные платежи за период, у.е.",
    "principal_received": "Погашено основного долга за период, у.е.", "interest_received": "Получено процентов за период, у.е.",
    "average_disbursed_amount": "Средняя выдача на профинансированный договор за период, у.е.",
    "average_risk": "Средняя оценка риска среди решённых заявок периода", "total_credit_limits": "Лимиты активных кредитных линий, у.е.",
    "drawn_balance": "Выбранный основной долг по кредитным линиям, у.е."}


def readable_value(key, value):
    if value is None:
        return "не определено (пустая совокупность)"
    if key in ("approval_rate", "average_risk"):
        return f"{float(value):.2%}"
    if isinstance(value, float):
        return f"{value:,.2f}".replace(",", " ")
    return str(value)


def report_metadata(as_of, period_start, period_end, filters=None) -> dict:
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reporting_date": str(as_of), "period_start": str(period_start),
            "period_end": str(period_end), "currency": "у.е.",
            "provenance": "Местный реестр; синтетические записи отмечены source_kind / is_synthetic; внешнее бюро исключено",
            "filters": json.dumps(filters or {}, ensure_ascii=False, sort_keys=True)}


def report_frame(records, metadata: dict) -> pd.DataFrame:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    for key, value in metadata.items():
        frame["report_" + key] = str(value)
    return frame


def csv_bytes(records, metadata: dict) -> bytes:
    return report_frame(records, metadata).to_csv(index=False).encode("utf-8-sig")


def xlsx_bytes(tables: dict, metadata: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(list(metadata.items()), columns=["Параметр", "Значение"]).to_excel(writer, sheet_name="Метаданные", index=False)
        for name, records in tables.items():
            frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
            # Nested snapshots are exported as explicit JSON strings.
            for column in frame.columns:
                frame[column] = frame[column].map(lambda v: json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v)
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()


def pdf_bytes(kpis: dict, metadata: dict) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import PdfPages
    from textwrap import wrap

    output = BytesIO()
    lines = ["Учебная кредитная платформа", "Портфельный отчёт", ""]
    metadata_labels = {"generated_at_utc": "Сформировано (UTC)", "reporting_date": "Отчётная дата (снимок)",
                       "period_start": "Период операций: с", "period_end": "Период операций: по", "currency": "Валюта / единица суммы",
                       "provenance": "Происхождение данных", "filters": "Выбранные фильтры"}
    for key, value in metadata.items():
        if key == "filters":
            filters = json.loads(value) if isinstance(value, str) else value
            filter_names = {"product": "продукт", "status": "статус договора", "client_type": "тип клиента", "source_kind": "происхождение", "client_id": "ID клиента"}
            filter_values = {"cash": "денежный кредит", "revolving": "кредитная линия", "Active": "активен", "Closed": "закрыт", "New": "новый", "Repeat": "повторный", "local": "местные операции пользователя", "synthetic": "демонстрационные операции", "home_credit": "исторический Home Credit"}
            value = "; ".join(f"{filter_names.get(name, name)}: {filter_values.get(str(selected), selected)}" for name, selected in filters.items()) if filters else "все записи выбранного периода и отчётной даты"
        lines.extend(wrap(f"{metadata_labels.get(key, key)}: {value}", width=85))
    lines.append("")
    lines.extend(["Остатки и статусы — на отчётную дату; выдачи и платежи — за период.",
                  "Все суммы — у.е.; учитываются только собственные активы платформы.", ""])
    for key, value in kpis.items():
        if not isinstance(value, (dict, list)):
            lines.extend(wrap(f"{KPI_LABELS.get(key, key)}: {readable_value(key, value)}", width=85))
    lines.extend(["", "Доля одобрений = одобрено / (одобрено + отказано).",
                  "Полученные проценты — доход до расходов, а не чистая прибыль."])
    with PdfPages(output) as pdf:
        for start in range(0, len(lines), 48):
            figure = Figure(figsize=(8.27, 11.69), facecolor="white")
            axis = figure.add_axes((.08, .06, .84, .9))
            axis.axis("off")
            axis.text(0, 1, "\n".join(lines[start:start + 48]), va="top", fontsize=10,
                      fontfamily="DejaVu Sans", linespacing=1.5, color="#17243B")
            pdf.savefig(figure)
    return output.getvalue()
