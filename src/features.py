"""Features whose units and cutoff are shared by training and manual applications."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FORM_NUMERIC = ["monthly_income", "loan_amount", "monthly_payment", "employment_years",
                "loan_to_annual_income", "payment_to_income"]
FORM_CATEGORICAL = ["product", "employment_type"]
FORM_FEATURES = FORM_NUMERIC + FORM_CATEGORICAL
HISTORY_FEATURES = ["bureau_count", "bureau_active_count", "bureau_debt_sum", "bureau_max_dpd",
    "bureau_overdue_sum", "bureau_months", "bureau_late_fraction", "bureau_max_severity",
    "previous_count", "previous_approval_rate", "previous_recent_count", "previous_mean_credit",
    "pos_months", "pos_max_dpd", "pos_late_fraction", "card_months", "card_max_dpd",
    "card_mean_utilization", "payment_events", "payment_late_fraction", "payment_max_lateness",
    "payment_total_amount", "history_present"]


def estimated_payment(inputs: dict, config: dict | None = None) -> float:
    """Monthly cash annuity or revolving principal minimum plus one-month interest."""
    amount = float(inputs["amount"])
    rate = float(inputs.get("annual_rate", 0)) / 1200
    product = inputs.get("product", "cash")
    if product == "revolving":
        drawn = float(inputs.get("initial_draw", amount))
        policy = (config or {}).get("products", {}).get("revolving", {})
        principal = min(drawn, max(float(policy.get("revolving_min_payment", 100)),
                                   drawn * float(policy.get("revolving_min_principal_pct", .05))))
        return principal + drawn * rate
    n = int(inputs.get("term_months", 12))
    if n <= 0:
        raise ValueError("Loan term must be positive")
    return amount / n if rate == 0 else amount * rate / (1 - (1 + rate) ** -n)


def form_to_features(inputs: dict, config: dict | None = None) -> dict:
    income, amount = float(inputs["monthly_income"]), float(inputs["amount"])
    if not np.isfinite(income) or income <= 0 or not np.isfinite(amount) or amount <= 0:
        raise ValueError("Monthly income and requested amount must be positive finite numbers")
    product = inputs.get("product", "cash")
    if product not in ("cash", "revolving"):
        raise ValueError("The serving model supports cash and revolving products only")
    payment = float(inputs.get("_monthly_payment", estimated_payment(inputs, config)))
    tenure = inputs.get("employment_years")
    return {"monthly_income": income, "loan_amount": amount, "monthly_payment": payment,
            "employment_years": float(tenure) if tenure is not None else np.nan,
            "loan_to_annual_income": amount / (income * 12), "payment_to_income": payment / income,
            "product": product, "employment_type": str(inputs.get("employment_type", "Unknown"))}


def application_features(raw: pd.DataFrame) -> pd.DataFrame:
    income = pd.to_numeric(raw["AMT_INCOME_TOTAL"], errors="coerce") / 12
    amount = pd.to_numeric(raw["AMT_CREDIT"], errors="coerce")
    payment = pd.to_numeric(raw["AMT_ANNUITY"], errors="coerce")
    days = pd.to_numeric(raw["DAYS_EMPLOYED"], errors="coerce").replace(365243, np.nan)
    return pd.DataFrame({"monthly_income": income, "loan_amount": amount, "monthly_payment": payment,
        "employment_years": (-days / 365.25).where(days <= 0),
        "loan_to_annual_income": amount / (income * 12).where(income > 0),
        "payment_to_income": payment / income.where(income > 0),
        "product": raw["NAME_CONTRACT_TYPE"].astype(str).map({"Cash loans": "cash", "Revolving loans": "revolving"}).fillna("Unknown"),
        "employment_type": raw["NAME_INCOME_TYPE"].astype(str)})


def _batches(path: Path, columns: list[str], batch_size: int):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def aggregate_history(config: dict, application_ids: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Stream historical rows into fixed-size application accumulators; never raw joins."""
    folder = Path(config["paths"]["processed_dir"])
    size = int(config["training"]["chunksize"])
    index = pd.Index(application_ids)
    n = len(index)
    data: dict[str, np.ndarray] = {}
    audit: dict[str, dict] = {}

    def add(name, positions, values=1):
        target = data.setdefault(name, np.zeros(n, dtype=np.float64))
        np.add.at(target, positions, values)

    def maximum(name, positions, values):
        target = data.setdefault(name, np.full(n, -np.inf))
        np.maximum.at(target, positions, values)

    def valid_rows(frame, table, cutoff, key="SK_ID_CURR", parent_known=None, missing_cutoff=None):
        stats = audit.setdefault(table, {"input_rows": 0, "excluded_after_cutoff": 0, "missing_parent_rows": 0, "missing_cutoff_rows": 0, "unmatched_application_rows": 0})
        stats["input_rows"] += len(frame)
        keep = cutoff.fillna(False).to_numpy()
        known = np.ones(len(frame), dtype=bool) if parent_known is None else parent_known.to_numpy(dtype=bool)
        missing = np.zeros(len(frame), dtype=bool) if missing_cutoff is None else missing_cutoff.to_numpy(dtype=bool)
        stats["missing_parent_rows"] += int((~known).sum())
        stats["missing_cutoff_rows"] += int((known & missing).sum())
        stats["excluded_after_cutoff"] += int((known & ~keep & ~missing).sum())
        frame = frame.loc[keep]
        positions = index.get_indexer(frame[key])
        stats["unmatched_application_rows"] += int((positions < 0).sum())
        return frame.loc[positions >= 0], positions[positions >= 0]

    bureau_map_parts = []
    for frame in _batches(folder / "bureau.parquet", ["SK_ID_CURR", "SK_ID_BUREAU", "DAYS_CREDIT", "DAYS_CREDIT_UPDATE", "CREDIT_ACTIVE", "AMT_CREDIT_SUM_DEBT", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM_OVERDUE"], size):
        # Map even unusual unmatched/future parent records for a complete child audit.
        bureau_map_parts.append(frame[["SK_ID_BUREAU", "SK_ID_CURR", "DAYS_CREDIT"]])
        # Debt, overdue and active status reflect the last update. A credit may
        # originate before the application but have an inadmissible later update.
        future_updates = int(((frame.DAYS_CREDIT <= 0) & (frame.DAYS_CREDIT_UPDATE > 0)).sum())
        frame, p = valid_rows(frame, "bureau", (frame.DAYS_CREDIT <= 0) & (frame.DAYS_CREDIT_UPDATE <= 0),
                              missing_cutoff=frame.DAYS_CREDIT.isna() | frame.DAYS_CREDIT_UPDATE.isna())
        audit["bureau"]["excluded_future_update_rows"] = audit["bureau"].get("excluded_future_update_rows", 0) + future_updates
        add("bureau_count", p)
        add("bureau_active_count", p, (frame.CREDIT_ACTIVE.astype(str) == "Active").to_numpy(dtype=float))
        for raw, name in [("AMT_CREDIT_SUM_DEBT", "bureau_debt_sum"), ("AMT_CREDIT_SUM_OVERDUE", "bureau_overdue_sum")]:
            present = frame[raw].notna().to_numpy()
            add(name, p[present], frame.loc[present, raw].to_numpy(dtype=float))
            add(name + "_observed", p[present])
        maximum("bureau_max_dpd", p, frame.CREDIT_DAY_OVERDUE.fillna(0).to_numpy(dtype=float))
    mapping = pd.concat(bureau_map_parts, ignore_index=True).drop_duplicates("SK_ID_BUREAU").set_index("SK_ID_BUREAU")
    del bureau_map_parts
    for frame in _batches(folder / "bureau_balance.parquet", ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"], size):
        frame["SK_ID_CURR"] = frame.SK_ID_BUREAU.map(mapping.SK_ID_CURR)
        parent_cutoff = frame.SK_ID_BUREAU.map(mapping.DAYS_CREDIT) <= 0
        frame, p = valid_rows(frame, "bureau_balance", (frame.MONTHS_BALANCE <= 0) & parent_cutoff,
                              parent_known=frame.SK_ID_CURR.notna(), missing_cutoff=frame.MONTHS_BALANCE.isna())
        severity = pd.to_numeric(frame.STATUS.astype(str), errors="coerce")
        add("bureau_months", p)
        known = severity.notna().to_numpy()
        add("bureau_known_months", p[known])
        add("bureau_late_months", p[known], (severity.loc[known] > 0).to_numpy(dtype=float))
        maximum("bureau_max_severity", p[known], severity.loc[known].to_numpy(dtype=float))
    del mapping
    for frame in _batches(folder / "previous_application.parquet", ["SK_ID_CURR", "DAYS_DECISION", "NAME_CONTRACT_STATUS", "AMT_CREDIT"], size):
        frame, p = valid_rows(frame, "previous_application", frame.DAYS_DECISION <= 0, missing_cutoff=frame.DAYS_DECISION.isna())
        add("previous_count", p)
        add("previous_approved", p, (frame.NAME_CONTRACT_STATUS.astype(str) == "Approved").to_numpy(dtype=float))
        add("previous_recent_count", p, (frame.DAYS_DECISION >= -365).to_numpy(dtype=float))
        observed = frame.AMT_CREDIT.notna().to_numpy()
        add("previous_credit_total", p[observed], frame.loc[observed, "AMT_CREDIT"].to_numpy(dtype=float))
        add("previous_credit_observed", p[observed])
    for table, prefix in [("POS_CASH_balance", "pos"), ("credit_card_balance", "card")]:
        cols = ["SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD"]
        if prefix == "card":
            cols += ["AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"]
        for frame in _batches(folder / (table + ".parquet"), cols, size):
            frame, p = valid_rows(frame, table, frame.MONTHS_BALANCE <= 0, missing_cutoff=frame.MONTHS_BALANCE.isna())
            add(prefix + "_months", p)
            maximum(prefix + "_max_dpd", p, frame.SK_DPD.fillna(0).to_numpy(dtype=float))
            if prefix == "pos":
                add("pos_late_months", p, (frame.SK_DPD > 0).to_numpy(dtype=float))
            else:
                utilization = frame.AMT_BALANCE / frame.AMT_CREDIT_LIMIT_ACTUAL.where(frame.AMT_CREDIT_LIMIT_ACTUAL > 0)
                valid = utilization.notna().to_numpy() & np.isfinite(utilization.to_numpy())
                add("card_utilization_total", p[valid], utilization.loc[valid].to_numpy(dtype=float))
                add("card_utilization_observed", p[valid])
    for frame in _batches(folder / "installments_payments.parquet", ["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_PAYMENT"], size):
        # Paid-in-the-future events and future due installments cannot reveal actual outcomes.
        frame, p = valid_rows(frame, "installments_payments", (frame.DAYS_ENTRY_PAYMENT <= 0) & (frame.DAYS_INSTALMENT <= 0),
                              missing_cutoff=frame.DAYS_ENTRY_PAYMENT.isna() | frame.DAYS_INSTALMENT.isna())
        lateness = (frame.DAYS_ENTRY_PAYMENT - frame.DAYS_INSTALMENT).clip(lower=0)
        add("payment_events", p)
        add("payment_late_events", p, (lateness > 0).to_numpy(dtype=float))
        maximum("payment_max_lateness", p, lateness.to_numpy(dtype=float))
        add("payment_total_amount", p, frame.AMT_PAYMENT.fillna(0).to_numpy(dtype=float))

    def ratio(out, numerator, denominator):
        den = data.get(denominator, np.zeros(n))
        data[out] = np.divide(data.get(numerator, np.zeros(n)), den, out=np.full(n, np.nan), where=den > 0)
    ratio("bureau_late_fraction", "bureau_late_months", "bureau_known_months")
    ratio("previous_approval_rate", "previous_approved", "previous_count")
    ratio("previous_mean_credit", "previous_credit_total", "previous_credit_observed")
    ratio("pos_late_fraction", "pos_late_months", "pos_months")
    ratio("card_mean_utilization", "card_utilization_total", "card_utilization_observed")
    ratio("payment_late_fraction", "payment_late_events", "payment_events")
    observed = sum((data.get(key, np.zeros(n)) for key in ["bureau_count", "previous_count", "pos_months", "card_months", "payment_events"])) > 0
    data["history_present"] = observed.astype(float)
    result = pd.DataFrame({"SK_ID_CURR": application_ids})
    for name in HISTORY_FEATURES:
        values = data.get(name, np.full(n, np.nan)).copy()
        values[~np.isfinite(values)] = np.nan
        if name in ("bureau_debt_sum", "bureau_overdue_sum"):
            values[data.get(name + "_observed", np.zeros(n)) == 0] = np.nan
        if name != "history_present":
            values[~observed] = np.nan
        result[name] = values
    result.to_parquet(folder / "history_features.parquet", index=False)
    return result, audit


def feature_metadata() -> dict:
    descriptions = {
        "monthly_income": "AMT_INCOME_TOTAL / 12; form monthly income multiplied by 12 for annual ratios",
        "loan_amount": "AMT_CREDIT; requested cash principal or revolving credit limit",
        "monthly_payment": "AMT_ANNUITY monthly; form cash annuity or revolving principal minimum plus monthly interest estimate",
        "employment_years": "-DAYS_EMPLOYED / 365.25; sentinel 365243 becomes missing",
        "loan_to_annual_income": "loan_amount / (monthly_income * 12)",
        "payment_to_income": "monthly_payment / monthly_income",
        "product": "Cash loans -> cash; Revolving loans -> revolving",
        "employment_type": "NAME_INCOME_TYPE; form employment type with identical categories",
        "bureau_count": "Count of external credits with origination and last update no later than original application",
        "bureau_active_count": "External credits labelled Active at the original snapshot",
        "bureau_debt_sum": "Sum of observed external debt; not local platform assets",
        "bureau_max_dpd": "Maximum external credit days overdue at original snapshot",
        "bureau_overdue_sum": "Sum of observed externally overdue amounts",
        "bureau_months": "Monthly bureau records at MONTHS_BALANCE <= 0",
        "bureau_late_fraction": "Fraction of known numeric bureau statuses that are 1..5; C/X excluded from denominator",
        "bureau_max_severity": "Maximum observed numeric bureau severity 0..5",
        "previous_count": "Previous applications with DAYS_DECISION <= 0, including unsuccessful applications",
        "previous_approval_rate": "Approved previous applications / all observed previous applications",
        "previous_recent_count": "Previous applications decided between -365 and 0 days",
        "previous_mean_credit": "Mean observed AMT_CREDIT of previous applications; not confirmed disbursements",
        "pos_months": "POS/cash monthly observations at MONTHS_BALANCE <= 0",
        "pos_max_dpd": "Maximum observed POS/cash SK_DPD",
        "pos_late_fraction": "POS/cash monthly observations with SK_DPD > 0 / observed months",
        "card_months": "Card monthly observations at MONTHS_BALANCE <= 0",
        "card_max_dpd": "Maximum observed card SK_DPD",
        "card_mean_utilization": "Mean observed AMT_BALANCE / positive AMT_CREDIT_LIMIT_ACTUAL across historical monthly records",
        "payment_events": "Payment records with actual payment and scheduled due day <= 0; split payments remain distinct events",
        "payment_late_fraction": "Late actual payment events / observed payment events, not installment default rate",
        "payment_max_lateness": "Maximum max(0,DAYS_ENTRY_PAYMENT-DAYS_INSTALMENT) among eligible actual events",
        "payment_total_amount": "Sum of eligible actual payments, without claiming principal/interest separation",
        "history_present": "1 when any eligible original history exists, 0 otherwise",
    }
    return {"version": "features-v2", "form_numeric": FORM_NUMERIC, "categorical": FORM_CATEGORICAL,
        "history_numeric": HISTORY_FEATURES, "definitions": descriptions,
        "excluded": ["TARGET as predictor", "all identifiers", "names", "gender", "age/birth", "religion", "nationality", "EXT_SOURCE_1/2/3"],
        "cutoff": "Original relative application day/month 0. Bureau state requires DAYS_CREDIT<=0 and DAYS_CREDIT_UPDATE<=0; historical monthly bureau records are independently restricted to MONTHS_BALANCE<=0. Actual events after 0 excluded. No absolute calendar dates can be reconstructed.",
        "missing": "Median imputation fitted inside each training fold; missingness indicators. Unknown categories encoded safely. No blanket zero filling.",
        "history_routing": "Only linked original profiles with eligible history use history variant. New/unlinked clients use form variant; local history remains separate.",
        "limitations": ["Source data is a historical snapshot, not a present-day credit check.", "Original dataset does not provide actual interest rates/terms for current applications.", "Form installment amounts derive from declared scenario; existing debts and expenses are affordability policy inputs, not ML predictors."]}


def get_history_features(config: dict, source_application_id) -> dict:
    if source_application_id is None:
        return {}
    path = Path(config["paths"]["processed_dir"]) / "history_features.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path, filters=[("SK_ID_CURR", "=", int(source_application_id))])
    if frame.empty:
        return {}
    return {k: (None if pd.isna(v) else float(v)) for k, v in frame.iloc[0].items() if k != "SK_ID_CURR"}


def load_source_profile(config: dict, source_application_id) -> dict:
    folder = Path(config["paths"]["processed_dir"])
    raw = None
    database = Path(config["paths"]["analytics_database"])
    columns = ["SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL",
               "DAYS_EMPLOYED", "DAYS_BIRTH", "NAME_INCOME_TYPE"]
    for table in ("application_train", "application_test"):
        path = folder / (table + ".parquet")
        if database.exists():
            with closing(sqlite3.connect(database)) as connection:
                selected = ", ".join('"' + column + '"' for column in columns)
                found = pd.read_sql_query(f'SELECT {selected} FROM "{table}" WHERE SK_ID_CURR=?', connection, params=(int(source_application_id),))
        elif path.exists():
            found = pd.read_parquet(path, columns=columns, filters=[("SK_ID_CURR", "=", int(source_application_id))])
        else:
            continue
        if not found.empty:
            raw = found.iloc[0]
            break
    if raw is None:
        raise KeyError(f"Original application {source_application_id} was not found; run data preparation first")
    product = "cash" if raw.NAME_CONTRACT_TYPE == "Cash loans" else "revolving"
    policy = config["products"][product]
    amount = float(raw.AMT_CREDIT)
    payment = float(raw.AMT_ANNUITY) if pd.notna(raw.AMT_ANNUITY) else amount / 24
    # Term/rate are an explicit demonstration scenario, never asserted as original terms.
    r = float(policy["annual_rate"]) / 1200
    approximate_term = int(round(-np.log1p(-amount * r / payment) / np.log1p(r))) if r > 0 and payment > amount * r else int(round(amount / max(payment, 1)))
    term = int(np.clip(approximate_term, policy["min_term_months"], policy["max_term_months"]))
    days = raw.DAYS_EMPLOYED
    inputs = {"monthly_income": float(raw.AMT_INCOME_TOTAL) / 12, "amount": amount,
        "term_months": term, "annual_rate": float(policy["annual_rate"]), "product": product,
        "age": int(-float(raw.DAYS_BIRTH) / 365.25), "employment_type": str(raw.NAME_INCOME_TYPE),
        "employment_years": float(-days / 365.25) if pd.notna(days) and days <= 0 else None,
        "existing_monthly_debt": 0.0, "essential_expenses": 0.0,
        "initial_draw": amount if product == "revolving" else 0.0}
    return {"source_application_id": int(source_application_id), "form_inputs": inputs,
        "history": get_history_features(config, source_application_id),
        "snapshot": "Original application-relative snapshot; absolute date unavailable",
        "source_kind": "original_historical", "terms_are_demo_assumptions": True}
