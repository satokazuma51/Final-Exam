"""Reproducible synthetic operations sampled from real source distributions.

These records never become training labels. Every saved decision is produced by
the trained model and the same committee policy used by the interactive form.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import db, ledger
from .committee import evaluate_application
from .models import load_model_bundle, score_application
from .schedules import add_months, annuity_schedule, as_date

LOGGER = logging.getLogger(__name__)
GIVEN = ["Алексей", "Анна", "Дмитрий", "Елена", "Илья", "Мария", "Никита", "Ольга", "Павел", "София"]
FAMILY = ["Орлов", "Соколов", "Лебедев", "Волков", "Морозов", "Крылов", "Белов", "Лукин"]


def _exists(config, operation_key):
    with db.connect(config) as connection:
        return connection.execute("SELECT 1 FROM transactions WHERE operation_key=?", (operation_key,)).fetchone() is not None


def _term_distribution(config):
    counts = pd.Series(dtype=float)
    source = Path(config["paths"]["data_dir"]) / "previous_application.csv"
    for chunk in pd.read_csv(source, usecols=["CNT_PAYMENT", "NAME_CONTRACT_STATUS"], chunksize=100_000):
        valid = chunk.loc[(chunk.NAME_CONTRACT_STATUS == "Approved") & chunk.CNT_PAYMENT.between(3, 60), "CNT_PAYMENT"]
        counts = counts.add(valid.round().astype(int).value_counts(), fill_value=0)
    if counts.empty:
        raise ValueError("No observed approved repayment terms were available for demo sampling")
    counts = counts.sort_index()
    return counts.index.to_numpy(dtype=int), (counts / counts.sum()).to_numpy()


def _inputs(raw, config, rng, terms, probabilities, index, reference):
    product = "cash" if raw.NAME_CONTRACT_TYPE == "Cash loans" else "revolving"
    rules = config["products"][product]
    source_amount = float(raw.AMT_CREDIT)
    amount = float(np.clip(source_amount, rules["min_amount"], rules["max_amount"]))
    term = int(np.clip(rng.choice(terms, p=probabilities), rules["min_term_months"], rules["max_term_months"]))
    income = float(raw.AMT_INCOME_TOTAL) / 12
    # Current-source terms are undisclosed. Most generated requests are capped
    # to a sampled 12-25% income payment budget under the new demo terms; 30%
    # keep the original-size request to provide genuine unaffordable examples.
    if index % 10 not in (1, 2, 3):
        rate = float(rules["annual_rate"]) / 1200
        factor = (rate / (1 - (1 + rate) ** (-term)) if rate else 1 / term) if product == "cash" else float(rules.get("revolving_min_principal_pct", .05)) + rate
        affordable_amount = income * float(rng.uniform(.12, .25)) / factor
        amount = max(float(rules["min_amount"]), min(amount, affordable_amount))
    opened = reference - timedelta(days=int(rng.integers(45, 330)))
    tenure = max(0.0, -float(raw.DAYS_EMPLOYED) / 365.25) if raw.DAYS_EMPLOYED <= 0 else 0.0
    return dict(product=product, amount=round(amount, 2), term_months=term,
                annual_rate=float(rules["annual_rate"]), monthly_income=round(income, 2),
                age=int(-float(raw.DAYS_BIRTH) / 365.25), employment_type=str(raw.NAME_INCOME_TYPE),
                employment_years=round(tenure, 2), existing_monthly_debt=round(income * float(rng.choice([0, .05, .10, .20])), 2),
                essential_expenses=round(income * .20, 2), initial_draw=round(amount * .55, 2) if product == "revolving" else 0.0,
                application_date=opened.isoformat(), intended_disbursement_date=opened.isoformat(),
                source_original_credit=source_amount,
                purpose="Учебная операция: распределения Home Credit; календарь и условия синтетические",
                comment=f"Воспроизводимый демонстрационный профиль {index + 1}")


def _save_application(config, client_id, inputs, key, source_id):
    application_id = db.submit_application(config, client_id, inputs, key, inputs["application_date"])
    existing = db.get_application(config, application_id)
    if existing.get("decision"):
        return application_id, existing["decision_details"]
    score = score_application(config, inputs, source_application_id=source_id)
    decision = evaluate_application(config, inputs, score,
                local_monthly_debt=ledger.local_monthly_debt(config, client_id, inputs["application_date"]),
                evaluation_date=inputs["application_date"])
    db.save_decision(config, application_id, decision)
    return application_id, decision


def _cash_events(config, agreement_id, inputs, reference, mode, prefix):
    opened = as_date(inputs["intended_disbursement_date"])
    if mode == 0:
        payment_date = min(opened + timedelta(days=20), reference)
        key = prefix + ":early-full"
        if not _exists(config, key):
            quote = ledger.settlement_quote(config, agreement_id, payment_date)
            ledger.record_payment(config, agreement_id, quote["amount"], payment_date, key)
        return "closed"
    schedule = annuity_schedule(inputs["amount"], inputs["annual_rate"], inputs["term_months"], opened)
    if mode == 1:
        return "arrears"
    if mode == 2:
        first = schedule[0]
        payment_date = as_date(first["due_date"]) + timedelta(days=5)
        if payment_date <= reference and not _exists(config, prefix + ":partial"):
            ledger.record_payment(config, agreement_id, round(first["payment"] / 2, 2), payment_date, prefix + ":partial")
        return "partial"
    for number, row in enumerate(schedule):
        payment_date = as_date(row["due_date"]) + timedelta(days=10 if mode == 3 else 0)
        if payment_date > reference:
            break
        key = prefix + f":payment:{number}"
        if _exists(config, key):
            continue
        quote = ledger.settlement_quote(config, agreement_id, payment_date)
        amount = min(float(row["payment"]), float(quote["amount"]))
        if amount > 0:
            ledger.record_payment(config, agreement_id, amount, payment_date, key)
    return "late_recovered" if mode == 3 else "timely"


def _revolving_events(config, agreement_id, inputs, reference, mode, prefix):
    opened = as_date(inputs["intended_disbursement_date"])
    key = prefix + ":extra-draw"
    if not _exists(config, key):
        extra = round(inputs["amount"] * .10, 2)
        ledger.drawdown(config, agreement_id, extra, opened + timedelta(days=10), key)
    if mode == 1:
        return "revolving_arrears"
    number = 1
    while add_months(opened, number) <= reference:
        day = add_months(opened, number)
        key = prefix + f":minimum:{number}"
        if not _exists(config, key):
            state = ledger.agreement_snapshot(config, agreement_id, day)
            due = sum(float(row.get("unpaid_amount", 0)) for row in state["schedule"] if as_date(row["due_date"]) <= day)
            amount = min(float(state["settlement_amount"]), due)
            if mode == 2:
                amount = round(amount * .5, 2)
            if amount > 0:
                ledger.record_payment(config, agreement_id, amount, day, key)
        number += 1
    return "revolving_drawn_and_available"


def generate_demo(config: dict, clients: int | None = None) -> dict:
    """Generate once, resumably; stable keys prevent duplicated operations."""
    db.init_db(config)
    settings = config["demo"]
    count = int(clients or settings["clients"])
    if count < 8:
        raise ValueError("Use at least eight demo clients to cover the requested scenarios")
    reference = as_date(settings["reference_date"])
    seed = int(settings["seed"])
    version = str(settings["generator_version"])
    key = f"{version}:{seed}:{reference}:{count}"
    with db.connect(config) as connection:
        finished = connection.execute("SELECT metadata_json FROM demo_metadata WHERE demo_key=?", (key,)).fetchone()
    if finished:
        return dict(json.loads(finished[0]), already_generated=True)
    # Fail before creating records if actual training has not completed.
    bundle = load_model_bundle(config)
    rng = np.random.default_rng(seed)
    columns = ["SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED", "NAME_INCOME_TYPE"]
    source = pd.read_csv(Path(config["paths"]["data_dir"]) / "application_train.csv", usecols=columns)
    sampled = source.sample(n=min(count, len(source)), random_state=seed).reset_index(drop=True)
    terms, probabilities = _term_distribution(config)
    results = dict(demo_key=key, seed=seed, generator_version=version, reference_date=str(reference),
                   clients=0, applications=0, approvals=0, rejections=0, agreements=0, scenarios={},
                   source_kind="synthetic", is_synthetic=True, model_predictions_are_real=True,
                   model_version=bundle["version"],
                   assumptions=["Fictional names; synthetic calendar, payment behavior and interest rates.",
                     "Source income, product and employment distributions; source amounts are the starting distribution.",
                     "70% of generated requests capped to a sampled 12-25% monthly-income payment budget under the demo terms; 30% retain source-size requests. Product limits also apply.",
                     "Terms sampled from approved historical applications; not recovered current-loan terms.",
                     "Unknown source employment tenure is zero in the generated current demo scenario, not an assertion about the historical record.",
                     "No synthetic operation is used as an observed ML target."])
    repeat_candidate = None
    funded_by_product = {"cash": 0, "revolving": 0}
    for index, raw in sampled.iterrows():
        prefix = f"demo:{key}:{index}"
        source_id = None if index == 0 else int(raw.SK_ID_CURR)
        name = f"Демо-клиент · {GIVEN[index % len(GIVEN)]} {FAMILY[(index // len(GIVEN)) % len(FAMILY)]} · {index + 1:03}"
        with db.connect(config) as connection:
            existing = connection.execute("SELECT client_id FROM clients WHERE json_extract(metadata_json,'$.demo_operation_key')=?", (prefix,)).fetchone()
        client_id = existing[0] if existing else db.create_client(config, name, source_id, "synthetic", True,
                         seed=seed, generator_version=version, demo_operation_key=prefix, created_at=(reference - timedelta(days=365)).isoformat())
        inputs = _inputs(raw, config, rng, terms, probabilities, index, reference)
        if index == 1:
            inputs["existing_monthly_debt"] = round(inputs["monthly_income"] * 1.10, 2)
            inputs["comment"] += "; высокая долговая нагрузка"
        application_id, decision = _save_application(config, client_id, inputs, prefix + ":application", source_id)
        results["clients"] += 1
        results["applications"] += 1
        results["approvals" if decision["decision"] == "Approved" else "rejections"] += 1
        if decision["decision"] != "Approved" or index % 5 == 0:
            continue
        agreement_id = ledger.disburse(config, application_id, inputs["intended_disbursement_date"], prefix + ":disbursement")
        results["agreements"] += 1
        mode = funded_by_product[inputs["product"]] % 6
        funded_by_product[inputs["product"]] += 1
        if inputs["product"] == "cash":
            scenario = _cash_events(config, agreement_id, inputs, reference, mode, prefix)
            if scenario == "timely" and repeat_candidate is None:
                repeat_candidate = client_id, source_id, inputs.copy()
        else:
            scenario = _revolving_events(config, agreement_id, inputs, reference, mode, prefix)
        results["scenarios"][scenario] = results["scenarios"].get(scenario, 0) + 1
        if index % 30 == 0:
            LOGGER.info("Generated demo profile %s/%s", index + 1, count)
    if repeat_candidate:
        client_id, source_id, inputs = repeat_candidate
        inputs.update(application_date=(reference - timedelta(days=1)).isoformat(),
                      intended_disbursement_date=(reference - timedelta(days=1)).isoformat(),
                      amount=float(config["products"]["cash"]["min_amount"]), term_months=12,
                      purpose="Повторная заявка клиента с операциями платформы")
        application_id, decision = _save_application(config, client_id, inputs, f"demo:{key}:repeat", source_id)
        results["applications"] += 1
        results["approvals" if decision["decision"] == "Approved" else "rejections"] += 1
        if decision["decision"] == "Approved":
            ledger.disburse(config, application_id, inputs["intended_disbursement_date"], f"demo:{key}:repeat:disbursement")
            results["agreements"] += 1
        results["repeat_application_id"] = application_id
    with db.connect(config) as connection:
        connection.execute("INSERT INTO demo_metadata VALUES(?,?,?,?,?,?)", (key, seed, version, str(reference), db.now_iso(), db.json_text(results)))
    output = Path(config["paths"]["reports_dir"]) / "demo_generation.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
