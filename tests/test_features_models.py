"""Substantive checks for units, cutoffs, pipeline leakage boundaries and scoring."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import load_config
from src.features import (FORM_FEATURES, HISTORY_FEATURES, aggregate_history,
                          application_features, estimated_payment, form_to_features)


class FeatureTests(unittest.TestCase):
    def test_form_units_and_sensitive_fields(self):
        inputs = {"monthly_income": 10000, "amount": 120000, "term_months": 12,
                  "annual_rate": 0, "employment_years": 3, "employment_type": "Working",
                  "product": "cash", "age": 55, "gender": "F", "full_name": "Demo"}
        result = form_to_features(inputs)
        self.assertAlmostEqual(result["monthly_payment"], 10000)
        self.assertAlmostEqual(result["loan_to_annual_income"], 1)
        self.assertAlmostEqual(result["payment_to_income"], 1)
        self.assertEqual(set(result), set(FORM_FEATURES))
        altered = dict(inputs, age=20, gender="M", full_name="Other")
        self.assertEqual(result, form_to_features(altered))
        self.assertEqual(form_to_features(dict(inputs, amount=60000))["monthly_payment"], 5000)

    def test_dataset_form_conversion_and_sentinel(self):
        raw = pd.DataFrame({"AMT_INCOME_TOTAL": [120000., 240000.], "AMT_CREDIT": [120000., 60000.],
            "AMT_ANNUITY": [10000., 5000.], "DAYS_EMPLOYED": [-1095.75, 365243],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"], "NAME_INCOME_TYPE": ["Working", "Pensioner"]})
        result = application_features(raw)
        self.assertEqual(result.loc[0, "monthly_income"], 10000)
        self.assertEqual(result.loc[0, "employment_years"], 3)
        self.assertTrue(pd.isna(result.loc[1, "employment_years"]))
        self.assertEqual(result.loc[1, "product"], "revolving")

    def test_revolving_minimum_keeps_interest(self):
        inputs = {"amount": 10000, "initial_draw": 50, "annual_rate": 24, "product": "revolving"}
        self.assertEqual(estimated_payment(inputs), 51)

    def test_history_cutoff_missingness_and_no_row_multiplication(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            fixtures = {
                "bureau": {"SK_ID_CURR": [1, 1, 1], "SK_ID_BUREAU": [10, 11, 12], "DAYS_CREDIT": [-20, 5, -25],
                    "DAYS_CREDIT_UPDATE": [-1, -1, 2],
                    "CREDIT_ACTIVE": ["Active", "Active", "Active"], "AMT_CREDIT_SUM_DEBT": [100., 99999., 88888.],
                    "CREDIT_DAY_OVERDUE": [2., 999., 888.], "AMT_CREDIT_SUM_OVERDUE": [5., 99999., 88888.]},
                "bureau_balance": {"SK_ID_BUREAU": [10, 10, 99, 12], "MONTHS_BALANCE": [-1, 1, -1, -1], "STATUS": ["1", "5", "5", "0"]},
                "previous_application": {"SK_ID_CURR": [1, 1], "DAYS_DECISION": [-30, 10],
                    "NAME_CONTRACT_STATUS": ["Approved", "Refused"], "AMT_CREDIT": [50., 999.]},
                "POS_CASH_balance": {"SK_ID_CURR": [1, 1], "MONTHS_BALANCE": [-1, 1], "SK_DPD": [3., 999.]},
                "credit_card_balance": {"SK_ID_CURR": [1, 1], "MONTHS_BALANCE": [-1, 1], "SK_DPD": [4., 999.],
                    "AMT_BALANCE": [50., 999.], "AMT_CREDIT_LIMIT_ACTUAL": [100., 100.]},
                "installments_payments": {"SK_ID_CURR": [1, 1, 1], "DAYS_INSTALMENT": [-10, -10, 5],
                    "DAYS_ENTRY_PAYMENT": [-8, 2, -1], "AMT_PAYMENT": [20., 999., 999.]},
            }
            for name, values in fixtures.items():
                pd.DataFrame(values).to_parquet(folder / (name + ".parquet"), index=False)
            config = load_config()
            config["paths"]["processed_dir"] = str(folder)
            config["training"]["chunksize"] = 1
            frame, audit = aggregate_history(config, np.array([1, 2]))
            self.assertEqual(len(frame), 2)
            one, two = frame.iloc[0], frame.iloc[1]
            self.assertEqual(one.bureau_count, 1)
            self.assertEqual(one.bureau_debt_sum, 100)
            self.assertEqual(audit["bureau"]["excluded_future_update_rows"], 1)
            self.assertEqual(one.bureau_max_severity, 1)
            self.assertEqual(one.bureau_months, 2)  # Known past months survive a later parent-state update.
            self.assertEqual(one.previous_count, 1)
            self.assertEqual(one.payment_events, 1)
            self.assertEqual(one.payment_total_amount, 20)
            self.assertEqual(one.payment_max_lateness, 2)
            self.assertEqual(one.card_mean_utilization, .5)
            self.assertEqual(two.history_present, 0)
            self.assertTrue(pd.isna(two.bureau_debt_sum))
            self.assertEqual(audit["bureau_balance"]["missing_parent_rows"], 1)
            self.assertEqual(audit["bureau_balance"]["excluded_after_cutoff"], 1)
            self.assertEqual(audit["installments_payments"]["excluded_after_cutoff"], 2)


class ModelTests(unittest.TestCase):
    def test_threshold_uses_probability_predictions(self):
        from src.models import evaluate_predictions, select_threshold
        y = np.array([0, 0, 1, 1])
        probability = np.array([.1, .2, .3, .8])
        threshold, rule = select_threshold(y, probability)
        self.assertAlmostEqual(threshold, .3)
        metrics = evaluate_predictions(y, probability, threshold)
        self.assertEqual(metrics["f1"], 1)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(rule["criterion"], "maximum_validation_f1")

    def test_pipeline_unknown_category_and_train_fitted_imputation(self):
        from src.models import make_pipeline
        frame = pd.DataFrame([form_to_features({"monthly_income": 1000 + i * 100,
            "amount": 2000 + i * 50, "term_months": 12, "annual_rate": 10,
            "employment_years": None if i == 0 else i, "employment_type": "Working"}) for i in range(30)])
        model = make_pipeline("form", "LogisticRegression", load_config())
        model.fit(frame, np.arange(30) % 2)
        medians = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_.copy()
        altered = frame.iloc[[0]].copy()
        altered.loc[:, "employment_type"] = "Unseen occupation"
        altered.loc[:, "monthly_income"] = 100000000
        self.assertTrue(np.isfinite(model.predict_proba(altered)).all())
        np.testing.assert_array_equal(medians, model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_)

    def test_missing_model_does_not_produce_fabricated_score(self):
        from src.models import score_application
        with tempfile.TemporaryDirectory() as directory:
            config = load_config()
            config["paths"]["artifact_dir"] = directory
            with self.assertRaises(FileNotFoundError):
                score_application(config, {"monthly_income": 1000, "amount": 1000})


class PreparationSafetyTests(unittest.TestCase):
    def test_interrupted_force_rebuild_cannot_reuse_previous_success(self):
        from src.data_pipeline import SOURCE_TABLES, prepare_data
        from src.features import feature_metadata
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config()
            config["paths"].update({"data_dir": str(root / "source"), "processed_dir": str(root / "prepared"),
                "artifact_dir": str(root / "artifacts"), "reports_dir": str(root / "reports"),
                "database": str(root / "operational.sqlite"), "analytics_database": str(root / "source.sqlite")})
            (root / "source").mkdir()
            (root / "prepared").mkdir()
            for name in SOURCE_TABLES:
                (root / "source" / (name + ".csv")).write_text("value\n1\n", encoding="utf-8")
            files = list((root / "source").glob("*.csv"))
            fingerprint = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files}
            previous = {"source_fingerprint": fingerprint, "feature_version": feature_metadata()["version"]}
            (root / "prepared" / "profile.json").write_text(json.dumps(previous), encoding="utf-8")
            (root / "prepared" / "application_features.parquet").write_bytes(b"previous-success-fixture")
            (root / "source.sqlite").touch()
            with patch("src.data_pipeline._import_table", side_effect=RuntimeError("interrupted import")) as importer:
                with self.assertRaisesRegex(RuntimeError, "interrupted import"):
                    prepare_data(config, force=True)
                self.assertTrue((root / "prepared" / "preparation_in_progress.json").exists())
                with self.assertRaisesRegex(RuntimeError, "interrupted import"):
                    prepare_data(config)
                self.assertEqual(importer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
