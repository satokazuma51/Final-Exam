"""Reproducible model comparison, held-out evaluation and deterministic local scoring."""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "data" / ".matplotlib"))
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names", category=UserWarning)

import joblib
import sklearn  # Load the bundled Windows OpenMP runtime before the LightGBM DLL.
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score, precision_recall_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import ensure_directories
from .data_pipeline import process_memory_mb
from .features import (FORM_CATEGORICAL, FORM_FEATURES, FORM_NUMERIC, HISTORY_FEATURES,
                       feature_metadata, form_to_features, get_history_features)

LOG = logging.getLogger(__name__)


def _signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


def make_pipeline(variant: str, model_name: str, config: dict) -> Pipeline:
    numeric = FORM_NUMERIC + (HISTORY_FEATURES if variant == "history" else [])
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
             ("log", FunctionTransformer(_signed_log, feature_names_out="one-to-one"))]
    if model_name == "LogisticRegression":
        steps.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer([
        ("numeric", Pipeline(steps), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=10), FORM_CATEGORICAL),
    ], sparse_threshold=1.0, verbose_feature_names_out=False)
    seed, jobs = int(config["training"]["seed"]), int(config["training"]["n_jobs"])
    if model_name == "DummyClassifier":
        estimator = DummyClassifier(strategy="prior", random_state=seed)
    elif model_name == "LogisticRegression":
        estimator = LogisticRegression(C=1., solver="lbfgs", max_iter=500, random_state=seed)
    elif model_name == "LightGBM":
        estimator = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=31,
            min_child_samples=60, colsample_bytree=.9, reg_lambda=1., random_state=seed,
            n_jobs=jobs, deterministic=True, force_col_wise=True, verbosity=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def select_threshold(y_true, probability) -> tuple[float, dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if not len(thresholds):
        return .5, {"criterion": "maximum_validation_f1", "f1": 0.}
    f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                   out=np.zeros(len(thresholds)), where=(precision[:-1] + recall[:-1]) > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), {"criterion": "maximum_validation_f1", "f1": float(f1[best]),
        "precision": float(precision[best]), "recall": float(recall[best])}


def evaluate_predictions(y_true, probabilities, threshold: float) -> dict:
    predicted = np.asarray(probabilities) >= threshold
    observed, estimated = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    return {"roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "pr_auc_method": "sklearn.average_precision_score: step-weighted average precision, not trapezoidal PR area",
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": [["TN", "FP"], ["FN", "TP"]],
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "threshold": float(threshold),
        "calibration_curve": {"predicted": estimated.tolist(), "observed": observed.tolist()}}


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def train_models(config: dict, force: bool = False) -> dict:
    ensure_directories(config)
    artifact = Path(config["paths"]["artifact_dir"])
    processed = Path(config["paths"]["processed_dir"])
    feature_path = processed / "application_features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError("Prepared features are missing. Run: python -m src.cli prepare")
    if any((processed / marker).exists() for marker in ("preparation_in_progress.json", "feature_build_in_progress.json")):
        raise RuntimeError("Source preparation or feature rebuild is incomplete; finish preparation before training")
    profile_path = processed / "profile.json"
    if not profile_path.exists() or json.loads(profile_path.read_text(encoding="utf-8")).get("feature_version") != feature_metadata()["version"]:
        raise ValueError("Prepared feature version is stale. Run: python -m src.cli prepare --features-only")
    metrics_path = artifact / "metrics.json"
    fingerprint = {"feature_size": feature_path.stat().st_size, "feature_mtime_ns": feature_path.stat().st_mtime_ns,
                   "feature_version": feature_metadata()["version"],
                   "training": config["training"]}
    if not force and metrics_path.exists() and (artifact / "model_bundle.joblib").exists():
        prior = json.loads(metrics_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") == fingerprint:
            return prior
    started = time.perf_counter()
    initial_memory = process_memory_mb()
    frame = pd.read_parquet(feature_path)
    frame = frame.loc[frame.TARGET.notna()].reset_index(drop=True)
    if frame.SK_ID_CURR.duplicated().any():
        raise ValueError("Duplicate source applications must not cross evaluation splits")
    seed = int(config["training"]["seed"])
    max_rows = config["training"].get("max_train_rows")
    if max_rows and len(frame) > int(max_rows):
        frame, _ = train_test_split(frame, train_size=int(max_rows), stratify=frame.TARGET, random_state=seed)
        frame = frame.reset_index(drop=True)
    y = frame.TARGET.astype(int)
    # There is no universal person identifier or absolute chronology in the supplied data.
    dev, holdout = train_test_split(np.arange(len(frame)), test_size=float(config["training"]["test_size"]), stratify=y, random_state=seed)
    train, valcal = train_test_split(dev, test_size=float(config["training"]["validation_size"]) / (1 - float(config["training"]["test_size"])), stratify=y.iloc[dev], random_state=seed + 1)
    calibration, validation = train_test_split(valcal, test_size=.5, stratify=y.iloc[valcal], random_state=seed + 2)
    split_values = np.full(len(frame), "train", dtype=object)
    for name, ids in [("calibration", calibration), ("validation", validation), ("holdout", holdout)]:
        split_values[ids] = name
    pd.DataFrame({"SK_ID_CURR": frame.SK_ID_CURR, "split": split_values}).to_parquet(artifact / "split_assignments.parquet", index=False)
    search_rows = min(len(train), int(config["training"]["search_rows"]))
    if search_rows < len(train):
        search, _ = train_test_split(train, train_size=search_rows, stratify=y.iloc[train], random_state=seed + 3)
    else:
        search = train
    version = "homecredit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {"version": version, "fingerprint": fingerprint, "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": [], "selected": {}, "feature_importance": {}, "calibration": {}, "search": [],
        "split": {"train": len(train), "calibration": len(calibration), "validation": len(validation), "holdout": len(holdout),
            "seed": seed, "method": "stratified source-application split; identical IDs for both variants",
            "limitation": "SK_ID_CURR is an application identifier. No universal person identity or absolute dates are supplied; unknown repeated persons cannot be grouped.",
            "search_rows": search_rows, "cv_folds": int(config["training"]["cv_folds"]),
            "holdout_access": "Holdout predictions evaluated after fitting/calibration/threshold/model selection; no holdout-based tuning"},
        "library_versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__},
        "memory_before": initial_memory,
        "calibration_method": "Sigmoid calibration fitted on independent calibration split; thresholds and model selection use separate validation split",
        "risk_boundaries": "Low below half selected threshold; elevated between half threshold and threshold; high at or above threshold. Educational categories.",
        "limitations": ["Outcome is payment difficulties, not guaranteed non-repayment.",
            "Raw historical snapshots cannot establish current debts or current creditworthiness.",
            "Serving form features omit undisclosed external scores and sensitive fields.",
            "Validation F1 optimizes identification of payment difficulties, not economic lending profit or a statutory policy.",
            "Form use and demo terms may differ from source distributions; monitor calibration and drift before real use."]}
    candidates = {}
    cv = StratifiedKFold(n_splits=int(config["training"]["cv_folds"]), shuffle=True, random_state=seed)
    # Selection occurs without using the final holdout.
    for variant in ("form", "history"):
        cols = FORM_FEATURES + (HISTORY_FEATURES if variant == "history" else [])
        x = frame[cols]
        candidates[variant] = []
        for name in ("DummyClassifier", "LogisticRegression", "LightGBM"):
            fit_started = time.perf_counter()
            model = make_pipeline(variant, name, config)
            if name != "DummyClassifier":
                grid = {"model__C": [.3, 1.]} if name == "LogisticRegression" else {"model__num_leaves": [15, 31]}
                searcher = GridSearchCV(model, grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=False, error_score="raise", return_train_score=False)
                searcher.fit(x.iloc[search], y.iloc[search])
                model.set_params(**searcher.best_params_)
                for params, score, std in zip(searcher.cv_results_["params"], searcher.cv_results_["mean_test_score"], searcher.cv_results_["std_test_score"]):
                    report["search"].append({"variant": variant, "model": name, "params": params, "cv_roc_auc": float(score), "cv_std": float(std)})
            model.fit(x.iloc[train], y.iloc[train])
            calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid", n_jobs=1)
            calibrated.fit(x.iloc[calibration], y.iloc[calibration])
            val_prob = calibrated.predict_proba(x.iloc[validation])[:, 1]
            threshold, criterion = select_threshold(y.iloc[validation], val_prob)
            validation_metrics = evaluate_predictions(y.iloc[validation], val_prob, threshold)
            record = {"variant": variant, "model": name, "training_seconds": time.perf_counter() - fit_started,
                      "threshold": threshold, "threshold_selection": criterion,
                      "validation_roc_auc": validation_metrics["roc_auc"], "validation": validation_metrics,
                      "params": model.named_steps["model"].get_params(), "memory": process_memory_mb()}
            candidates[variant].append({"base": model, "calibrated": calibrated, "record": record})
            LOG.info("Fitted %s/%s validation AUC %.4f in %.1fs", variant, name, validation_metrics["roc_auc"], record["training_seconds"])
        # Prefer the strongest non-dummy validation discrimination, predetermined criterion.
        selected = max(candidates[variant][1:], key=lambda item: item["record"]["validation_roc_auc"])
        report["selected"][variant] = {"model": selected["record"]["model"], "threshold": selected["record"]["threshold"],
                                          "criterion": "highest validation ROC-AUC among trained candidates"}
    bundle = {"version": version, "feature_version": feature_metadata()["version"], "variants": {}, "metadata": feature_metadata()}
    # Final evaluation begins here; choices above are already fixed.
    for variant, items in candidates.items():
        cols = FORM_FEATURES + (HISTORY_FEATURES if variant == "history" else [])
        x = frame[cols]
        selected_name = report["selected"][variant]["model"]
        for item in items:
            base, calibrated, record = item["base"], item["calibrated"], item["record"]
            inference_started = time.perf_counter()
            probabilities = calibrated.predict_proba(x.iloc[holdout])[:, 1]
            record["inference_seconds"] = time.perf_counter() - inference_started
            record["inference_rows"] = len(holdout)
            record.update(evaluate_predictions(y.iloc[holdout], probabilities, record["threshold"]))
            path = artifact / f"{variant}_{record['model']}.joblib"
            joblib.dump({"base": base, "calibrated": calibrated, "threshold": record["threshold"], "features": cols}, path, compress=3)
            record["model_bytes"] = path.stat().st_size
            report["models"].append(record)
            if record["model"] == selected_name:
                bundle["variants"][variant] = {"base": base, "calibrated": calibrated, "threshold": record["threshold"], "features": cols, "model": record["model"]}
                report["selected"][variant].update({key: record[key] for key in ["roc_auc", "pr_auc", "f1", "brier_score", "precision", "recall", "confusion_matrix"]})
                sample_n = min(int(config["training"]["permutation_rows"]), len(holdout))
                sample_ids, _ = train_test_split(holdout, train_size=sample_n, stratify=y.iloc[holdout], random_state=seed + 4) if sample_n < len(holdout) else (holdout, [])
                importance = permutation_importance(calibrated, x.iloc[sample_ids], y.iloc[sample_ids], scoring="roc_auc", n_repeats=2, random_state=seed, n_jobs=1)
                report["feature_importance"][variant] = sorted([{"feature": c, "importance": float(v), "std": float(s)} for c, v, s in zip(cols, importance.importances_mean, importance.importances_std)], key=lambda entry: entry["importance"], reverse=True)
                raw_prob = base.predict_proba(x.iloc[holdout])[:, 1]
                report["calibration"][variant] = {"holdout_brier_before": float(brier_score_loss(y.iloc[holdout], raw_prob)),
                    "holdout_brier_after": record["brier_score"], "holdout_curve": record["calibration_curve"]}
                errors = pd.DataFrame({"SK_ID_CURR": frame.iloc[holdout].SK_ID_CURR.to_numpy(), "TARGET": y.iloc[holdout].to_numpy(), "probability": probabilities})
                errors["predicted"] = (errors.probability >= record["threshold"]).astype(int)
                errors["error_type"] = np.select([(errors.TARGET == 0) & (errors.predicted == 1), (errors.TARGET == 1) & (errors.predicted == 0)], ["false_positive", "false_negative"], default="correct")
                errors["product"] = x.iloc[holdout]["product"].to_numpy()
                errors.to_parquet(artifact / f"{variant}_holdout_predictions.parquet", index=False)
                report.setdefault("error_analysis", {})[variant] = {str(k): int(v) for k, v in errors.error_type.value_counts().items()}
    joblib.dump(bundle, artifact / "model_bundle.joblib", compress=3)
    report["model_bundle_bytes"] = (artifact / "model_bundle.joblib").stat().st_size
    report["seconds"] = time.perf_counter() - started
    report["memory_after"] = process_memory_mb()
    _write_json(metrics_path, report)
    _write_json(artifact / "feature_metadata.json", feature_metadata())
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in report["models"]]).to_parquet(artifact / "experiment_results.parquet", index=False)
    archive = artifact / "versions" / version
    archive.mkdir(parents=True, exist_ok=False)
    filenames = ["model_bundle.joblib", "metrics.json", "feature_metadata.json", "split_assignments.parquet", "experiment_results.parquet"]
    filenames += [f"{variant}_{name}.joblib" for variant in ("form", "history") for name in ("DummyClassifier", "LogisticRegression", "LightGBM")]
    for filename in filenames:
        shutil.copy2(artifact / filename, archive / filename)
    return report


@lru_cache(maxsize=3)
def _cached_bundle(path: str, modified_ns: int):
    return joblib.load(path)


def load_model_bundle(config: dict) -> dict:
    path = Path(config["paths"]["artifact_dir"]) / "model_bundle.joblib"
    if not path.exists():
        raise FileNotFoundError("Trained model is unavailable. Run: .\\.venv\\Scripts\\python.exe -m src.cli train")
    bundle = _cached_bundle(str(path), path.stat().st_mtime_ns)
    if bundle.get("feature_version") != feature_metadata()["version"]:
        raise ValueError("Model feature version is incompatible; prepare and retrain the models")
    return bundle


def _local_contributions(base: Pipeline, frame: pd.DataFrame) -> list[dict]:
    transformed = base.named_steps["preprocess"].transform(frame)
    names = base.named_steps["preprocess"].get_feature_names_out()
    model = base.named_steps["model"]
    if isinstance(model, lgb.LGBMClassifier):
        contributions = model.booster_.predict(transformed, pred_contrib=True)
        if hasattr(contributions, "toarray"):
            contributions = contributions.toarray()
        values = np.asarray(contributions)[0, :-1]
        method = "TreeSHAP contribution to uncalibrated LightGBM log-odds; sigmoid calibration follows"
    elif isinstance(model, LogisticRegression):
        values = np.asarray(transformed.multiply(model.coef_[0]).toarray() if hasattr(transformed, "multiply") else transformed * model.coef_[0])[0]
        method = "Linear contribution to uncalibrated logistic log-odds; sigmoid calibration follows"
    else:
        return []
    order = np.argsort(np.abs(values))[::-1][:8]
    return [{"feature": str(names[i]), "contribution": float(values[i]), "direction": "increases" if values[i] > 0 else "decreases", "method": method} for i in order]


def score_application(config: dict, inputs: dict, source_application_id=None, bundle: dict | None = None) -> dict:
    bundle = bundle or load_model_bundle(config)
    features = form_to_features(inputs, config)
    history = get_history_features(config, source_application_id)
    use_history = bool(history and history.get("history_present", 0) == 1)
    variant = "history" if use_history else "form"
    selected = bundle["variants"][variant]
    if use_history:
        features.update({k: (np.nan if v is None else v) for k, v in history.items()})
    frame = pd.DataFrame([features], columns=selected["features"])
    probability = float(selected["calibrated"].predict_proba(frame)[0, 1])
    threshold = float(selected["threshold"])
    return {"probability": probability, "model_version": bundle["version"] + "/" + variant,
        "feature_snapshot": {k: (None if pd.isna(v) else v) for k, v in features.items()},
        "history_used": use_history, "threshold": threshold,
        "risk_category": "high" if probability >= threshold else "elevated" if probability >= threshold / 2 else "low",
        "contributions": _local_contributions(selected["base"], frame), "model_variant": variant,
        "explanation_note": "Local contributions describe the model prediction, not proven causes; values refer to uncalibrated log-odds."}
