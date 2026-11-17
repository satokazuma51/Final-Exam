"""Portable configuration for the local educational lending platform."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "paths": {"data_dir": "home-credit-default-risk", "processed_dir": "data/processed",
              "artifact_dir": "artifacts", "reports_dir": "reports",
              "database": "data/lending.sqlite", "analytics_database": "data/source.sqlite"},
    "training": {"seed": 42, "chunksize": 100_000, "n_jobs": 4,
                 "cv_folds": 3, "search_rows": 50_000, "permutation_rows": 2500,
                 "test_size": 0.20, "validation_size": 0.20,
                 "threshold_criterion": "maximum_validation_f1", "max_train_rows": None},
    "policy": {"version": "educational-v1", "min_age": 18, "max_age": 75,
               "max_dti": 0.50, "min_remaining_income": 1000.0},
    "demo": {"seed": 42, "reference_date": "2026-10-31", "clients": 240,
             "generator_version": "demo-v2"},
    "products": {
        "cash": {"name": "Денежный кредит", "annual_rate": 18.0, "min_amount": 1000.0,
                 "max_amount": 5_000_000.0, "min_term_months": 3, "max_term_months": 60},
        "revolving": {"name": "Возобновляемая кредитная линия", "annual_rate": 24.0,
                      "min_amount": 1000.0, "max_amount": 1_000_000.0,
                      "min_term_months": 1, "max_term_months": 60,
                      "revolving_min_principal_pct": 0.05, "revolving_min_payment": 100.0},
    },
    "currency": "у.е.",
}


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None) -> dict:
    """Read optional JSON, resolving paths relative to its directory (or project)."""
    selected = path or os.environ.get("LENDING_CONFIG")
    if selected is None:
        selected = next((p for p in (PROJECT_ROOT / "config.json", PROJECT_ROOT / "config.example.json") if p.exists()), None)
    config = copy.deepcopy(DEFAULTS)
    root = PROJECT_ROOT
    if selected:
        selected = Path(selected).expanduser().resolve()
        with selected.open(encoding="utf-8-sig") as stream:
            _merge(config, json.load(stream))
        root = selected.parent
    for key, value in config["paths"].items():
        p = Path(value).expanduser()
        config["paths"][key] = str((root / p).resolve() if not p.is_absolute() else p.resolve())
    config["config_path"] = str(selected) if selected else None
    return config


def ensure_directories(config: dict) -> None:
    for key in ("processed_dir", "artifact_dir", "reports_dir"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
    for key in ("database", "analytics_database"):
        Path(config["paths"][key]).parent.mkdir(parents=True, exist_ok=True)
