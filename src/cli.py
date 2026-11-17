"""Expensive preparation, training and demo work stays outside Streamlit reruns."""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from src.config import ensure_directories, load_config


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Local educational Home Credit lending platform")
    cli.add_argument("--config", help="JSON settings; relative paths resolve beside that file")
    cli.add_argument("--data-dir", help="Override the directory containing the original CSV files")
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="Create the operational schema and product table")
    prepare = commands.add_parser("prepare", help="Stream source CSVs to Parquet/SQLite and aggregate history")
    prepare.add_argument("--force", action="store_true", help="Rebuild derived source data (never change CSVs)")
    prepare.add_argument("--features-only", action="store_true", help="Rebuild features from an already completed Parquet import")
    train = commands.add_parser("train", help="Fit and evaluate the real labeled-data experiments")
    train.add_argument("--force", action="store_true")
    train.add_argument("--max-rows", type=int, help="Optional reproducible labeled sample; recorded in metadata")
    train.add_argument("--search-rows", type=int, help="Bounded hyperparameter-search sample")
    train.add_argument("--threads", type=int, help="CPU workers (default from configuration)")
    demo = commands.add_parser("demo", help="Generate the reproducible, separately labeled operational portfolio")
    demo.add_argument("--clients", type=int, help="Number of demo clients; no regeneration at app startup")
    commands.add_parser("reports", help="Generate the research report, actual-result figures and defense PPTX")
    commands.add_parser("verify", help="Run substantive accounting, model and Streamlit checks")
    commands.add_parser("submission", help="Write optional application_test probability CSV")
    return cli


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.data_dir:
        config["paths"]["data_dir"] = str(Path(args.data_dir).expanduser().resolve())
    ensure_directories(config)
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "platform.log", encoding="utf-8")])
    try:
        if args.command == "init-db":
            from src.db import init_db
            init_db(config)
            result = {"database": config["paths"]["database"], "status": "initialized"}
        elif args.command == "prepare":
            from src.data_pipeline import prepare_data, rebuild_features
            if args.features_only and args.force:
                raise ValueError("Choose either --features-only or --force")
            result = rebuild_features(config) if args.features_only else prepare_data(config, force=args.force)
        elif args.command == "train":
            from src.models import train_models
            for argument, key in ((args.max_rows, "max_train_rows"), (args.search_rows, "search_rows"),
                                  (args.threads, "n_jobs")):
                if argument is not None:
                    if argument <= 0:
                        raise ValueError(f"{key} must be positive")
                    config["training"][key] = argument
            result = train_models(config, force=args.force)
        elif args.command == "demo":
            from src.demo_data import generate_demo
            result = generate_demo(config, clients=args.clients)
        elif args.command == "reports":
            from src.research_report import build_reports
            result = build_reports(config)
        elif args.command == "submission":
            from src.research_report import export_submission
            result = export_submission(config)
        else:
            report = Path(config["paths"]["reports_dir"]) / "verification.xml"
            environment = os.environ.copy()
            if (Path(config["paths"]["artifact_dir"]) / "model_bundle.joblib").exists():
                environment.setdefault("RUN_REAL_MODEL_UI", "1")
            return subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", f"--junitxml={report}"],
                                  cwd=Path(__file__).resolve().parents[1], env=environment, check=False).returncode
        if args.command == "prepare":
            result = {"tables": {name: {"rows": item["rows"], "columns": item["columns"] if isinstance(item["columns"], int) else len(item["columns"])} for name, item in result["tables"].items()},
                      "feature_rows": result["feature_rows"], "target_balance": result["target_balance"],
                      "seconds": result["seconds"], "profile": str(Path(config["paths"]["processed_dir"]) / "profile.json")}
        elif args.command == "train":
            result = {key: result[key] for key in ["version", "selected", "split", "seconds", "model_bundle_bytes", "memory_before", "memory_after"]}
            result["metrics_file"] = str(Path(config["paths"]["artifact_dir"]) / "metrics.json")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    except Exception:
        logging.exception("Command failed; inspect the logged traceback before retrying")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
