"""Chunked, reproducible source import, EDA and application-level feature preparation."""
from __future__ import annotations

import ctypes
from contextlib import closing
import json
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import ensure_directories
from .features import aggregate_history, application_features, feature_metadata

LOG = logging.getLogger(__name__)
SOURCE_TABLES = ["application_train", "application_test", "bureau", "bureau_balance",
    "previous_application", "POS_CASH_balance", "credit_card_balance", "installments_payments",
    "HomeCredit_columns_description", "sample_submission"]


def process_memory_mb() -> dict:
    """Measured Windows working set including native arrays; portable /proc fallback."""
    if hasattr(ctypes, "windll"):
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return {"working_set_mb": counters.WorkingSetSize / 1024 ** 2,
                    "peak_working_set_mb": counters.PeakWorkingSetSize / 1024 ** 2}
    try:
        status = Path("/proc/self/status").read_text()
        out = {}
        for line in status.splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                out["working_set_mb" if line.startswith("VmRSS:") else "peak_working_set_mb"] = int(line.split()[1]) / 1024
        return out
    except OSError:
        return {"working_set_mb": None, "peak_working_set_mb": None}


def _json_write(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _optimize(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame:
        if pd.api.types.is_object_dtype(frame[column]) or isinstance(frame[column].dtype, pd.StringDtype):
            frame[column] = frame[column].astype("category")
        elif column.startswith("SK_ID_"):
            frame[column] = frame[column].astype("int32")
        elif pd.api.types.is_float_dtype(frame[column]) and not column.startswith("AMT_"):
            frame[column] = frame[column].astype("float32")
    return frame


def _import_table(config: dict, table: str, connection: sqlite3.Connection) -> dict:
    source = Path(config["paths"]["data_dir"]) / (table + ".csv")
    destination = Path(config["paths"]["processed_dir"]) / (table + ".parquet")
    encoding = "latin1" if table == "HomeCredit_columns_description" else "utf-8"
    sample = pd.read_csv(source, nrows=5000, encoding=encoding)
    dtypes = {c: ("int64" if c.startswith("SK_ID_") else "float64") if pd.api.types.is_numeric_dtype(sample[c]) else "string" for c in sample}
    # An unnamed first dictionary column is retained as an original row identifier.
    report = {"rows": 0, "columns": list(sample.columns), "source_bytes": source.stat().st_size,
              "missing": {c: 0 for c in sample}, "numeric": {}, "categories": {},
              "types": {}, "first_chunk_memory_before_mb": None, "first_chunk_memory_after_mb": None}
    numeric = {}
    bounds = {}
    for c in sample.select_dtypes(include=np.number):
        values = sample[c].dropna()
        if len(values):
            q1, q3 = values.quantile([.25, .75])
            bounds[c] = (float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1)))
            numeric[c] = [0, 0., 0., np.inf, -np.inf, 0]
    connection.execute(f'DROP TABLE IF EXISTS "{table}"')
    connection.commit()
    writer = None
    started = time.perf_counter()
    try:
        for chunk_number, frame in enumerate(pd.read_csv(source, chunksize=int(config["training"]["chunksize"]), dtype=dtypes, encoding=encoding)):
            report["rows"] += len(frame)
            for c, missing in frame.isna().sum().items():
                report["missing"][c] += int(missing)
            for c, totals in numeric.items():
                values = frame[c].dropna().to_numpy(dtype=float)
                if len(values):
                    totals[0] += len(values)
                    totals[1] += float(values.sum())
                    totals[2] += float(np.square(values).sum())
                    totals[3] = min(totals[3], float(values.min()))
                    totals[4] = max(totals[4], float(values.max()))
                    low, high = bounds[c]
                    totals[5] += int(((values < low) | (values > high)).sum())
            for c in sample.select_dtypes(exclude=np.number):
                counts = frame[c].value_counts(dropna=True)
                category_counts = report["categories"].setdefault(c, {})
                for value, count in counts.items():
                    category_counts[str(value)] = category_counts.get(str(value), 0) + int(count)
            if chunk_number == 0:
                report["first_chunk_memory_before_mb"] = float(frame.memory_usage(deep=True).sum() / 1024 ** 2)
            frame = _optimize(frame)
            if chunk_number == 0:
                report["first_chunk_memory_after_mb"] = float(frame.memory_usage(deep=True).sum() / 1024 ** 2)
                report["types"] = {c: str(t) for c, t in frame.dtypes.items()}
            arrow = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, arrow.schema, compression="zstd")
            elif arrow.schema != writer.schema:
                arrow = arrow.cast(writer.schema)
            writer.write_table(arrow)
            frame.to_sql(table, connection, if_exists="append", index=False, chunksize=10_000)
            if chunk_number % 20 == 0:
                LOG.info("Imported %s: %s rows", table, report["rows"])
    finally:
        if writer is not None:
            writer.close()
    for c in sample.columns:
        if c.startswith("SK_ID_"):
            connection.execute(f'CREATE INDEX IF NOT EXISTS "ix_{table}_{c}" ON "{table}" ("{c}")')
    connection.commit()
    for c, (count, total, total_square, minimum, maximum, outliers) in numeric.items():
        report["numeric"][c] = {"count": count, "mean": total / count if count else None,
            "std": float(np.sqrt(max(0, total_square / count - (total / count) ** 2))) if count else None,
            "min": minimum if count else None, "max": maximum if count else None,
            "outliers": outliers, "outlier_method": "Full-table count outside 1.5 IQR bounds estimated on first 5000 rows",
            "outlier_bounds": list(bounds[c])}
    report["seconds"] = time.perf_counter() - started
    report["parquet_bytes"] = destination.stat().st_size
    report["process_memory"] = process_memory_mb()
    return report


def _unmatched_audit(connection: sqlite3.Connection) -> dict:
    connection.execute("CREATE TABLE IF NOT EXISTS source_application_ids (SK_ID_CURR INTEGER PRIMARY KEY)")
    connection.execute("DELETE FROM source_application_ids")
    connection.execute("INSERT INTO source_application_ids SELECT SK_ID_CURR FROM application_train UNION SELECT SK_ID_CURR FROM application_test")
    connection.commit()
    queries = {}
    for table in ["bureau", "previous_application", "POS_CASH_balance", "credit_card_balance", "installments_payments"]:
        queries[table + "_without_current_application"] = f'SELECT COUNT(*) AS unmatched FROM "{table}" c LEFT JOIN source_application_ids a ON a.SK_ID_CURR=c.SK_ID_CURR WHERE a.SK_ID_CURR IS NULL'
    queries["bureau_balance_without_bureau"] = "SELECT COUNT(*) AS unmatched FROM bureau_balance c LEFT JOIN bureau b ON b.SK_ID_BUREAU=c.SK_ID_BUREAU WHERE b.SK_ID_BUREAU IS NULL"
    for table in ["POS_CASH_balance", "credit_card_balance", "installments_payments"]:
        queries[table + "_without_previous_application"] = f'SELECT COUNT(*) AS unmatched FROM "{table}" c LEFT JOIN previous_application p ON p.SK_ID_PREV=c.SK_ID_PREV WHERE p.SK_ID_PREV IS NULL'
    return {name: int(pd.read_sql_query(sql, connection).iloc[0, 0]) for name, sql in queries.items()}


def prepare_data(config: dict, force: bool = False) -> dict:
    ensure_directories(config)
    folder = Path(config["paths"]["processed_dir"])
    profile_path = folder / "profile.json"
    marker = folder / "preparation_in_progress.json"
    sources = [Path(config["paths"]["data_dir"]) / (name + ".csv") for name in SOURCE_TABLES]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(missing))
    fingerprint = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sources}
    if profile_path.exists() and not force and not marker.exists():
        previous = json.loads(profile_path.read_text(encoding="utf-8"))
        if previous.get("source_fingerprint") == fingerprint and (folder / "application_features.parquet").exists() and Path(config["paths"]["analytics_database"]).exists():
            if previous.get("feature_version") == feature_metadata()["version"] and not (folder / "feature_build_in_progress.json").exists():
                return previous
            if all((folder / (name + ".parquet")).exists() for name in SOURCE_TABLES):
                return rebuild_features(config)
    started = time.perf_counter()
    report = {"source_fingerprint": fingerprint, "tables": {}, "currency": "unspecified generic units",
              "memory_before": process_memory_mb(), "cutoff": "Original application day/month 0"}
    # A previous successful profile must never mask an interrupted destructive
    # rebuild. Keep the marker on failure and remove it only after all outputs.
    _json_write(marker, {"source_fingerprint": fingerprint, "feature_version": feature_metadata()["version"]})
    with closing(sqlite3.connect(config["paths"]["analytics_database"])) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA cache_size=-65536")
        for table in SOURCE_TABLES:
            LOG.info("Preparing %s", table)
            report["tables"][table] = _import_table(config, table, connection)
            _json_write(folder / "preparation_progress.json", report)
        LOG.info("Auditing unmatched historical records")
        report["unmatched_joins"] = _unmatched_audit(connection)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    report = _build_features(config, report)
    report["seconds"] = time.perf_counter() - started
    report["memory_after"] = process_memory_mb()
    _json_write(profile_path, report)
    marker.unlink()
    LOG.info("Data preparation complete in %.1f seconds", report["seconds"])
    return report


def rebuild_features(config: dict) -> dict:
    """Recompute only cutoff-aware features from already imported source Parquet."""
    ensure_directories(config)
    folder = Path(config["paths"]["processed_dir"])
    if (folder / "preparation_in_progress.json").exists():
        raise RuntimeError("A raw-source preparation did not complete. Run prepare again before rebuilding features")
    profile_path = folder / "profile.json"
    if not profile_path.exists():
        raise FileNotFoundError("Source profile is missing; run prepare before prepare --features-only")
    missing = [name for name in SOURCE_TABLES if not (folder / (name + ".parquet")).exists()]
    if missing:
        raise FileNotFoundError("Imported source Parquet is missing: " + ", ".join(missing))
    started = time.perf_counter()
    report = json.loads(profile_path.read_text(encoding="utf-8"))
    marker = folder / "feature_build_in_progress.json"
    _json_write(marker, {"feature_version": feature_metadata()["version"]})
    report["feature_rebuild_memory_before"] = process_memory_mb()
    report = _build_features(config, report)
    report["feature_rebuild_seconds"] = time.perf_counter() - started
    report["feature_rebuild_memory_after"] = process_memory_mb()
    _json_write(profile_path, report)
    marker.unlink()
    LOG.info("Feature rebuild complete in %.1f seconds", report["feature_rebuild_seconds"])
    return report


def _build_features(config: dict, report: dict) -> dict:
    folder = Path(config["paths"]["processed_dir"])
    applications = []
    for name in ["application_train", "application_test"]:
        raw = pd.read_parquet(folder / (name + ".parquet"))
        feats = application_features(raw)
        feats.insert(0, "SK_ID_CURR", raw.SK_ID_CURR.astype("int64"))
        feats["TARGET"] = raw.TARGET if "TARGET" in raw else np.nan
        feats["source_table"] = name
        applications.append(feats)
    main = pd.concat(applications, ignore_index=True)
    if main.SK_ID_CURR.duplicated().any():
        raise ValueError("Current-application identifiers are not unique; grouping strategy required")
    LOG.info("Aggregating historical features for %s original applications", len(main))
    history, cutoff_audit = aggregate_history(config, main.SK_ID_CURR.to_numpy())
    merged = main.merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    if len(merged) != len(main) or merged.SK_ID_CURR.duplicated().any():
        raise AssertionError("History aggregation multiplied application rows")
    merged.to_parquet(folder / "application_features.parquet", index=False)
    report["feature_rows"] = len(merged)
    report["feature_columns"] = len(merged.columns)
    report["history_cutoff_audit"] = cutoff_audit
    report["applicants_without_history"] = int((merged.history_present == 0).sum())
    labelled = merged[merged.TARGET.notna()]
    report["target_balance"] = {str(int(k)): int(v) for k, v in labelled.TARGET.value_counts().items()}
    corr = labelled.select_dtypes(include=np.number).drop(columns="SK_ID_CURR").corr()
    corr.to_parquet(folder / "correlations.parquet")
    report["target_correlations"] = {k: float(v) for k, v in corr.TARGET.drop("TARGET").dropna().sort_values(key=abs, ascending=False).items()}
    report["feature_version"] = feature_metadata()["version"]
    _json_write(Path(config["paths"]["artifact_dir"]) / "feature_metadata.json", feature_metadata())
    return report
