# Учебная кредитная платформа · Home Credit

A working **local, Russian-language** lending demonstration with exactly two main sections: **Кредитная деятельность** and **Аналитика**. It includes actual trained models, persistent SQLite accounting, loan schedules, a reproducible synthetic portfolio, filtered reports, and examination materials.

This is an educational simulation. No real transfers, bank integrations or paid services are used. Original amounts are displayed in **у.е.** (generic currency units), not tenge. `TARGET=1` means payment difficulties under the source definition; it does not establish bankruptcy or complete non-repayment.

## Open the prepared project

From this project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost --server.port 8501
```

Open **http://localhost:8501**. The equivalent installed entry point is `streamlit run app.py --server.address localhost --server.port 8501`. The explicit `.venv` command needs no environment activation. Normal interactions load saved models; they never trigger training or regenerate the portfolio.

## Reproduce from source on Windows

Use **64-bit Python 3.12**. The project was installed and verified with Python 3.12.14 and the versions in `requirements.txt`; `requirements-lock.txt` records the complete verified environment.

```powershell
# If the Python launcher and Python 3.12 are installed:
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Copy-Item -LiteralPath config.example.json -Destination config.json
# Edit paths.data_dir in config.json if your CSVs are elsewhere.
$env:LENDING_CONFIG = (Resolve-Path -LiteralPath config.json).Path

.\.venv\Scripts\python.exe -m src.cli init-db
.\.venv\Scripts\python.exe -m src.cli prepare
.\.venv\Scripts\python.exe -m src.cli train
.\.venv\Scripts\python.exe -m src.cli demo
.\.venv\Scripts\python.exe -m src.cli verify
.\.venv\Scripts\python.exe -m src.cli reports
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost --server.port 8501
```

For an exact dependency replay, replace `-r requirements.txt` with `-r requirements-lock.txt`. Installing Python/packages requires internet access. Prepared data, models and normal app operation are local and require no internet.

If `python.exe` is only a Microsoft Store alias and `py` is unavailable, the existing `uv` installer can create a project-local runtime:

```powershell
uv python install 3.12 --install-dir .runtime --cache-dir .uv-cache
$taskRuntime = Get-ChildItem -LiteralPath .runtime -Directory -Filter 'cpython-3.12.*' |
    Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
uv venv .venv --python (Join-Path $taskRuntime 'python.exe') --cache-dir .uv-cache
uv pip install --python .venv\Scripts\python.exe --cache-dir .uv-cache pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Executable-link or registry warnings from `uv python install` do not prevent use of that explicit interpreter path. On this Windows installation, scikit-learn loads its packaged OpenMP runtime before LightGBM; the application preserves that import order.

## Configuration and commands

Configuration precedence: explicit CLI `--config`, then `LENDING_CONFIG`, then `config.json`, then `config.example.json`, then defaults. Paths are relative to the selected configuration file, so the project does not depend on a particular Desktop path.

```powershell
.\.venv\Scripts\python.exe -m src.cli --config config.json --data-dir 'D:\datasets\home-credit-default-risk' prepare
.\.venv\Scripts\python.exe -m src.cli prepare --features-only
.\.venv\Scripts\python.exe -m src.cli train --threads 4 --search-rows 50000
.\.venv\Scripts\python.exe -m src.cli submission
```

`--features-only` rebuilds features from a completed Parquet import. `prepare --force` rebuilds the analytical import. Do not run preparation or model training while presenting the app. Failed-build markers prevent an incomplete rebuild from being reused as a successful cache.

`train --max-rows 30000` is an optional bounded experiment; the actual sample size and split are recorded. The supplied final experiment uses all 307,511 labeled applications. Threads, CV folds, search budget, paths, product rates, eligibility, affordability limits, seed, and fixed demonstration reference date are configurable.

Rates and policy limits are **educational assumptions**, not statutory rules. Defaults: cash nominal annual rate 18%, revolving 24%, maximum debt-payment burden 50%, minimum remaining income 1,000 у.е., age eligibility 18–75. Fees and penalties are not implemented and are zero.

## Files and architecture

| Path | Role |
|---|---|
| `app.py`, `src/ui.py` | Entry point, Russian forms, decisions, loan operations and analytics |
| `src/config.py`, `config.example.json` | Portable paths and educational policy |
| `src/data_pipeline.py` | Chunked CSV inspection, dtype optimization, Parquet/SQLite imports, EDA |
| `src/features.py` | Shared training/form units, historical cutoff and application-level aggregates |
| `src/models.py` | Six real experiments, validation selection, calibration, persistence and scoring |
| `src/db.py`, `sql/schema.sql` | Operational schema, transactions, clients and decision versions |
| `src/committee.py` | Input validation and risk/affordability rules |
| `src/schedules.py`, `src/ledger.py` | Annuities, revolving credit, dated payments, principal, interest and DPD |
| `src/analytics.py`, `src/exports.py` | Filtered snapshots, SQL results, CSV/Excel/PDF exports |
| `src/demo_data.py` | Reproducible, separately labeled synthetic operations |
| `src/research_report.py` | Actual-result report, scientific figures, defense PowerPoint, submission |
| `src/cli.py` | Separate preparation/training/generation/verification commands |
| `sql/analytics.sql`, `sql/source_analytics.sql` | 10 operational and 3 historical-source queries |
| `sql/er_diagram.mmd` | Mermaid ER diagram |
| `tests/` | Accounting, history, models, genuine-model UI and generated-portfolio checks |
| `docs/` | Feature/KPI definitions and architectural explanation |

The original CSV directory is read-only by convention. `data/source.sqlite` and `data/processed/*.parquet` contain the analytical source layer; `data/lending.sqlite` contains operational accounting. Never add bureau debts to platform assets. Never join several raw histories together: each is aggregated before a validated one-to-one application join.

`client_id` belongs to this platform. `source_application_id` links a fictional demonstration profile to an original `SK_ID_CURR`; it is not a universal person identifier. Names never identify or merge clients automatically. Full names, IDs, gender, age, nationality, religion and external scores are excluded from model inputs. Age is checked separately by eligibility policy.

## Models, cutoff and actual results

Two serving variants use the same train/calibration/validation/holdout split:

- **Form model:** only the fields available in the manual form. Annual source income is divided by 12; payment and affordability are monthly.
- **History model:** those fields plus eligible application-level source-history aggregates. An unlinked client, or a client without eligible source history, uses the form model. Local accounting history is displayed separately; it is not silently substituted for semantically different source aggregates.

Both variants compare DummyClassifier, LogisticRegression and LightGBM. Learned imputation/encoding/scaling stay inside Pipeline/ColumnTransformer and the CV folds. The bounded search uses three-fold stratified CV and 50,000 training rows by default. Independent calibration and validation sets precede the untouched final holdout. Validation ROC-AUC selects the trained candidate; validation F1 selects the risk threshold. PR-AUC means `average_precision_score`, not trapezoidal PR area.

Actual past events must have relative date/month at or before 0. Bureau state rows updated after day 0 are excluded; past monthly observations can remain eligible. Future contractual maturity is a known plan, not a realized event. `DAYS_EMPLOYED=365243` becomes missing. Missing history is explicit and is not evidence of perfect repayment.

`artifacts/metrics.json` is the authoritative calculated result: all six models, ROC-AUC, average precision, precision/recall/F1, confusion matrices, Brier/log loss, calibration curves, timing, measured memory, model sizes, hyperparameters, seeds and limitations. `artifacts/split_assignments.parquet` records the split. Global permutation importance is distinct from the local TreeSHAP/logistic contributions shown for an individual prediction. Contributions describe model calculations, not proven causes of behavior.

Models and preprocessing are saved together with joblib. Versioned bundles and metrics are archived under `artifacts/versions/`. Decision records retain original inputs, transformed feature snapshot, thresholds, model/policy version and evaluation date. Re-evaluation creates a traceable decision version; it cannot rewrite an already issued agreement.

The original full-import process loaded an earlier process-memory helper, so its process working-set measurements are unavailable. Actual before/after chunk DataFrame measurements and actual final training working-set/peak measurements are recorded. No missing measurement is invented. Calibration can worsen a holdout metric; it is assessed and reported without choosing a replacement on the holdout.

## Accounting and demonstration portfolio

**Approved does not mean disbursed.** A separate action creates the agreement, schedule and initial transaction. Rejected/pending applications create no platform principal.

- Money is stored in integer hundredths; schedules use Decimal rounding, original-day/month-end anchoring and a final principal adjustment.
- Cash loans have an initial annuity schedule. Educational interest is charged at monthly due dates on actual outstanding principal. No daily interim cash interest is charged. Early principal repayment reduces future interest.
- Revolving credit separates limit, drawn principal, available amount and accrued interest. Interest uses simple ACT/365 daily accrual; monthly minimum principal is 5% or 100 у.е. (bounded by debt), plus accrued interest. Additional draws are transactions. Principal repayment restores available credit. Zero debt leaves a line open until an explicit closure action.
- Payments settle accrued interest first, then principal from the oldest obligations. Partial, late and early full repayments are supported. Arrears use unpaid obligations strictly before the reporting date. Current DPD can return to zero; historical maximum DPD remains.
- Dated replay reconstructs historical balances; future transactions do not appear in earlier snapshots. Backdating before a recorded event is rejected. Unique operation keys and SQLite transactions protect applications, disbursements, draws and payments from repeated clicks.

The fixed-seed `demo-v2` portfolio is generated once by a CLI command. It includes fictional names and synthetic dates, rates and payment behavior. Source product/income/amount/employment distributions provide the starting profiles; terms are sampled from approved historical applications. To make the new synthetic terms affordable, 70% of requested amounts are capped to a sampled 12–25% monthly-income payment budget; 30% keep source-size requests. This transformation applies only to synthetic operations and does not alter training data or predictions.

Unknown source employment tenure is explicitly assigned zero in the newly generated demo scenario, not asserted as historical truth. Imported interactive profiles instead require a tenure entry when unknown. The generator stores seed, version, source link, provenance and model version. It never invents TARGET labels or model metrics. Re-running the same generation key is idempotent. Retraining does not silently rewrite prior operational decisions.

For the supplied reference snapshot choose `2026-10-31` in Analytics. New locally entered transactions use their entered dates; select the matching reporting date/period. The default analytics date includes today. For a presentation, follow `reports/demo_script.txt`.

## Reports, verification and submission

```powershell
.\.venv\Scripts\python.exe -m src.cli verify
.\.venv\Scripts\python.exe -m src.cli reports
```

Verification automatically includes the genuine-model UI route when fitted artifacts exist. The generated-portfolio audit copies the source database read-only and reconciles all event/due dates, monthly snapshots and filter populations against independent SQL. Fresh installations skip optional artifact-dependent checks until data/models/demo exist. `reports/verification.xml` records the checks actually executed.

The UI exports filtered application/loan registers, schedules and portfolio summaries as CSV, plus Excel and PDF summaries. Metadata includes generation time, reporting date, period, currency and provenance. `reports/research_report.html` is a self-contained written report with embedded actual charts; open it locally or print to PDF. `reports/defense_presentation.pptx` is the defense deck. `reports/model_comparison.csv`, `reports/portfolio_summary.json`, and `reports/sql_*.csv` contain calculated results. Rebuild reports after a new experiment or verification run.

Optional `src.cli submission` writes 48,744 actual test probabilities in sample-submission order. Test labels are unavailable; test performance is never claimed.

The interface and Streamlit AppTest workflows were verified, together with the actual localhost page/health endpoints. Browser-pixel/screenshot QA was unavailable in this environment; this is distinct from the executed interaction tests. Logs are in `logs/platform.log`.

`.gitignore` excludes source CSVs, runtime/virtual environment, local databases, model binaries, processed data and large generated artifacts. Publish source and reproduction instructions; do not commit a database with user-entered information. Publication and Moodle upload remain the user's actions. Submit materials **2–3 days before the examination date supplied by the professor**; no exam date is assumed here.

## Sources

- [Home Credit Default Risk data](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- [Streamlit forms](https://docs.streamlit.io/develop/concepts/architecture/forms) and [caching](https://docs.streamlit.io/develop/concepts/architecture/caching)
- [scikit-learn leakage pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) and [pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [LightGBM documentation](https://lightgbm.readthedocs.io/)
