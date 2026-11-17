PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS products (
 product_code TEXT PRIMARY KEY, name TEXT NOT NULL, annual_rate REAL NOT NULL CHECK(annual_rate>=0), configuration_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clients (
 client_id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT NOT NULL CHECK(length(trim(full_name))>0),
 source_application_id INTEGER, source_kind TEXT NOT NULL DEFAULT 'local', is_synthetic INTEGER NOT NULL DEFAULT 0 CHECK(is_synthetic IN (0,1)),
 created_at TEXT NOT NULL, seed INTEGER, generator_version TEXT, metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_clients_source ON clients(source_application_id);
CREATE TABLE IF NOT EXISTS applications (
 application_id INTEGER PRIMARY KEY AUTOINCREMENT, client_id INTEGER NOT NULL REFERENCES clients(client_id),
 product TEXT NOT NULL REFERENCES products(product_code), amount_cents INTEGER NOT NULL CHECK(amount_cents>0),
 term_months INTEGER NOT NULL CHECK(term_months>0), annual_rate REAL NOT NULL CHECK(annual_rate>=0),
 initial_draw_cents INTEGER NOT NULL DEFAULT 0 CHECK(initial_draw_cents>=0), application_date TEXT NOT NULL,
 intended_disbursement_date TEXT, inputs_json TEXT NOT NULL, operation_key TEXT NOT NULL UNIQUE,
 source_kind TEXT NOT NULL, is_synthetic INTEGER NOT NULL CHECK(is_synthetic IN(0,1)), source_application_id INTEGER,
 created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_applications_client ON applications(client_id,application_date);
CREATE TABLE IF NOT EXISTS model_versions (
 model_version TEXT PRIMARY KEY, created_at TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS decisions (
 decision_id INTEGER PRIMARY KEY AUTOINCREMENT, application_id INTEGER NOT NULL REFERENCES applications(application_id),
 version INTEGER NOT NULL CHECK(version>0), decision TEXT NOT NULL CHECK(decision IN('Approved','Rejected')),
 risk_probability REAL NOT NULL CHECK(risk_probability BETWEEN 0 AND 1), threshold REAL NOT NULL CHECK(threshold BETWEEN 0 AND 1),
 model_version TEXT NOT NULL REFERENCES model_versions(model_version), policy_version TEXT NOT NULL,
 decision_date TEXT NOT NULL, result_json TEXT NOT NULL, inputs_json TEXT NOT NULL,
 feature_snapshot_json TEXT NOT NULL, thresholds_json TEXT NOT NULL, operation_key TEXT UNIQUE, UNIQUE(application_id,version)
);
CREATE INDEX IF NOT EXISTS idx_decisions_latest ON decisions(application_id,version DESC);
CREATE TABLE IF NOT EXISTS agreements (
 agreement_id INTEGER PRIMARY KEY AUTOINCREMENT, application_id INTEGER NOT NULL UNIQUE REFERENCES applications(application_id),
 decision_id INTEGER NOT NULL REFERENCES decisions(decision_id), client_id INTEGER NOT NULL REFERENCES clients(client_id),
 product TEXT NOT NULL REFERENCES products(product_code), opened_date TEXT NOT NULL, scheduled_end_date TEXT NOT NULL,
 annual_rate REAL NOT NULL CHECK(annual_rate>=0), credit_limit_cents INTEGER NOT NULL CHECK(credit_limit_cents>=0),
 initial_principal_cents INTEGER NOT NULL CHECK(initial_principal_cents>=0), term_months INTEGER NOT NULL CHECK(term_months>0),
 min_principal_pct REAL NOT NULL DEFAULT 0.05, min_payment_cents INTEGER NOT NULL DEFAULT 10000,
 source_kind TEXT NOT NULL, is_synthetic INTEGER NOT NULL CHECK(is_synthetic IN(0,1)), source_application_id INTEGER,
 created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agreements_client ON agreements(client_id,opened_date);
CREATE TABLE IF NOT EXISTS scheduled_installments (
 schedule_id INTEGER PRIMARY KEY AUTOINCREMENT, agreement_id INTEGER NOT NULL REFERENCES agreements(agreement_id),
 installment_no INTEGER NOT NULL CHECK(installment_no>0), due_date TEXT NOT NULL,
 payment_cents INTEGER NOT NULL CHECK(payment_cents>=0), principal_cents INTEGER NOT NULL CHECK(principal_cents>=0),
 interest_cents INTEGER NOT NULL CHECK(interest_cents>=0), remaining_principal_cents INTEGER NOT NULL CHECK(remaining_principal_cents>=0),
 UNIQUE(agreement_id,installment_no)
);
CREATE INDEX IF NOT EXISTS idx_schedule_due ON scheduled_installments(agreement_id,due_date);
CREATE TABLE IF NOT EXISTS transactions (
 transaction_id INTEGER PRIMARY KEY AUTOINCREMENT, agreement_id INTEGER NOT NULL REFERENCES agreements(agreement_id),
 transaction_date TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN('disbursement','drawdown','payment','line_closure')),
 amount_cents INTEGER NOT NULL CHECK(amount_cents>=0), principal_cents INTEGER NOT NULL DEFAULT 0 CHECK(principal_cents>=0),
 interest_cents INTEGER NOT NULL DEFAULT 0 CHECK(interest_cents>=0), operation_key TEXT NOT NULL UNIQUE,
 request_json TEXT NOT NULL, created_at TEXT NOT NULL,
 CHECK(kind!='payment' OR amount_cents=principal_cents+interest_cents)
);
CREATE INDEX IF NOT EXISTS idx_transactions_asof ON transactions(agreement_id,transaction_date,transaction_id);
CREATE TABLE IF NOT EXISTS demo_metadata (
 demo_key TEXT PRIMARY KEY, seed INTEGER NOT NULL, generator_version TEXT NOT NULL, reference_date TEXT NOT NULL,
 created_at TEXT NOT NULL, metadata_json TEXT NOT NULL
);
