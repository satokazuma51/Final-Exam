"""Transactional operational SQLite storage. Original source data lives elsewhere."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
import json
import hashlib
from pathlib import Path
import sqlite3

from .schedules import as_date, to_cents


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)


def database_path(config):
    return Path(config['paths']['database'])


@contextmanager
def connect(config):
    path = database_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys=ON')
    connection.execute('PRAGMA busy_timeout=30000')
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(config):
    schema = Path(__file__).resolve().parents[1] / 'sql' / 'schema.sql'
    with connect(config) as connection:
        connection.executescript(schema.read_text(encoding='utf-8'))
        if 'operation_key' not in [row['name'] for row in connection.execute('PRAGMA table_info(decisions)')]:
            connection.execute('ALTER TABLE decisions ADD COLUMN operation_key TEXT')
        connection.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_decision_operation ON decisions(operation_key)')
        for code, product in config.get('products', {}).items():
            if code not in ('cash', 'revolving'):
                continue
            connection.execute('INSERT INTO products VALUES(?,?,?,?) ON CONFLICT(product_code) DO UPDATE SET name=excluded.name,annual_rate=excluded.annual_rate,configuration_json=excluded.configuration_json',
                               (code, product.get('name', code), float(product.get('annual_rate', 0)), json_text(product)))
        for code in ('cash', 'revolving'):
            connection.execute('INSERT OR IGNORE INTO products VALUES(?,?,?,?)', (code, code, 0, '{}'))


def create_client(config, full_name, source_application_id=None, source_kind='local', is_synthetic=False, **metadata):
    if not str(full_name).strip():
        raise ValueError('Укажите имя клиента.')
    with connect(config) as connection:
        # Import identity is its original application, never the generated name.
        if source_application_id is not None and source_kind != 'local':
            existing = connection.execute('SELECT client_id FROM clients WHERE source_application_id=? AND source_kind=? AND is_synthetic=?',
                                          (int(source_application_id), source_kind, int(is_synthetic))).fetchone()
            if existing:
                return existing['client_id']
        cursor = connection.execute('INSERT INTO clients(full_name,source_application_id,source_kind,is_synthetic,created_at,seed,generator_version,metadata_json) VALUES(?,?,?,?,?,?,?,?)',
                                    (str(full_name).strip(), source_application_id, source_kind, int(is_synthetic),
                                     str(metadata.pop('created_at', now_iso())), metadata.get('seed'), metadata.get('generator_version'), json_text(metadata)))
        return cursor.lastrowid


def get_client(config, client_id):
    with connect(config) as connection:
        row = connection.execute('SELECT * FROM clients WHERE client_id=?', (int(client_id),)).fetchone()
    if not row:
        raise ValueError('Клиент не найден.')
    return dict(row)


def list_clients(config, search=''):
    search = str(search).strip()
    with connect(config) as connection:
        rows = connection.execute('SELECT * FROM clients WHERE ?="" OR CAST(client_id AS TEXT)=? OR full_name LIKE ? OR CAST(source_application_id AS TEXT)=? ORDER BY client_id DESC',
                                  (search, search, '%' + search + '%', search)).fetchall()
    return [dict(row) for row in rows]


search_clients = list_clients


def submit_application(config, client_id, inputs, operation_key, application_date=None, **provenance):
    if not operation_key:
        raise ValueError('Требуется уникальный ключ операции.')
    from .committee import validate_inputs
    validate_inputs(config, inputs)
    application_date = as_date(application_date or inputs.get('application_date') or date.today()).isoformat()
    payload = json_text(inputs)
    with connect(config) as connection:
        connection.execute('BEGIN IMMEDIATE')
        existing = connection.execute('SELECT * FROM applications WHERE operation_key=?', (operation_key,)).fetchone()
        if existing:
            if existing['client_id'] != int(client_id) or existing['inputs_json'] != payload or existing['application_date'] != application_date:
                raise ValueError('Ключ операции уже использован для другой заявки.')
            return existing['application_id']
        client = connection.execute('SELECT * FROM clients WHERE client_id=?', (int(client_id),)).fetchone()
        if not client:
            raise ValueError('Клиент не найден.')
        cursor = connection.execute('''INSERT INTO applications(client_id,product,amount_cents,term_months,annual_rate,initial_draw_cents,application_date,intended_disbursement_date,inputs_json,operation_key,source_kind,is_synthetic,source_application_id,created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
           (int(client_id), inputs['product'], to_cents(inputs['amount']), int(inputs['term_months']), float(inputs['annual_rate']),
            to_cents(inputs.get('initial_draw', 0)), application_date, str(inputs.get('intended_disbursement_date') or application_date), payload, operation_key,
            provenance.get('source_kind', client['source_kind']), int(provenance.get('is_synthetic', client['is_synthetic'])),
            provenance.get('source_application_id', client['source_application_id']), now_iso()))
        return cursor.lastrowid


def _application(row):
    result = dict(row)
    result['inputs'] = json.loads(result['inputs_json'])
    result['amount'] = result['amount_cents'] / 100
    result['status'] = result.get('decision') or 'Pending'
    if result.get('result_json'):
        result['decision_details'] = json.loads(result['result_json'])
    return result


APPLICATION_QUERY = '''SELECT a.*,c.full_name,d.decision,d.risk_probability,d.model_version,d.policy_version,d.version AS decision_version,d.decision_date,d.result_json,
 g.agreement_id FROM applications a JOIN clients c USING(client_id)
 LEFT JOIN decisions d ON d.decision_id=(SELECT d2.decision_id FROM decisions d2 WHERE d2.application_id=a.application_id ORDER BY version DESC LIMIT 1)
 LEFT JOIN agreements g USING(application_id)'''


def get_application(config, application_id):
    with connect(config) as connection:
        row = connection.execute(APPLICATION_QUERY + ' WHERE a.application_id=?', (int(application_id),)).fetchone()
    if not row:
        raise ValueError('Заявка не найдена.')
    return _application(row)


def list_applications(config, client_id=None):
    with connect(config) as connection:
        rows = connection.execute(APPLICATION_QUERY + (' WHERE a.client_id=?' if client_id is not None else '') + ' ORDER BY a.application_id DESC',
                                  (int(client_id),) if client_id is not None else ()).fetchall()
    return [_application(row) for row in rows]


def save_decision(config, application_id, decision, operation_key=None):
    if decision.get('decision') not in ('Approved', 'Rejected') or not decision.get('model_version'):
        raise ValueError('Для решения необходимы обученная модель и полный результат комитета.')
    with connect(config) as connection:
        connection.execute('BEGIN IMMEDIATE')
        payload = json_text(decision)
        operation_key = operation_key or f'decision:{int(application_id)}:' + hashlib.sha256(payload.encode('utf-8')).hexdigest()
        duplicate = connection.execute('SELECT * FROM decisions WHERE operation_key=?', (operation_key,)).fetchone()
        if duplicate:
            if duplicate['application_id'] != int(application_id) or duplicate['result_json'] != payload:
                raise ValueError('Ключ операции уже использован для другого решения.')
            return duplicate['decision_id']
        app = connection.execute('SELECT * FROM applications WHERE application_id=?', (int(application_id),)).fetchone()
        if not app:
            raise ValueError('Заявка не найдена.')
        if connection.execute('SELECT 1 FROM agreements WHERE application_id=?', (int(application_id),)).fetchone():
            raise ValueError('Выданную заявку нельзя переоценить. Создайте новую заявку.')
        version = connection.execute('SELECT COALESCE(MAX(version),0)+1 FROM decisions WHERE application_id=?', (int(application_id),)).fetchone()[0]
        model_version = str(decision['model_version'])
        connection.execute('INSERT OR IGNORE INTO model_versions VALUES(?,?,?)', (model_version, now_iso(), '{}'))
        decision_date = str(decision.get('evaluation_date') or decision.get('decision_date') or app['application_date'])
        cursor = connection.execute('''INSERT INTO decisions(application_id,version,decision,risk_probability,threshold,model_version,policy_version,decision_date,result_json,inputs_json,feature_snapshot_json,thresholds_json,operation_key)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''', (int(application_id), version, decision['decision'],
           float(decision.get('risk_probability', decision.get('probability'))), float(decision['threshold']), model_version,
           str(decision.get('policy_version', 'educational-v1')), decision_date, payload, app['inputs_json'],
           json_text(decision.get('feature_snapshot', {})), json_text(decision.get('policy', config.get('policy', {}))),operation_key))
        return cursor.lastrowid


def list_decisions(config, application_id):
    with connect(config) as connection:
        rows = connection.execute('SELECT * FROM decisions WHERE application_id=? ORDER BY version DESC', (int(application_id),)).fetchall()
    return [dict(row, result=json.loads(row['result_json'])) for row in rows]


def client_history(config, client_id, as_of=None):
    from .analytics import snapshot
    from .ledger import local_monthly_debt
    as_of = as_of or date.today()
    report = snapshot(config, as_of, {'client_id': int(client_id)})
    return dict(client=get_client(config, client_id), applications=report['applications'], agreements=report['agreements'],
                payments=report['payments'], monthly_debt=local_monthly_debt(config, client_id, as_of),
                as_of=as_date(as_of).isoformat(), source_note='Операции платформы. Исходный профиль Home Credit отображается отдельно.')
