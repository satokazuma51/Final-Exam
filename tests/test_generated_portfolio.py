"""Optional integration audit of the actual reproducibly generated portfolio.

The source operational database is opened read-only, copied using SQLite's
backup API, and never modified. These checks skip when no generated portfolio
exists, so fresh-install unit tests remain reproducible.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd

from src.config import load_config
from src import analytics,ledger
from src.schedules import as_date,add_months,to_cents


class GeneratedPortfolioTests(unittest.TestCase):
    audit_summary={}

    @classmethod
    def setUpClass(cls):
        cls.config=load_config()
        cls.original_path=Path(cls.config['paths']['database'])
        if not cls.original_path.exists():
            raise unittest.SkipTest('No generated operational database yet.')
        original=sqlite3.connect('file:'+cls.original_path.as_posix()+'?mode=ro',uri=True)
        original.row_factory=sqlite3.Row
        tables={row[0] for row in original.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'demo_metadata' not in tables:
            original.close()
            raise unittest.SkipTest('No generated portfolio metadata yet.')
        metadata=original.execute('SELECT * FROM demo_metadata ORDER BY created_at DESC LIMIT 1').fetchone()
        if metadata is None:
            original.close()
            raise unittest.SkipTest('No completed generated portfolio yet.')
        cls.metadata=dict(metadata)
        cls.generation=json.loads(metadata['metadata_json'])
        cls.reference=metadata['reference_date']
        cls.original_hash=hashlib.sha256(cls.original_path.read_bytes()).hexdigest()
        cls.temp=tempfile.TemporaryDirectory()
        copy_path=Path(cls.temp.name)/'audit.sqlite'
        target=sqlite3.connect(copy_path)
        original.backup(target)
        target.close()
        original.close()
        cls.config['paths']['database']=str(copy_path)
        cls.read_connection=sqlite3.connect('file:'+copy_path.as_posix()+'?mode=ro',uri=True)
        cls.read_connection.row_factory=sqlite3.Row
        cls.all_transactions=[dict(r) for r in cls.read_connection.execute('SELECT * FROM transactions ORDER BY transaction_date,transaction_id')]
        cls.all_agreements=[dict(r) for r in cls.read_connection.execute('SELECT * FROM agreements')]
        cls.report=analytics.snapshot(cls.config,cls.reference,period_start='1900-01-01')
        cls.audit_summary=dict(reference_date=cls.reference,demo_key=metadata['demo_key'],
                               database_read_only=True,kpis=cls.report['kpis'],
                               generated_metadata=cls.generation,checks=[])

    @classmethod
    def tearDownClass(cls):
        cls.read_connection.close()
        cls.temp.cleanup()
        after=hashlib.sha256(cls.original_path.read_bytes()).hexdigest()
        if after!=cls.original_hash:
            # A concurrently entered real operation would also change this
            # fingerprint; the audit itself has no writable source connection.
            raise AssertionError('Source database changed during read-only audit.')

    def test_metadata_provenance_foreign_keys_and_actual_scenario_coverage(self):
        self.assertEqual(self.read_connection.execute('PRAGMA integrity_check').fetchone()[0],'ok')
        self.assertEqual(self.read_connection.execute('PRAGMA foreign_key_check').fetchall(),[])
        for table in ('clients','applications','agreements'):
            rows=self.read_connection.execute(f'SELECT * FROM {table}').fetchall()
            synthetic=[r for r in rows if r['source_kind']=='synthetic']
            self.assertTrue(synthetic)
            self.assertTrue(all(r['is_synthetic']==1 for r in synthetic))
        for row in self.read_connection.execute("SELECT * FROM clients WHERE source_kind='synthetic'"):
            self.assertEqual(row['seed'],self.metadata['seed'])
            self.assertEqual(row['generator_version'],self.metadata['generator_version'])
        for field,table in [('clients','clients'),('applications','applications'),('agreements','agreements')]:
            count=self.read_connection.execute(f"SELECT COUNT(*) FROM {table} WHERE source_kind='synthetic'").fetchone()[0]
            self.assertEqual(count,self.generation[field])
        self.assertEqual(self.read_connection.execute('''SELECT COUNT(*) FROM agreements g JOIN decisions d USING(decision_id) WHERE d.decision!='Approved' ''').fetchone()[0],0)
        agreements=self.report['agreements']
        scenarios={
            'closed': [g['agreement_id'] for g in agreements if g['status']=='Closed'],
            'partial': [g['agreement_id'] for g in agreements if any(0<r['actual_paid']<r['payment'] for r in g['schedule'])],
            'arrears': [g['agreement_id'] for g in agreements if g['dpd']>0],
            'late_recovered': [g['agreement_id'] for g in agreements if g['max_dpd']>0 and g['dpd']==0],
            'revolving_drawn_available': [g['agreement_id'] for g in agreements if g['product']=='revolving' and g['outstanding_principal']>0 and g['available_credit']>0],
        }
        for name,ids in scenarios.items():
            self.assertTrue(ids,name)
        self.assertTrue(self.read_connection.execute('SELECT 1 FROM clients WHERE source_application_id IS NULL').fetchone())
        self.assertTrue(all(r['count']>0 for r in self.report['client_type_summary']))
        self.assertEqual(sum(r['count'] for r in self.report['client_type_summary']),self.report['kpis']['clients'])
        self.audit_summary['scenario_examples']={name:ids[:5] for name,ids in scenarios.items()}
        self.audit_summary['checks'].append('integrity, foreign keys, provenance, actual scenario coverage')

    def test_all_agreement_event_and_due_dates_against_independent_cents_sql(self):
        audited_dates=0
        for agreement in self.all_agreements:
            aid=agreement['agreement_id']
            record,schedule,transactions=ledger._rows(self.read_connection,aid)
            dates={t['transaction_date'] for t in transactions if t['transaction_date']<=self.reference}
            dates|={r['due_date'] for r in schedule if r['due_date']<=self.reference}
            dates.add(self.reference)
            for day in sorted(dates):
                state=ledger._replay(record,schedule,transactions,day)
                expected=self.read_connection.execute('''SELECT COALESCE(SUM(CASE
                    WHEN kind IN('disbursement','drawdown') THEN amount_cents
                    WHEN kind='payment' THEN -principal_cents ELSE 0 END),0)
                    FROM transactions WHERE agreement_id=? AND transaction_date<=?''',(aid,day)).fetchone()[0]
                self.assertEqual(to_cents(state['outstanding_principal']),expected,(aid,day))
                self.assertGreaterEqual(state['max_dpd'],state['dpd'])
                self.assertGreaterEqual(state['outstanding_principal'],0)
                paid=[t for t in transactions if t['kind']=='payment' and t['transaction_date']<=day]
                self.assertEqual(to_cents(state['payment_total']),sum(t['amount_cents'] for t in paid))
                self.assertEqual(to_cents(state['principal_paid']),sum(t['principal_cents'] for t in paid))
                self.assertEqual(to_cents(state['interest_paid']),sum(t['interest_cents'] for t in paid))
                saved={t['transaction_id']:t for t in paid}
                for payment in state['payments']:
                    raw=saved[payment['transaction_id']]
                    self.assertEqual(to_cents(payment['principal']),raw['principal_cents'])
                    self.assertEqual(to_cents(payment['interest']),raw['interest_cents'])
                    self.assertLessEqual(payment['payment_date'],day)
                if state['product']=='revolving' and state['status']=='Active':
                    self.assertEqual(to_cents(state['available_credit'])+to_cents(state['outstanding_principal']),state['credit_limit_cents'])
                if state['status']=='Closed':
                    self.assertEqual(state['settlement_amount'],0)
                audited_dates+=1
        self.audit_summary['audited_agreement_dates']=audited_dates
        self.audit_summary['checks'].append('every event/due date: independent SQL principal, persisted payment allocations, limits, historical exclusion')

    def test_month_end_historical_portfolio_and_filtered_exports_reconcile(self):
        earliest=min(g['opened_date'] for g in self.all_agreements)
        cursor=as_date(earliest).replace(day=1)
        dates=[]
        while cursor<=as_date(self.reference):
            end=as_date(add_months(cursor,1)).toordinal()-1
            dates.append(min(as_date(self.reference),as_date(cursor).fromordinal(end)).isoformat())
            cursor=add_months(cursor,1)
        for day in dates:
            report=analytics.snapshot(self.config,day,period_start=earliest)
            expected=self.read_connection.execute('''SELECT COALESCE(SUM(CASE
                WHEN kind IN('disbursement','drawdown') THEN amount_cents
                WHEN kind='payment' THEN -principal_cents ELSE 0 END),0)
                FROM transactions WHERE transaction_date<=?''',(day,)).fetchone()[0]
            self.assertEqual(to_cents(report['kpis']['active_portfolio_balance']),expected,day)
            self.assertEqual(round(report['sql_results']['principal_reconciliation']['ledger_principal'].sum(),2),round(expected/100,2))
            self.assertTrue(all(g['opened_date']<=day for g in report['agreements']))
            self.assertTrue(all(t['transaction_date']<=day for t in report['transactions']))
        filters=[{'product':'cash'},{'product':'revolving'},{'status':'Closed'},{'status':'Active'},
                 {'client_type':'Repeat'},{'client_type':'New'},{'source_kind':'synthetic'},
                 {'product':'revolving','status':'Active','source_kind':'synthetic'}]
        for selection in filters:
            report=analytics.snapshot(self.config,self.reference,selection,period_start=earliest)
            frame=pd.read_csv(io.BytesIO(analytics.export_csv(report,'agreements')))
            total=round(frame['outstanding_principal'].sum(),2) if len(frame) else 0
            self.assertEqual(total,report['kpis']['active_portfolio_balance'],selection)
            self.assertEqual(round(report['sql_results']['principal_reconciliation']['ledger_principal'].sum(),2),total,selection)
            self.assertEqual(round(sum(p['amount'] for p in report['payments']),2),report['kpis']['payments_received'])
            self.assertEqual(to_cents(report['kpis']['payments_received']),to_cents(report['kpis']['principal_received'])+to_cents(report['kpis']['interest_received']))
            self.assertEqual(sum(r['count'] for r in report['client_type_summary']),report['kpis']['clients'])
        self.audit_summary['audited_month_end_dates']=dates
        self.audit_summary['audited_filter_populations']=filters
        self.audit_summary['checks'].append('month-end historical balances, eight filter populations, CSV/SQL/KPI totals')

    def test_saved_predictions_are_reproducible_real_models_without_forbidden_features(self):
        artifact=Path(self.config['paths']['artifact_dir'])/'model_bundle.joblib'
        # The trained artifact name is owned by src.models, not inferred here.
        from src.models import load_model_bundle,score_application
        try:
            bundle=load_model_bundle(self.config)
        except FileNotFoundError as error:
            self.skipTest(str(error))
        records=self.read_connection.execute('''SELECT a.*,d.model_version,d.risk_probability,d.feature_snapshot_json
            FROM applications a JOIN decisions d USING(application_id)
            WHERE a.source_kind='synthetic' ORDER BY application_id''').fetchall()
        forbidden={'TARGET','SK_ID_CURR','SK_ID_PREV','SK_ID_BUREAU','client_id','full_name','name','sex','gender','CODE_GENDER','nationality','religion','age'}
        for record in records:
            self.assertFalse(forbidden.intersection(json.loads(record['feature_snapshot_json'])))
        indexes=sorted(set([0,1,2,len(records)//2,len(records)-1]))
        for index in indexes:
            record=records[index]
            score=score_application(self.config,json.loads(record['inputs_json']),source_application_id=record['source_application_id'],bundle=bundle)
            self.assertEqual(score['model_version'],record['model_version'])
            self.assertAlmostEqual(score['probability'],record['risk_probability'],places=12)
        self.audit_summary['reproduced_model_predictions']=len(indexes)
        self.audit_summary['checks'].append('saved actual model predictions reproduce; prohibited identifiers/demographics absent')


if __name__=='__main__':
    unittest.main()
