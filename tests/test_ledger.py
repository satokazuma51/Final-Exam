"""Substantive accounting, date, transaction, and report reconciliation checks."""
import copy
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from src.config import load_config
from src import db,ledger,analytics
from src.committee import evaluate_application
from src.schedules import annuity_schedule,add_months,to_cents


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.config=load_config()
        self.config['paths']['database']=str(Path(self.temp.name)/'operations.sqlite')
        db.init_db(self.config)
        self.client=db.create_client(self.config,'Учебный клиент',created_at='2024-01-01')
        self.sequence=0

    def tearDown(self):
        self.temp.cleanup()

    def application(self, product='cash',amount=12000,rate=12,term=12,draw=0,probability=.01):
        self.sequence+=1
        inputs=dict(monthly_income=100000,amount=amount,term_months=term,annual_rate=rate,product=product,
            age=35,employment_type='Working',employment_years=8,existing_monthly_debt=0,essential_expenses=1000,
            initial_draw=draw,application_date='2024-01-31',intended_disbursement_date='2024-01-31')
        app=db.submit_application(self.config,self.client,inputs,f'app-{self.sequence}',application_date='2024-01-31')
        result=evaluate_application(self.config,inputs,dict(probability=probability,threshold=.5,model_version='test-real-contract',feature_snapshot={},history_used=False),evaluation_date='2024-01-31')
        db.save_decision(self.config,app,result)
        return app,inputs

    def issue(self,**kwargs):
        app,_=self.application(**kwargs)
        return ledger.disburse(self.config,app,'2024-01-31',f'issue-{app}')

    def test_zero_rate_month_end_and_rounding(self):
        rows=annuity_schedule('1000',0,3,'2024-01-31')
        self.assertEqual([r['due_date'] for r in rows],['2024-02-29','2024-03-31','2024-04-30'])
        self.assertEqual(sum(to_cents(r['principal']) for r in rows),100000)
        self.assertEqual(to_cents(rows[-1]['payment']),33334)
        self.assertEqual(rows[-1]['remaining_principal'],0)

    def test_nonzero_schedule_principal_conservation(self):
        for rate in (1,18,60):
            rows=annuity_schedule('123456.78',rate,37,'2025-08-30')
            self.assertEqual(sum(to_cents(r['principal']) for r in rows),12345678)
            self.assertEqual(rows[-1]['remaining_principal'],0)
            self.assertTrue(all(r['principal']>=0 for r in rows))
        self.assertEqual(add_months('2024-02-29',1).isoformat(),'2024-03-31')

    def test_approval_does_not_disburse_and_rejection_cannot_issue(self):
        approved,_=self.application()
        report=analytics.snapshot(self.config,'2024-01-31',period_start='2024-01-01')
        self.assertEqual(report['kpis']['active_portfolio_balance'],0)
        self.assertEqual(report['kpis']['active_agreements'],0)
        rejected,_=self.application(probability=.9)
        with self.assertRaises(ValueError):
            ledger.disburse(self.config,rejected,'2024-01-31','reject-disbursement')
        self.assertEqual(db.get_application(self.config,approved)['status'],'Approved')

    def test_operation_keys_protect_all_money_events(self):
        app,inputs=self.application()
        self.assertEqual(db.submit_application(self.config,self.client,inputs,'app-1',application_date='2024-01-31'),app)
        agreement=ledger.disburse(self.config,app,'2024-01-31','once')
        self.assertEqual(ledger.disburse(self.config,app,'2024-01-31','once'),agreement)
        with self.assertRaises(ValueError):
            ledger.disburse(self.config,app,'2024-01-31','second-issue')
        one=ledger.record_payment(self.config,agreement,100,'2024-02-10','pay-once')
        two=ledger.record_payment(self.config,agreement,100,'2024-02-10','pay-once')
        self.assertEqual(one['transaction_id'],two['transaction_id'])
        with self.assertRaises(ValueError):
            ledger.record_payment(self.config,agreement,101,'2024-02-10','pay-once')
        with db.connect(self.config) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM transactions').fetchone()[0],2)

    def test_interest_principal_allocation_and_late_clearance(self):
        agreement=self.issue()
        before=ledger.agreement_snapshot(self.config,agreement,'2024-03-10')
        self.assertEqual(before['dpd'],10)
        tx=ledger.record_payment(self.config,agreement,60,'2024-03-10','partial')
        self.assertEqual(tx['interest'],60)
        self.assertEqual(tx['principal'],0)
        partial=ledger.agreement_snapshot(self.config,agreement,'2024-03-10')
        self.assertEqual(partial['outstanding_principal'],12000)
        self.assertEqual(partial['dpd'],10)
        due=partial['schedule'][0]['unpaid_amount']
        ledger.record_payment(self.config,agreement,due,'2024-03-11','catch-up')
        cleared=ledger.agreement_snapshot(self.config,agreement,'2024-03-11')
        self.assertEqual(cleared['dpd'],0)
        self.assertEqual(cleared['max_dpd'],11)
        self.assertLess(cleared['outstanding_principal'],12000)

    def test_early_full_repayment_and_historical_reconstruction(self):
        agreement=self.issue()
        quote=ledger.settlement_quote(self.config,agreement,'2024-02-10')
        self.assertEqual(quote['interest'],0)
        ledger.record_payment(self.config,agreement,quote['amount'],'2024-02-10','early-settle')
        self.assertEqual(ledger.agreement_snapshot(self.config,agreement,'2024-02-10')['status'],'Closed')
        earlier=ledger.agreement_snapshot(self.config,agreement,'2024-02-01')
        self.assertEqual(earlier['status'],'Active')
        self.assertEqual(earlier['outstanding_principal'],12000)
        report=analytics.snapshot(self.config,'2024-02-01',period_start='2024-01-01')
        self.assertEqual(report['kpis']['active_portfolio_balance'],12000)
        self.assertEqual(report['kpis']['payments_received'],0)

    def test_scheduled_maturity_does_not_close_unpaid_loan(self):
        agreement=self.issue(term=3)
        state=ledger.agreement_snapshot(self.config,agreement,'2025-01-01')
        self.assertEqual(state['status'],'Active')
        self.assertGreater(state['dpd'],0)

    def test_timely_annuity_payments_close_with_exact_principal(self):
        agreement=self.issue()
        original=annuity_schedule(12000,12,12,'2024-01-31')
        for row in original:
            quote=ledger.settlement_quote(self.config,agreement,row['due_date'])
            amount=min(row['payment'],quote['amount'])
            ledger.record_payment(self.config,agreement,amount,row['due_date'],f"regular-{row['installment_no']}")
        state=ledger.agreement_snapshot(self.config,agreement,'2025-01-31')
        self.assertEqual(state['status'],'Closed')
        self.assertEqual(state['outstanding_principal'],0)
        self.assertEqual(to_cents(state['principal_paid']),1200000)
        self.assertEqual(state['dpd'],0)

    def test_revolving_limit_restored_and_explicit_closure(self):
        agreement=self.issue(product='revolving',amount=10000,rate=0,draw=3000)
        ledger.drawdown(self.config,agreement,2000,'2024-02-01','extra')
        ledger.record_payment(self.config,agreement,1000,'2024-02-02','restore')
        state=ledger.agreement_snapshot(self.config,agreement,'2024-02-02')
        self.assertEqual(state['outstanding_principal'],4000)
        self.assertEqual(state['available_credit'],6000)
        with self.assertRaises(ValueError):
            ledger.drawdown(self.config,agreement,6001,'2024-02-02','over-limit')
        with self.assertRaises(ValueError):
            ledger.close_line(self.config,agreement,'2024-02-02','premature-close')
        ledger.record_payment(self.config,agreement,4000,'2024-02-03','zero-line')
        self.assertEqual(ledger.agreement_snapshot(self.config,agreement,'2024-02-03')['status'],'Active')
        ledger.close_line(self.config,agreement,'2024-02-04','close-line')
        self.assertEqual(ledger.agreement_snapshot(self.config,agreement,'2024-02-04')['status'],'Closed')
        self.assertEqual(ledger.agreement_snapshot(self.config,agreement,'2024-02-03')['status'],'Active')

    def test_revolving_daily_interest_and_minimum_arrears(self):
        agreement=self.issue(product='revolving',amount=10000,rate=36.5,draw=5000)
        self.assertEqual(ledger.agreement_snapshot(self.config,agreement,'2024-02-10')['accrued_interest'],50)
        quote=ledger.settlement_quote(self.config,agreement,'2024-03-02')
        self.assertEqual(quote['interest'],155)
        self.assertGreater(ledger.agreement_snapshot(self.config,agreement,'2024-03-02')['dpd'],0)
        ledger.record_payment(self.config,agreement,quote['amount'],'2024-03-02','revolver-settle')
        state=ledger.agreement_snapshot(self.config,agreement,'2024-03-02')
        self.assertEqual(state['accrued_interest'],0)
        self.assertEqual(state['outstanding_principal'],0)
        self.assertEqual(state['dpd'],0)
        self.assertEqual(state['status'],'Active')

    def test_sql_kpis_exports_reconcile_and_provenance_filters(self):
        agreement=self.issue()
        ledger.record_payment(self.config,agreement,500,'2024-03-01','report-pay')
        report=analytics.snapshot(self.config,'2024-03-10',period_start='2024-01-01',filters={'product':'cash','source_kind':'local'})
        self.assertEqual(report['kpis']['payments_received'],500)
        self.assertEqual(report['kpis']['payments_received'],report['kpis']['principal_received']+report['kpis']['interest_received'])
        reconciliation=report['sql_results']['principal_reconciliation']
        self.assertEqual(round(reconciliation['ledger_principal'].sum(),2),report['kpis']['active_portfolio_balance'])
        self.assertEqual(report['kpis']['clients'],1)
        self.assertGreaterEqual(len(report['sql_results']),8)
        self.assertIn('report_as_of',analytics.export_csv(report).decode('utf-8-sig'))
        empty=analytics.snapshot(self.config,'2024-03-10',filters={'source_kind':'synthetic'})
        self.assertEqual(empty['kpis']['active_portfolio_balance'],0)

    def test_missing_model_invalid_input_and_decision_versions(self):
        app,inputs=self.application()
        with self.assertRaises(ValueError):
            evaluate_application(self.config,inputs,None)
        bad=dict(inputs,monthly_income=0)
        with self.assertRaises(ValueError):
            db.submit_application(self.config,self.client,bad,'invalid')
        revised=evaluate_application(self.config,inputs,dict(probability=.9,threshold=.5,model_version='test-real-contract'),evaluation_date='2024-01-31')
        db.save_decision(self.config,app,revised)
        versions=db.list_decisions(self.config,app)
        self.assertEqual([r['version'] for r in versions],[2,1])
        self.assertEqual(db.get_application(self.config,app)['status'],'Rejected')
        self.assertEqual(analytics.snapshot(self.config,'2024-01-31')['kpis']['rejections'],1)

    def test_backdating_and_overpayment_rollback(self):
        agreement=self.issue()
        ledger.record_payment(self.config,agreement,100,'2024-02-10','latest')
        with self.assertRaises(ValueError):
            ledger.record_payment(self.config,agreement,100,'2024-02-09','past')
        with self.assertRaises(ValueError):
            ledger.record_payment(self.config,agreement,999999,'2024-02-11','too-much')
        self.assertEqual(len(ledger.agreement_snapshot(self.config,agreement,'2024-02-11')['payments']),1)

    def test_client_donut_exclusive_debt_series_and_decision_idempotency(self):
        first=self.issue()
        second=self.issue()
        report=analytics.snapshot(self.config,'2024-04-01',period_start='2024-03-01')
        self.assertEqual(sum(row['count'] for row in report['client_type_summary']),1)
        self.assertEqual(next(r['count'] for r in report['client_type_summary'] if r['client_type']=='Repeat'),1)
        self.assertEqual(report['activity_series'][0]['outstanding_principal'],24000)
        self.assertEqual(len(analytics.snapshot(self.config,'2024-04-01',filters={'client_type':'New'})['agreements']),0)
        app,inputs=self.application(probability=.9)
        result=db.list_decisions(self.config,app)[0]['result']
        previous=db.list_decisions(self.config,app)[0]['decision_id']
        self.assertEqual(db.save_decision(self.config,app,result),previous)
        self.assertEqual(len(db.list_decisions(self.config,app)),1)
        db.save_decision(self.config,app,result,operation_key='intentional-reevaluation')
        self.assertEqual(len(db.list_decisions(self.config,app)),2)

    def test_operation_provenance_differs_from_imported_client_profile(self):
        imported=db.create_client(self.config,'Демо-профиль',source_application_id=100001,
                                  source_kind='home_credit',is_synthetic=True,created_at='2024-01-01')
        self.client=imported
        _,inputs=self.application()
        app=db.submit_application(self.config,imported,inputs,'local-new-operation',application_date='2024-01-31',source_kind='local',is_synthetic=False)
        result=evaluate_application(self.config,inputs,dict(probability=.01,threshold=.5,model_version='test-real-contract'),evaluation_date='2024-01-31')
        db.save_decision(self.config,app,result)
        ledger.disburse(self.config,app,'2024-01-31','local-operation-issue')
        report=analytics.snapshot(self.config,'2024-01-31',filters={'source_kind':'local','product':'cash'})
        self.assertEqual(report['kpis']['clients'],1)
        self.assertEqual(report['sql_results']['application_rates'].iloc[0]['applications'],1)
        self.assertEqual(report['clients'][0]['source_kind'],'home_credit')

    def test_historical_decision_version_and_repeat_classification(self):
        first=self.issue()
        app,inputs=self.application()
        ledger.disburse(self.config,app,'2024-03-01','later-origination')
        february=analytics.snapshot(self.config,'2024-02-15',period_start='2024-01-01')
        april=analytics.snapshot(self.config,'2024-04-01',period_start='2024-01-01')
        self.assertEqual(february['client_type_summary'][0]['count'],1)
        self.assertEqual(april['client_type_summary'][1]['count'],1)
        pending_app,inputs=self.application()
        revised=evaluate_application(self.config,inputs,dict(probability=.9,threshold=.5,model_version='test-real-contract'),evaluation_date='2024-03-10')
        db.save_decision(self.config,pending_app,revised)
        february=analytics.snapshot(self.config,'2024-02-15',period_start='2024-01-01')
        april=analytics.snapshot(self.config,'2024-04-01',period_start='2024-01-01')
        old=next(a for a in february['applications'] if a['application_id']==pending_app)
        latest=next(a for a in april['applications'] if a['application_id']==pending_app)
        self.assertEqual(old['status'],'Approved')
        self.assertEqual(latest['status'],'Rejected')
        self.assertEqual(february['kpis']['disbursed_amount'],12000)
        self.assertEqual(april['kpis']['disbursed_amount'],24000)


if __name__=='__main__':
    unittest.main()
