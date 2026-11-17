"""Filtered dated snapshots, real SQL results, and report-ready record populations."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
import io
import json
from pathlib import Path
import sqlite3

import pandas as pd

from .db import connect, get_client
from .ledger import _rows, _replay
from .schedules import as_date,add_months


def _chosen(value):
    return value not in (None, '', '*', 'all', 'All', 'Все')


def _sql_queries():
    text = (Path(__file__).resolve().parents[1] / 'sql' / 'analytics.sql').read_text(encoding='utf-8')
    result = {}
    for chunk in text.split('-- name: ')[1:]:
        name, query = chunk.split('\n', 1)
        result[name.strip()] = query.strip().split(';', 1)[0]
    return result


def _sql_results(agreements, applications, clients, transactions, parameters):
    columns = {
        'report_agreements': ['agreement_id','application_id','client_id','product','opened_date','status','outstanding_principal','credit_limit','overdue_amount','dpd','client_type','total_disbursed'],
        'report_applications': ['application_id','client_id','product','application_date','status','risk_probability'],
        'report_clients': ['client_id','full_name'],
        'report_transactions': ['transaction_id','agreement_id','transaction_date','kind','amount','principal','interest'],
    }
    with sqlite3.connect(':memory:') as connection:
        for name, rows in zip(columns, (agreements,applications,clients,transactions)):
            pd.DataFrame(rows).reindex(columns=columns[name]).to_sql(name,connection,index=False,if_exists='replace')
        return {name:pd.read_sql_query(query,connection,params=parameters) for name,query in _sql_queries().items()}


def snapshot(config, as_of=None, filters=None, period_start=None, period_end=None):
    day=as_date(as_of or date.today()).isoformat()
    filters=dict(filters or {})
    start=as_date(period_start or filters.get('period_start') or day[:7]+'-01').isoformat()
    end=as_date(period_end or filters.get('period_end') or day).isoformat()
    if start>end:
        raise ValueError('Начало периода позже окончания.')
    if end>day:
        raise ValueError('Конец периода не может быть позже отчетной даты.')
    with connect(config) as connection:
        all_clients=[dict(r) for r in connection.execute('SELECT * FROM clients')]
        all_apps=[dict(r) for r in connection.execute('SELECT a.*,c.full_name FROM applications a JOIN clients c USING(client_id) WHERE application_date<=?',(day,))]
        all_decisions=[dict(r) for r in connection.execute('SELECT * FROM decisions WHERE substr(decision_date,1,10)<=? ORDER BY application_id,version',(day,))]
        agreement_ids=[r[0] for r in connection.execute('SELECT agreement_id FROM agreements WHERE opened_date<=?',(day,))]
        all_agreements=[]
        for agreement_id in agreement_ids:
            agreement,schedule,transactions=_rows(connection,agreement_id)
            all_agreements.append(_replay(agreement,schedule,transactions,day))
    decisions={r['application_id']:r for r in all_decisions}
    by_client=defaultdict(list)
    for row in all_agreements:
        by_client[row['client_id']].append(row)
    funded_counts={client_id:sum(any(t['kind'] in ('disbursement','drawdown') and t['amount']>0 for t in g['transactions'])
                                for g in rows) for client_id,rows in by_client.items()}
    report_client_type={c['client_id']:('Repeat' if funded_counts.get(c['client_id'],0)>=2 else 'New') for c in all_clients}

    def classification(client_id, event_date, current_agreement_id=None):
        earlier=[g for g in by_client[client_id] if g['agreement_id']!=current_agreement_id and
                 any(t['kind'] in ('disbursement','drawdown') and t['amount']>0 and t['transaction_date']<event_date for t in g['transactions'])]
        return 'Repeat' if earlier else 'New'

    def provenance_ok(row):
        for name in ('product','source_kind','client_id'):
            if _chosen(filters.get(name)) and str(row.get(name))!=str(filters[name]):
                return False
        if _chosen(filters.get('is_synthetic')) and bool(row.get('is_synthetic'))!=bool(filters['is_synthetic']):
            return False
        return True

    for app in all_apps:
        decision=decisions.get(app['application_id'])
        app.update(status=decision['decision'] if decision else 'Pending',
                   decision=decision['decision'] if decision else None,
                   risk_probability=decision['risk_probability'] if decision else None,
                   model_version=decision['model_version'] if decision else None,
                   decision_version=decision['version'] if decision else None,
                   decision_details=json.loads(decision['result_json']) if decision else None,
                   inputs=json.loads(app['inputs_json']),amount=app['amount_cents']/100,
                   origination_client_type=classification(app['client_id'],app['application_date']),
                   client_type=report_client_type[app['client_id']])
    app_by_id={a['application_id']:a for a in all_apps}
    for agreement in all_agreements:
        agreement['origination_client_type']=classification(agreement['client_id'],agreement['opened_date'],agreement['agreement_id'])
        agreement['client_type']=report_client_type[agreement['client_id']]
        agreement['risk_probability']=app_by_id.get(agreement['application_id'],{}).get('risk_probability')
    status_filter=filters.get('status')
    agreements=[g for g in all_agreements if provenance_ok(g) and
                (not _chosen(status_filter) or status_filter not in ('Active','Closed') or g['status']==status_filter) and
                (not _chosen(status_filter) or status_filter not in ('Approved','Rejected','Pending') or app_by_id[g['application_id']]['status']==status_filter) and
                (not _chosen(filters.get('client_type')) or g['client_type']==filters['client_type'])]
    selected_agreement_apps={g['application_id'] for g in agreements}
    applications=[a for a in all_apps if provenance_ok(a) and
                  (not _chosen(status_filter) or status_filter not in ('Approved','Rejected','Pending') or a['status']==status_filter) and
                  (not _chosen(status_filter) or status_filter not in ('Active','Closed') or a['application_id'] in selected_agreement_apps) and
                  (not _chosen(filters.get('client_type')) or a['client_type']==filters['client_type'])]
    # Loan SQL joins must retain their applications even when the application
    # itself falls outside the transaction period. KPI applications use period.
    selected_client_ids={a['client_id'] for a in applications}|{g['client_id'] for g in agreements}
    clients=[]
    for client in all_clients:
        client_app_dates=[a['application_date'] for a in all_apps if a['client_id']==client['client_id']]
        existed=min([client['created_at'][:10]]+client_app_dates)<=day
        if not existed or (_chosen(filters.get('client_id')) and str(client['client_id'])!=str(filters['client_id'])):
            continue
        if _chosen(filters.get('client_type')) and report_client_type[client['client_id']]!=filters['client_type']:
            continue
        has_operation_filter=any(_chosen(filters.get(name)) for name in ('product','status','source_kind','is_synthetic'))
        has_operations=any(a['client_id']==client['client_id'] for a in all_apps) or bool(by_client[client['client_id']])
        profile_matches=(not _chosen(filters.get('source_kind')) or client['source_kind']==filters['source_kind']) and (not _chosen(filters.get('is_synthetic')) or bool(client['is_synthetic'])==bool(filters['is_synthetic']))
        orphan_profile_allowed=not has_operations and not _chosen(filters.get('product')) and not _chosen(filters.get('status')) and profile_matches
        if not has_operation_filter or client['client_id'] in selected_client_ids or orphan_profile_allowed:
            clients.append(dict(client,client_type=report_client_type[client['client_id']]))
    transactions=[dict(t,client_id=g['client_id'],full_name=g['full_name'],product=g['product'],source_kind=g['source_kind'],
                       is_synthetic=g['is_synthetic'],source_application_id=g['source_application_id'],client_type=g['client_type'])
                  for g in agreements for t in g['transactions']]
    activity=[t for t in transactions if start<=t['transaction_date']<=end]
    payments=[t for t in activity if t['kind']=='payment']
    disbursements=[t for t in activity if t['kind'] in ('disbursement','drawdown') and t['amount']>0]
    period_apps=[a for a in applications if start<=a['application_date']<=end]
    approvals=sum(a['status']=='Approved' for a in period_apps)
    rejections=sum(a['status']=='Rejected' for a in period_apps)
    risks=[a['risk_probability'] for a in period_apps if a['risk_probability'] is not None]
    amount=sum(t['amount'] for t in disbursements)
    disbursed_ids={t['agreement_id'] for t in disbursements}
    kpis=dict(clients=len(clients),applications=len(period_apps),decisions=approvals+rejections,
              approvals=approvals,rejections=rejections,pending=sum(a['status']=='Pending' for a in period_apps),
              approval_rate=approvals/(approvals+rejections) if approvals+rejections else None,
              disbursed_count=len(disbursed_ids),disbursed_amount=round(amount,2),
              active_agreements=sum(g['status']=='Active' for g in agreements),closed_agreements=sum(g['status']=='Closed' for g in agreements),
              active_portfolio_balance=round(sum(g['outstanding_principal'] for g in agreements if g['status']=='Active'),2),
              agreements_in_arrears=sum(g['dpd']>0 for g in agreements),overdue_amount=round(sum(g['overdue_amount'] for g in agreements),2),
              payments_received=round(sum(t['amount'] for t in payments),2),principal_received=round(sum(t['principal'] for t in payments),2),
              interest_received=round(sum(t['interest'] for t in payments),2),average_disbursed_amount=round(amount/len(disbursed_ids),2) if disbursed_ids else None,
              average_risk=sum(risks)/len(risks) if risks else None,
              total_credit_limits=round(sum(g['credit_limit'] for g in agreements if g['product']=='revolving' and g['status']=='Active'),2),
              drawn_balance=round(sum(g['outstanding_principal'] for g in agreements if g['product']=='revolving'),2))
    parameters=dict(as_of=day,period_start=start,period_end=end)
    sql_results=_sql_results(agreements,applications,clients,transactions,parameters)
    product_summary=[]
    for product in sorted({g['product'] for g in agreements}):
        rows=[g for g in agreements if g['product']==product]
        product_summary.append(dict(product=product,count=len(rows),active_count=sum(r['status']=='Active' for r in rows),
                                    outstanding_principal=round(sum(r['outstanding_principal'] for r in rows),2),credit_limit=round(sum(r['credit_limit'] for r in rows),2)))
    activity_by_month=defaultdict(lambda:dict(disbursed=0.0,payments=0.0,principal=0.0,interest=0.0))
    for tx in activity:
        bucket=activity_by_month[tx['transaction_date'][:7]]
        if tx['kind'] in ('disbursement','drawdown'):
            bucket['disbursed']+=tx['amount']
        elif tx['kind']=='payment':
            bucket['payments']+=tx['amount'];bucket['principal']+=tx['principal'];bucket['interest']+=tx['interest']
    month=as_date(start).replace(day=1)
    while month<=as_date(end):
        month_end=min(add_months(month,1).toordinal()-1,as_date(end).toordinal())
        closing=date.fromordinal(month_end).isoformat()
        balance=sum(t['amount'] if t['kind'] in ('disbursement','drawdown') else -t['principal'] if t['kind']=='payment' else 0
                    for t in transactions if t['transaction_date']<=closing)
        activity_by_month[month.isoformat()[:7]]['outstanding_principal']=round(balance,2)
        month=add_months(month,1)
    return dict(kpis=kpis,agreements=agreements,applications=period_apps,all_applications=applications,payments=payments,
                clients=clients,transactions=activity,product_summary=product_summary,
                decision_summary=[dict(status='Approved',count=approvals),dict(status='Rejected',count=rejections)],
                status_summary=[dict(status=s,count=sum(g['status']==s for g in agreements)) for s in ('Active','Closed')],
                client_type_summary=[dict(client_type=s,count=sum(c['client_type']==s for c in clients)) for s in ('New','Repeat')],
                dpd_summary=sql_results['arrears_buckets'].to_dict('records'),
                activity_series=[dict(month=m,**{k:round(v,2) for k,v in vals.items()}) for m,vals in sorted(activity_by_month.items())],
                risk_distribution=[dict(application_id=a['application_id'],probability=a['risk_probability']) for a in period_apps if a['risk_probability'] is not None],
                sql_results=sql_results,as_of=day,period_start=start,period_end=end,filters=filters,
                generated_at=datetime.now(timezone.utc).isoformat(),currency=config.get('currency','у.е.'),
                definitions={'portfolio':'Непогашенный основной долг на отчетную дату; только активы платформы.',
                  'approval_rate':'Одобрено / (одобрено + отклонено); заявки за период, последняя версия решения на отчетную дату.',
                  'average_risk':'Средний риск среди решенных заявок за период.',
                  'average_disbursed_amount':'Все фактические выдачи и выборки за период / число договоров с выдачей за период.',
                  'client_type':'На отчетную дату Repeat — клиент с 2+ фактически профинансированными договорами платформы; New — с 0/1. Каждый клиент учитывается один раз. origination_client_type отдельно фиксирует наличие выдачи до заявки.',
                  'dpd':'Дни с самой ранней непогашенной обязанности, срок которой строго раньше отчетной даты.',
                  'interest_received':'Фактически полученные проценты; не чистая прибыль.',
                  'bureau':'Внешние долги исключены из всех активов, выдач и доходов платформы.'})


def export_csv(report, register='agreements'):
    if register=='portfolio_summary':
        rows=[report['kpis']]
    elif register in ('agreements','applications','payments','clients'):
        rows=report[register]
    else:
        raise ValueError('Неизвестный регистр экспорта.')
    frame=pd.DataFrame(rows)
    # Nested technical snapshots are omitted; their originals remain in SQLite.
    frame=frame.drop(columns=[c for c in frame if c.endswith('_json') or c in ('schedule','transactions','payments','inputs','decision_details')],errors='ignore')
    for key in ('generated_at','as_of','period_start','period_end','currency'):
        frame['report_'+key]=report[key]
    if 'source_kind' not in frame:
        frame['source_kind']=str(report.get('filters',{}).get('source_kind','mixed_operational'))
    return frame.to_csv(index=False).encode('utf-8-sig')


get_snapshot=snapshot
