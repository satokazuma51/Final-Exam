"""Dated, replayable operational accounting in integer cents.

Cash: interest is charged monthly on actual principal at each contractual due
date; no daily interim interest, fees or penalty interest. Early principal is
applied to the oldest remaining scheduled principal, reducing future interest.
Revolving: ACT/365 simple daily interest on drawn principal, monthly statements,
interest-first payment allocation, then due principal, then other principal.
Backdating before an already recorded event is rejected to keep allocations
immutable. Historical snapshots can be taken for any earlier date.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import json

from .db import connect, now_iso, json_text
from .schedules import as_date, add_months, annuity_schedule, to_cents, from_cents


def _round(value):
    return int(Decimal(value).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def _rows(connection, agreement_id):
    row = connection.execute('''SELECT g.*,c.full_name FROM agreements g JOIN clients c USING(client_id) WHERE agreement_id=?''', (int(agreement_id),)).fetchone()
    if not row:
        raise ValueError('Договор не найден.')
    schedule = [dict(r) for r in connection.execute('SELECT * FROM scheduled_installments WHERE agreement_id=? ORDER BY installment_no', (int(agreement_id),))]
    transactions = [dict(r) for r in connection.execute('SELECT * FROM transactions WHERE agreement_id=? ORDER BY transaction_date,transaction_id', (int(agreement_id),))]
    return dict(row), schedule, transactions


def _replay(agreement, original_schedule, transactions, as_of):
    day = as_date(as_of)
    opened = as_date(agreement['opened_date'])
    if day < opened:
        raise ValueError('Договор еще не существовал на отчетную дату.')
    cash = agreement['product'] == 'cash'
    rate = Decimal(str(agreement['annual_rate'])) / 100
    obligations = []
    if cash:
        obligations = [dict(number=r['installment_no'], due=as_date(r['due_date']),
                            principal=r['principal_cents'], interest=0, principal_paid=0,
                            interest_paid=0, billed=False) for r in original_schedule]
    events = [(as_date(t['transaction_date']), 1, t['transaction_id'], 'transaction', t)
              for t in transactions if as_date(t['transaction_date']) <= day]
    if cash:
        events += [(o['due'], 0, o['number'], 'due', o) for o in obligations if o['due'] <= day]
    else:
        number = 1
        while add_months(opened, number) <= day:
            events.append((add_months(opened, number), 0, number, 'due', None))
            number += 1
            if number > 2400:
                raise ValueError('Слишком длинный период расчета.')
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    principal = 0
    unbilled_interest = Decimal(0)
    last_date = opened
    total_disbursed = 0
    total_paid = 0
    total_principal_paid = 0
    total_interest_paid = 0
    max_dpd = 0
    closed_date = None
    payment_details = []

    def unpaid_interest():
        return sum(max(0, o['interest'] - o['interest_paid']) for o in obligations) + _round(unbilled_interest)

    def current_dpd(on_date):
        unpaid = [o['due'] for o in obligations if o['billed'] and o['due'] < on_date and
                  o['principal'] - o['principal_paid'] + o['interest'] - o['interest_paid'] > 0]
        return (on_date - min(unpaid)).days if unpaid else 0

    for event_date, _, number, kind, payload in events:
        if not cash and principal:
            unbilled_interest += Decimal(principal) * rate * Decimal((event_date - last_date).days) / Decimal(365)
        last_date = event_date
        max_dpd = max(max_dpd, current_dpd(event_date))
        if kind == 'due':
            if cash:
                payload['interest'] = _round(Decimal(principal) * rate / 12) if principal > 0 else 0
                payload['billed'] = True
            elif closed_date is None:
                reserved = sum(max(0, o['principal'] - o['principal_paid']) for o in obligations)
                unreserved = max(0, principal - reserved)
                minimum = min(unreserved, max(agreement['min_payment_cents'], _round(Decimal(principal) * Decimal(str(agreement['min_principal_pct'])))))
                billed_interest = _round(unbilled_interest)
                unbilled_interest = Decimal(0)
                obligations.append(dict(number=number, due=event_date, principal=minimum, interest=billed_interest,
                                        principal_paid=0, interest_paid=0, billed=True))
            continue
        tx = payload
        amount = int(tx['amount_cents'])
        if tx['kind'] in ('disbursement', 'drawdown'):
            principal += amount
            total_disbursed += amount
        elif tx['kind'] == 'line_closure':
            closed_date = event_date.isoformat()
        elif tx['kind'] == 'payment':
            remaining = amount
            paid_interest = 0
            for obligation in obligations:
                due_interest = max(0, obligation['interest'] - obligation['interest_paid'])
                paid = min(remaining, due_interest)
                obligation['interest_paid'] += paid
                paid_interest += paid
                remaining -= paid
            if not cash and remaining:
                paid = min(remaining, _round(unbilled_interest))
                # Round accrued interest at payment events; no hidden fraction
                # can prevent a correctly quoted settlement or line closure.
                unbilled_interest = Decimal(_round(unbilled_interest) - paid)
                paid_interest += paid
                remaining -= paid
            paid_principal = min(principal, remaining)
            allocation = paid_principal
            for obligation in obligations:
                paid = min(allocation, max(0, obligation['principal'] - obligation['principal_paid']))
                obligation['principal_paid'] += paid
                allocation -= paid
                if not allocation:
                    break
            if remaining > principal:
                raise ValueError('Платеж превышает обязательства на дату операции.')
            principal -= paid_principal
            total_paid += amount
            total_principal_paid += paid_principal
            total_interest_paid += paid_interest
            payment_details.append(dict(transaction_id=tx['transaction_id'], agreement_id=agreement['agreement_id'],
                transaction_date=event_date.isoformat(), payment_date=event_date.isoformat(), amount=from_cents(amount),
                principal=from_cents(paid_principal), interest=from_cents(paid_interest), operation_key=tx['operation_key']))
            if cash and principal == 0 and unpaid_interest() == 0:
                closed_date = event_date.isoformat()
    if not cash and principal:
        unbilled_interest += Decimal(principal) * rate * Decimal((day - last_date).days) / Decimal(365)
    max_dpd = max(max_dpd, current_dpd(day))
    overdue = sum(max(0, o['principal'] - o['principal_paid']) + max(0, o['interest'] - o['interest_paid'])
                  for o in obligations if o['billed'] and o['due'] < day)
    schedule = []
    projected_balance = principal
    for o in obligations:
        outstanding_component = max(0, o['principal'] - o['principal_paid'])
        if o['billed']:
            interest = o['interest']
        else:
            interest = _round(Decimal(projected_balance) * rate / 12) if projected_balance > 0 else 0
            projected_balance = max(0, projected_balance - outstanding_component)
        unpaid = outstanding_component + max(0, interest - o['interest_paid'])
        schedule.append(dict(installment_no=o['number'], due_date=o['due'].isoformat(),
            payment=from_cents(o['principal'] + interest), principal=from_cents(o['principal']), interest=from_cents(interest),
            remaining_principal=from_cents(projected_balance if not o['billed'] else max(0, sum(max(0, x['principal']-x['principal_paid']) for x in obligations if x['number'] > o['number']))),
            actual_paid=from_cents(o['principal_paid'] + o['interest_paid']), unpaid_amount=from_cents(unpaid),
            dpd=(day-o['due']).days if o['due'] < day and unpaid > 0 else 0,
            principal_paid=from_cents(o['principal_paid']), interest_paid=from_cents(o['interest_paid']), billed=o['billed']))
    result = dict(agreement)
    result.update(as_of=day.isoformat(), status='Closed' if closed_date else 'Active', closed_date=closed_date,
        outstanding_principal=from_cents(principal), drawn_principal=from_cents(principal),
        accrued_interest=from_cents(unpaid_interest()), settlement_amount=from_cents(principal + unpaid_interest()),
        available_credit=from_cents(max(0, agreement['credit_limit_cents'] - principal)) if not cash and not closed_date else 0.0,
        credit_limit=from_cents(agreement['credit_limit_cents']), overdue_amount=from_cents(overdue),
        dpd=current_dpd(day), max_dpd=max_dpd, total_disbursed=from_cents(total_disbursed), payment_total=from_cents(total_paid),
        principal_paid=from_cents(total_principal_paid), interest_paid=from_cents(total_interest_paid),
        schedule=schedule, payments=payment_details,
        transactions=[dict(t, amount=from_cents(t['amount_cents']), principal=from_cents(t['principal_cents']), interest=from_cents(t['interest_cents']))
                      for t in transactions if as_date(t['transaction_date']) <= day],
        interest_method='monthly_on_due_date' if cash else 'ACT/365_daily_simple_monthly_statement',
        payment_allocation='начисленные проценты → основной долг; будущие неначисленные проценты не взыскиваются')
    return result


def agreement_snapshot(config, agreement_id, as_of=None):
    with connect(config) as connection:
        agreement, schedule, transactions = _rows(connection, agreement_id)
    return _replay(agreement, schedule, transactions, as_of or date.today())


def settlement_quote(config, agreement_id, as_of=None):
    snapshot = agreement_snapshot(config, agreement_id, as_of)
    return dict(agreement_id=int(agreement_id), as_of=snapshot['as_of'], amount=snapshot['settlement_amount'],
                principal=snapshot['outstanding_principal'], interest=snapshot['accrued_interest'])


def _idempotent(connection, operation_key, request):
    if not operation_key:
        raise ValueError('Требуется уникальный ключ операции.')
    existing = connection.execute('SELECT * FROM transactions WHERE operation_key=?', (operation_key,)).fetchone()
    if existing:
        if existing['request_json'] != json_text(request):
            raise ValueError('Ключ операции уже использован с другими параметрами.')
        return dict(existing)
    return None


def _transaction(connection, agreement_id, day, kind, amount, principal, interest, operation_key, request):
    cursor = connection.execute('''INSERT INTO transactions(agreement_id,transaction_date,kind,amount_cents,principal_cents,interest_cents,operation_key,request_json,created_at)
         VALUES(?,?,?,?,?,?,?,?,?)''', (int(agreement_id), day, kind, amount, principal, interest, operation_key, json_text(request), now_iso()))
    return dict(transaction_id=cursor.lastrowid, agreement_id=int(agreement_id), transaction_date=day, kind=kind,
                amount=from_cents(amount), principal=from_cents(principal), interest=from_cents(interest), operation_key=operation_key)


def _operation_date(agreement, transactions, value):
    day = as_date(value).isoformat()
    if day < agreement['opened_date']:
        raise ValueError('Операция не может предшествовать выдаче.')
    if transactions and day < max(t['transaction_date'] for t in transactions):
        raise ValueError('Операция раньше уже записанного события: вводите события договора по порядку дат.')
    return day


def disburse(config, application_id, disbursement_date, operation_key):
    day = as_date(disbursement_date).isoformat()
    request = dict(application_id=int(application_id), date=day, kind='disbursement')
    with connect(config) as connection:
        connection.execute('BEGIN IMMEDIATE')
        duplicate = _idempotent(connection, operation_key, request)
        if duplicate:
            return duplicate['agreement_id']
        app = connection.execute('SELECT * FROM applications WHERE application_id=?', (int(application_id),)).fetchone()
        if not app:
            raise ValueError('Заявка не найдена.')
        existing = connection.execute('SELECT agreement_id FROM agreements WHERE application_id=?', (int(application_id),)).fetchone()
        if existing:
            raise ValueError('По этой заявке выдача уже зарегистрирована.')
        decision = connection.execute('SELECT * FROM decisions WHERE application_id=? ORDER BY version DESC LIMIT 1', (int(application_id),)).fetchone()
        if not decision or decision['decision'] != 'Approved':
            raise ValueError('Выдача возможна только по одобренной заявке.')
        if day < app['application_date'] or day < str(decision['decision_date'])[:10]:
            raise ValueError('Выдача не может предшествовать заявке и решению.')
        product = config.get('products', {}).get(app['product'], {})
        amount = app['amount_cents'] if app['product'] == 'cash' else app['initial_draw_cents']
        limit = app['amount_cents'] if app['product'] == 'revolving' else 0
        cursor = connection.execute('''INSERT INTO agreements(application_id,decision_id,client_id,product,opened_date,scheduled_end_date,annual_rate,credit_limit_cents,initial_principal_cents,term_months,min_principal_pct,min_payment_cents,source_kind,is_synthetic,source_application_id,created_at)
             VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (int(application_id),decision['decision_id'],app['client_id'],app['product'],day,
             add_months(day,app['term_months']).isoformat(), app['annual_rate'],limit,amount,app['term_months'],
             float(product.get('revolving_min_principal_pct',.05)),to_cents(product.get('revolving_min_payment',100)),
             app['source_kind'],app['is_synthetic'],app['source_application_id'],now_iso()))
        agreement_id = cursor.lastrowid
        if app['product'] == 'cash':
            schedule = annuity_schedule(from_cents(amount),app['annual_rate'],app['term_months'],day)
            connection.executemany('INSERT INTO scheduled_installments(agreement_id,installment_no,due_date,payment_cents,principal_cents,interest_cents,remaining_principal_cents) VALUES(?,?,?,?,?,?,?)',
                [(agreement_id,row['installment_no'],row['due_date'],to_cents(row['payment']),to_cents(row['principal']),to_cents(row['interest']),to_cents(row['remaining_principal'])) for row in schedule])
        _transaction(connection,agreement_id,day,'disbursement',amount,amount,0,operation_key,request)
        return agreement_id


def record_payment(config, agreement_id, amount, payment_date, operation_key, **metadata):
    cents = to_cents(amount)
    if cents <= 0:
        raise ValueError('Сумма платежа должна быть положительной.')
    day = as_date(payment_date).isoformat()
    request = dict(agreement_id=int(agreement_id), amount_cents=cents,date=day,kind='payment')
    with connect(config) as connection:
        connection.execute('BEGIN IMMEDIATE')
        duplicate = _idempotent(connection,operation_key,request)
        if duplicate:
            return dict(duplicate, amount=from_cents(duplicate['amount_cents']), principal=from_cents(duplicate['principal_cents']), interest=from_cents(duplicate['interest_cents']))
        agreement,schedule,transactions = _rows(connection,agreement_id)
        _operation_date(agreement,transactions,day)
        state = _replay(agreement,schedule,transactions,day)
        if state['status'] == 'Closed':
            raise ValueError('Договор закрыт.')
        interest = min(cents,to_cents(state['accrued_interest']))
        principal = cents-interest
        if principal > to_cents(state['outstanding_principal']):
            raise ValueError(f"Платеж превышает сумму полного погашения {state['settlement_amount']:.2f} у.е.")
        return _transaction(connection,agreement_id,day,'payment',cents,principal,interest,operation_key,request)


def drawdown(config, agreement_id, amount, draw_date, operation_key):
    cents=to_cents(amount)
    if cents<=0:
        raise ValueError('Сумма выборки должна быть положительной.')
    day=as_date(draw_date).isoformat()
    request=dict(agreement_id=int(agreement_id),amount_cents=cents,date=day,kind='drawdown')
    with connect(config) as connection:
        connection.execute('BEGIN IMMEDIATE')
        duplicate=_idempotent(connection,operation_key,request)
        if duplicate:
            return dict(duplicate,amount=from_cents(duplicate['amount_cents']))
        agreement,schedule,transactions=_rows(connection,agreement_id)
        _operation_date(agreement,transactions,day)
        state=_replay(agreement,schedule,transactions,day)
        if agreement['product']!='revolving' or state['status']!='Active':
            raise ValueError('Выборка доступна только по открытой возобновляемой линии.')
        if day>agreement['scheduled_end_date']:
            raise ValueError('Срок доступности линии завершен.')
        if cents>to_cents(state['available_credit']):
            raise ValueError('Недостаточно доступного кредитного лимита.')
        return _transaction(connection,agreement_id,day,'drawdown',cents,cents,0,operation_key,request)


def close_line(config, agreement_id, closure_date, operation_key):
    day=as_date(closure_date).isoformat()
    request=dict(agreement_id=int(agreement_id),date=day,kind='line_closure')
    with connect(config) as connection:
        connection.execute('BEGIN IMMEDIATE')
        duplicate=_idempotent(connection,operation_key,request)
        if duplicate:
            return dict(duplicate)
        agreement,schedule,transactions=_rows(connection,agreement_id)
        _operation_date(agreement,transactions,day)
        state=_replay(agreement,schedule,transactions,day)
        if agreement['product']!='revolving':
            raise ValueError('Отдельное закрытие предусмотрено для кредитной линии.')
        if state['status']=='Closed':
            raise ValueError('Линия уже закрыта.')
        if to_cents(state['settlement_amount'])!=0:
            raise ValueError('Перед закрытием погасите основной долг и начисленные проценты.')
        return _transaction(connection,agreement_id,day,'line_closure',0,0,0,operation_key,request)


def list_agreements(config, client_id=None, as_of=None):
    day=as_date(as_of or date.today()).isoformat()
    with connect(config) as connection:
        ids=connection.execute('SELECT agreement_id FROM agreements WHERE opened_date<=?'+(' AND client_id=?' if client_id is not None else '')+' ORDER BY agreement_id DESC',
            (day,int(client_id)) if client_id is not None else (day,)).fetchall()
    return [agreement_snapshot(config,row['agreement_id'],day) for row in ids]


def local_monthly_debt(config, client_id, as_of=None):
    from .schedules import annuity_payment,revolving_minimum_payment
    total=0
    for agreement in list_agreements(config,client_id,as_of):
        if agreement['status']=='Closed':
            continue
        if agreement['product']=='cash':
            total+=min(agreement['settlement_amount'],float(annuity_payment(from_cents(agreement['initial_principal_cents']),agreement['annual_rate'],agreement['term_months'])))
        else:
            total+=float(revolving_minimum_payment(agreement['outstanding_principal'],agreement['annual_rate'],agreement['min_principal_pct'],from_cents(agreement['min_payment_cents'])))
    return round(total,2)
