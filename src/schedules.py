"""Decimal money and month-anchored repayment calculations (generic currency units)."""
from __future__ import annotations

import calendar
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP, localcontext

CENT = Decimal('0.01')


def as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def money(value) -> Decimal:
    result = Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)
    if not result.is_finite():
        raise ValueError('Денежная сумма должна быть конечным числом.')
    return result


def to_cents(value) -> int:
    return int(money(value) * 100)


def from_cents(value: int) -> float:
    return float(Decimal(int(value)) / 100)


def add_months(value, months: int) -> date:
    """Anchor every due date to the original day, preserving month-end."""
    original = as_date(value)
    serial = original.year * 12 + original.month - 1 + int(months)
    year, month0 = divmod(serial, 12)
    month = month0 + 1
    is_month_end = original.day == calendar.monthrange(original.year, original.month)[1]
    day = calendar.monthrange(year, month)[1] if is_month_end else min(original.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def annuity_payment(amount, annual_rate, term_months: int) -> Decimal:
    principal = money(amount)
    rate = Decimal(str(annual_rate)) / 1200
    n = int(term_months)
    if principal <= 0 or n <= 0 or rate < 0:
        raise ValueError('Сумма и срок должны быть положительными, ставка — неотрицательной.')
    with localcontext() as context:
        context.prec = 40
        payment = principal / n if rate == 0 else principal * rate / (1 - (1 + rate) ** (-n))
    return money(payment)


def annuity_schedule(amount, annual_rate, term_months: int, disbursement_date) -> list[dict]:
    principal = money(amount)
    payment = annuity_payment(principal, annual_rate, term_months)
    rate = Decimal(str(annual_rate)) / 1200
    balance = principal
    result = []
    for number in range(1, int(term_months) + 1):
        interest = money(balance * rate)
        component = min(balance, max(Decimal('0'), payment - interest))
        if number == int(term_months):
            component = balance
        total = component + interest
        balance = money(balance - component)
        result.append(dict(installment_no=number, due_date=add_months(disbursement_date, number).isoformat(),
                           payment=float(total), principal=float(component), interest=float(interest),
                           remaining_principal=float(balance), actual_paid=0.0, unpaid_amount=float(total), dpd=0))
    return result


def revolving_minimum_payment(drawn, annual_rate, min_principal_pct=0.05, min_payment=100) -> Decimal:
    principal = money(drawn)
    if principal <= 0:
        return Decimal('0.00')
    return money(min(principal, max(money(min_payment), money(principal * Decimal(str(min_principal_pct))))) + principal * Decimal(str(annual_rate)) / 1200)


def revolving_scenario(limit, drawn, annual_rate, disbursement_date, months=120,
                       min_principal_pct=0.05, min_payment=100) -> list[dict]:
    """Illustration only: monthly-rate estimate, no additional draws, timely minima."""
    balance = money(drawn)
    if balance < 0 or balance > money(limit):
        raise ValueError('Использованная сумма должна быть в пределах кредитного лимита.')
    result = []
    for number in range(1, int(months) + 1):
        if balance <= 0:
            break
        interest = money(balance * Decimal(str(annual_rate)) / 1200)
        principal = min(balance, max(money(min_payment), money(balance * Decimal(str(min_principal_pct)))))
        balance = money(balance - principal)
        result.append(dict(installment_no=number, due_date=add_months(disbursement_date, number).isoformat(),
                           payment=float(principal + interest), principal=float(principal), interest=float(interest),
                           remaining_principal=float(balance), scenario=True))
    return result


generate_annuity_schedule = annuity_schedule
