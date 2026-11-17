"""Explainable educational policy; a genuine trained-model result is mandatory."""
from __future__ import annotations

from datetime import date
import math
from .schedules import annuity_payment, revolving_minimum_payment, as_date


def validate_inputs(config, inputs):
    errors = []
    product = inputs.get('product')
    settings = config.get('products', {}).get(product)
    if product not in ('cash', 'revolving') or not settings:
        errors.append('Выберите поддерживаемый продукт.')
        settings = {}
    required = ('monthly_income', 'amount', 'term_months', 'annual_rate', 'age', 'employment_years', 'existing_monthly_debt', 'essential_expenses')
    values = {}
    for key in required:
        try:
            values[key] = float(inputs[key])
            if not math.isfinite(values[key]):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            errors.append('Заполните корректное числовое поле: ' + key)
    if not errors:
        if values['monthly_income'] <= 0 or values['amount'] <= 0:
            errors.append('Доход и сумма должны быть положительными.')
        if not settings.get('min_amount', 1) <= values['amount'] <= settings.get('max_amount', 1e12):
            errors.append('Сумма находится за пределами лимитов продукта.')
        if values['term_months'] != int(values['term_months']) or not settings.get('min_term_months', 1) <= values['term_months'] <= settings.get('max_term_months', 120):
            errors.append('Укажите допустимый целый срок в месяцах.')
        if not 0 <= values['annual_rate'] <= 100:
            errors.append('Номинальная годовая ставка должна быть от 0 до 100%.')
        if not 0 < values['age'] <= 120 or values['employment_years'] < 0 or values['employment_years'] > values['age']:
            errors.append('Проверьте возраст и стаж.')
        if values['existing_monthly_debt'] < 0 or values['essential_expenses'] < 0:
            errors.append('Обязательства и расходы не могут быть отрицательными.')
        if not str(inputs.get('employment_type', '')).strip():
            errors.append('Укажите вид занятости.')
        if product == 'revolving':
            try:
                draw = float(inputs.get('initial_draw', 0))
                if not math.isfinite(draw) or not 0 <= draw <= values['amount']:
                    raise ValueError()
            except (ValueError, TypeError):
                errors.append('Первоначальная выборка должна быть в пределах кредитного лимита.')
    for key in ('application_date', 'intended_disbursement_date'):
        if inputs.get(key):
            try:
                as_date(inputs[key])
            except (TypeError, ValueError):
                errors.append('Некорректная дата: ' + key)
    if inputs.get('application_date') and inputs.get('intended_disbursement_date'):
        if as_date(inputs['intended_disbursement_date']) < as_date(inputs['application_date']):
            errors.append('Дата выдачи не может предшествовать заявке.')
    if errors:
        raise ValueError('Исправьте входные данные: ' + ' '.join(errors))
    return True


def calculate_terms(config, inputs):
    validate_inputs(config, inputs)
    product = config['products'][inputs['product']]
    if inputs['product'] == 'cash':
        payment = annuity_payment(inputs['amount'], inputs['annual_rate'], inputs['term_months'])
    else:
        # Assess the whole limit, not only an initial zero/small draw: a line can
        # subsequently be fully utilized. The form model still sees actual draw.
        payment = revolving_minimum_payment(inputs['amount'], inputs['annual_rate'],
                                            product.get('revolving_min_principal_pct', .05),
                                            product.get('revolving_min_payment', 100))
    return dict(monthly_payment=float(payment), annual_rate=float(inputs['annual_rate']),
                amount=float(inputs['amount']), term_months=int(inputs['term_months']), product=inputs['product'])


def evaluate_application(config, inputs, score, local_monthly_debt=0, evaluation_date=None):
    terms = calculate_terms(config, inputs)
    if not score or score.get('probability') is None or not score.get('model_version'):
        raise ValueError('Обученная модель недоступна. Выполните команду обучения; окончательное решение не сформировано.')
    probability = float(score['probability'])
    threshold = float(score.get('threshold', .5))
    if not 0 <= probability <= 1 or not 0 < threshold < 1:
        raise ValueError('Некорректный результат обученной модели.')
    policy = config.get('policy', {})
    # Declared payments must include all known obligations; max prevents adding
    # the same local obligation twice and catches understated platform debt.
    existing = max(float(inputs['existing_monthly_debt']), float(local_monthly_debt))
    income = float(inputs['monthly_income'])
    total = existing + terms['monthly_payment']
    dti = total / income
    remaining = income - total - float(inputs['essential_expenses'])
    reasons = []
    age = float(inputs['age'])
    if not policy.get('min_age', 18) <= age <= policy.get('max_age', 75):
        reasons.append(f"Возраст {age:g} вне учебного диапазона {policy.get('min_age',18)}–{policy.get('max_age',75)} лет.")
    if probability >= threshold:
        reasons.append(f'Вероятность платежных трудностей {probability:.1%} достигает порога {threshold:.1%}.')
    if dti > float(policy.get('max_dti', .5)):
        reasons.append(f"Долговая нагрузка {dti:.1%} выше учебного лимита {policy.get('max_dti', .5):.1%}.")
    if remaining < float(policy.get('min_remaining_income', 1000)):
        reasons.append(f"Остаток дохода {remaining:,.2f} у.е. ниже {policy.get('min_remaining_income',1000):,.2f} у.е.")
    decision = 'Rejected' if reasons else 'Approved'
    if not reasons:
        reasons = [f'Риск {probability:.1%} ниже порога {threshold:.1%}.',
                   f'Долговая нагрузка {dti:.1%} и остаток дохода {remaining:,.2f} у.е. удовлетворяют учебной политике.']
    risk_category = 'Low' if probability < threshold / 2 else ('Elevated' if probability < threshold else 'High')
    return dict(decision=decision, reasons=reasons, probability=probability, risk_probability=probability,
                threshold=threshold, risk_category=risk_category,
                risk_boundaries={'Low': [0, threshold / 2], 'Elevated': [threshold / 2, threshold], 'High': [threshold, 1]},
                **terms, total_monthly_debt=round(total, 2), existing_monthly_debt=existing,
                dti=dti, remaining_income=round(remaining, 2),
                debt_combination_rule='max(заявленные обязательства, известные обязательства платформы), без повторного сложения',
                model_version=score['model_version'], policy_version=policy.get('version', 'educational-v1'),
                feature_snapshot=score.get('feature_snapshot', {}), history_used=score.get('history_used', False),
                contributions=score.get('contributions', []), policy=policy,
                evaluation_date=as_date(evaluation_date or date.today()).isoformat())


evaluate = evaluate_application
