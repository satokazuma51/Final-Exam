-- All queries run against filtered, dated operational snapshot tables built by
-- src.analytics.snapshot. Monetary values are generic currency units.
-- name: application_rates
SELECT a.product, COUNT(*) AS applications,
 SUM(a.status='Approved') AS approvals, SUM(a.status='Rejected') AS rejections,
 1.0*SUM(a.status='Approved')/NULLIF(SUM(a.status IN('Approved','Rejected')),0) AS approval_rate
FROM report_applications a JOIN report_clients c USING(client_id)
WHERE a.application_date BETWEEN :period_start AND :period_end
GROUP BY a.product;

-- name: portfolio_by_client
SELECT c.client_id,c.full_name,COUNT(g.agreement_id) AS agreements,
 SUM(g.outstanding_principal) AS outstanding_principal,SUM(g.overdue_amount) AS overdue_amount
FROM report_clients c JOIN report_agreements g USING(client_id)
WHERE g.opened_date<=:as_of GROUP BY c.client_id,c.full_name
HAVING SUM(g.outstanding_principal)>0 ORDER BY outstanding_principal DESC;

-- name: principal_reconciliation
SELECT g.agreement_id,g.outstanding_principal,
 COALESCE(SUM(CASE WHEN t.kind IN('disbursement','drawdown') THEN t.amount
 WHEN t.kind='payment' THEN -t.principal ELSE 0 END),0) AS ledger_principal
FROM report_agreements g LEFT JOIN report_transactions t
 ON t.agreement_id=g.agreement_id AND t.transaction_date<=:as_of
GROUP BY g.agreement_id,g.outstanding_principal;

-- name: arrears_buckets
SELECT CASE WHEN dpd=0 THEN '0' WHEN dpd<=30 THEN '1–30' WHEN dpd<=60 THEN '31–60'
 WHEN dpd<=90 THEN '61–90' ELSE '91+' END AS dpd_bucket,
 COUNT(*) AS agreements,SUM(outstanding_principal) AS outstanding_principal,SUM(overdue_amount) AS overdue_amount
FROM report_agreements GROUP BY dpd_bucket;

-- name: repeat_applicants
SELECT c.client_id,c.full_name,COUNT(*) AS application_count
FROM report_clients c JOIN report_applications a USING(client_id)
WHERE a.application_date<=:as_of GROUP BY c.client_id,c.full_name HAVING COUNT(*)>1;

-- name: monthly_payment_running_total
WITH monthly AS (
 SELECT substr(transaction_date,1,7) AS month,SUM(amount) AS payments,SUM(principal) AS principal,SUM(interest) AS interest
 FROM report_transactions WHERE kind='payment' AND transaction_date BETWEEN :period_start AND :period_end GROUP BY month)
SELECT *,SUM(payments) OVER(ORDER BY month ROWS UNBOUNDED PRECEDING) AS running_payments FROM monthly ORDER BY month;

-- name: monthly_disbursement_change
WITH monthly AS (
 SELECT substr(transaction_date,1,7) AS month,SUM(amount) AS disbursed
 FROM report_transactions WHERE kind IN('disbursement','drawdown') AND transaction_date BETWEEN :period_start AND :period_end GROUP BY month)
SELECT *,LAG(disbursed) OVER(ORDER BY month) AS previous_month,
 disbursed-LAG(disbursed) OVER(ORDER BY month) AS change FROM monthly ORDER BY month;

-- name: new_repeat_comparison
SELECT client_type,COUNT(*) AS agreements,COUNT(DISTINCT client_id) AS clients,
 SUM(total_disbursed) AS cumulative_disbursed,SUM(outstanding_principal) AS outstanding_principal
FROM report_agreements GROUP BY client_type;

-- name: latest_decision_per_client
WITH ranked AS (
 SELECT a.*,ROW_NUMBER() OVER(PARTITION BY client_id ORDER BY application_date DESC,application_id DESC) AS row_no
 FROM report_applications a WHERE status IN('Approved','Rejected') AND application_date<=:as_of)
SELECT client_id,application_id,status,risk_probability FROM ranked WHERE row_no=1;

-- name: product_portfolio
SELECT g.product,COUNT(*) AS agreements,SUM(g.status='Active') AS active_agreements,
 SUM(g.outstanding_principal) AS outstanding_principal,SUM(g.credit_limit) AS credit_limits,
 AVG(a.risk_probability) AS average_risk
FROM report_agreements g JOIN report_applications a USING(application_id)
GROUP BY g.product;
