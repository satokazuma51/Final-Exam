-- Original anonymized snapshots, never the platform's own assets.
-- name: external_debt_by_credit_type
SELECT CREDIT_TYPE, COUNT(*) AS credits,
       SUM(AMT_CREDIT_SUM_DEBT) AS observed_external_debt,
       SUM(AMT_CREDIT_SUM_DEBT IS NULL) AS missing_debt_records
FROM bureau GROUP BY CREDIT_TYPE ORDER BY observed_external_debt DESC;

-- name: latest_bureau_month_for_profile
WITH eligible AS (
 SELECT bb.*,b.SK_ID_CURR FROM bureau_balance bb
 JOIN bureau b USING(SK_ID_BUREAU)
 WHERE b.SK_ID_CURR=:application_id AND bb.MONTHS_BALANCE<=0
), ranked AS (
 SELECT *,ROW_NUMBER() OVER(PARTITION BY SK_ID_BUREAU ORDER BY MONTHS_BALANCE DESC) AS rn
 FROM eligible
)
SELECT SK_ID_CURR,SK_ID_BUREAU,MONTHS_BALANCE,STATUS FROM ranked WHERE rn=1;

-- name: financed_goods_applications
SELECT NAME_GOODS_CATEGORY,COUNT(*) AS applications,
       SUM(NAME_CONTRACT_STATUS='Approved') AS approved_applications
FROM previous_application GROUP BY NAME_GOODS_CATEGORY ORDER BY applications DESC;
