WITH rp AS (
  SELECT 
    product_key, product_number, product_name, category, subcategory,
    last_sale_date,
    DATEDIFF(day, last_sale_date, GETDATE()) AS recency_days,  -- R 
    total_orders,                                              -- F
    total_sales                                                -- M
  FROM gold.report_products
  WHERE total_orders > 0     
)
SELECT
  product_key, product_number, product_name, category, subcategory,
  last_sale_date, recency_days,
  total_orders AS frequency,
  total_sales  AS monetary,

  NTILE(4) OVER (PARTITION BY category ORDER BY recency_days ASC, product_key )   AS R_quartile,  -- R
  NTILE(4) OVER (PARTITION BY category ORDER BY total_orders DESC, product_key)  AS F_quartile,  -- F
  NTILE(4) OVER (PARTITION BY category ORDER BY total_sales  DESC, product_key)  AS M_quartile,  -- M

  CONCAT(
    NTILE(4) OVER (PARTITION BY category ORDER BY recency_days ASC, product_key),
    NTILE(4) OVER (PARTITION BY category ORDER BY total_orders DESC, product_key),
    NTILE(4) OVER (PARTITION BY category ORDER BY total_sales  DESC, product_key)
  ) AS rfm_code
FROM rp
ORDER BY category, total_sales DESC, product_key;  

DECLARE @A_cutoff decimal(5,4)=0.80, @B_cutoff decimal(5,4)=0.95;

WITH rp AS (
  SELECT product_key, product_name, category, subcategory,
         last_sale_date, total_orders, total_sales, lifespan
  FROM gold.report_products
),
rfm AS (
  SELECT product_key, category,
         NTILE(4) OVER (PARTITION BY category ORDER BY DATEDIFF(day,last_sale_date,GETDATE()) ASC, product_key) AS R_q,
         NTILE(4) OVER (PARTITION BY category ORDER BY total_orders DESC, product_key) AS F_q,
         NTILE(4) OVER (PARTITION BY category ORDER BY total_sales  DESC, product_key) AS M_q
  FROM rp
),
ranked AS (
  SELECT product_key, category, total_sales,
         SUM(total_sales) OVER (PARTITION BY category) AS grand_total_cat,
         SUM(total_sales) OVER (PARTITION BY category ORDER BY total_sales DESC, product_key) AS cum_sales_cat
  FROM rp
),
abc AS (
  SELECT product_key, category,
         CASE
           WHEN 1.0*cum_sales_cat/NULLIF(grand_total_cat,0) <= @A_cutoff THEN 'A'
           WHEN 1.0*cum_sales_cat/NULLIF(grand_total_cat,0) <= @B_cutoff THEN 'B'
           ELSE 'C'
         END AS abc_class_cat
  FROM ranked
)
SELECT
  rp.category, rp.subcategory, rp.product_key, rp.product_name,
  rp.total_orders, rp.total_sales, abc.abc_class_cat,
  CONCAT(rfm.R_q, rfm.F_q, rfm.M_q) AS rfm_code_cat,
  CASE
    WHEN abc.abc_class_cat='A' AND rfm.R_q>=3 AND rfm.F_q>=3 THEN 'Priority: Keep Lead (stock UP, premium)'
    WHEN abc.abc_class_cat='A' AND rfm.R_q<=2 THEN 'Priority: Reactivate (promo/placement)'
    WHEN abc.abc_class_cat='B' AND rfm.R_q>=3 THEN 'Grow to A (cross-sell, reviews)'
    WHEN abc.abc_class_cat='C' AND rfm.R_q<=2 THEN 'Deprioritize (SKU rationalize)'
    ELSE 'Monitor'
  END AS action_hint
FROM rp
JOIN rfm ON rfm.product_key = rp.product_key
JOIN abc ON abc.product_key = rp.product_key
ORDER BY rp.category, abc.abc_class_cat, rp.total_sales DESC;
