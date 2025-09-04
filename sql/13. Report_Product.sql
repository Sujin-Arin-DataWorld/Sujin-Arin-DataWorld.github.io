CREATE VIEW gold.report_products AS 
WITH base_info AS (
  SELECT 
    f.order_number,
    p.product_key,
    p.product_number,
    p.product_name,
    p.category,
    p.subcategory,
    f.customer_key,
    f.order_date,
    p.cost,              
    f.sales_amount,
    f.quantity
  FROM gold.dim_products p
  LEFT JOIN gold.fact_sales f
    ON p.product_key = f.product_key
  WHERE f.order_date IS NOT NULL
),
product_aggregation AS (
  SELECT
    product_key,
    MAX(product_number) AS product_number,    
    product_name,
    category,
    subcategory,
    MIN(order_date) AS first_dt,
    MAX(order_date) AS last_sale_date,
    DATEDIFF(month, MIN(order_date), MAX(order_date)) AS lifespan,

    ROUND(AVG(CAST(sales_amount AS float) / NULLIF(quantity, 0)), 1) AS avg_selling_price,

    AVG(CAST(cost AS decimal(18,4))) AS avg_unit_cost,

    SUM(CAST(sales_amount AS decimal(18,2)) 
        - CAST(cost AS decimal(18,4)) * CAST(quantity AS decimal(18,2))) AS profit,

    COUNT(DISTINCT customer_key) AS total_customers,
    COUNT(DISTINCT order_number) AS total_orders,
    SUM(CAST(sales_amount AS decimal(18,2))) AS total_sales,
    SUM(CAST(quantity     AS decimal(18,2))) AS total_quantity
  FROM base_info
  GROUP BY product_key, product_name, category, subcategory
)
SELECT
  -- 1) Identifier & Taxonomy
  pa.product_key,
  pa.product_number,
  pa.product_name,
  pa.category,
  pa.subcategory,

  -- 2) Lifecycle & Recency
  pa.first_dt,
  pa.last_sale_date,
  DATEDIFF(month, pa.last_sale_date, GETDATE()) AS recency_in_months,
  pa.lifespan,

  -- 3) Scale
  pa.total_orders,
  pa.total_quantity,
  pa.total_customers,

  -- 4) Revenue
  pa.total_sales,

  -- 5) Pricing / Cost
  pa.avg_selling_price,
  pa.avg_unit_cost,

  -- 6) Unit Economics / KPIs
  pa.profit,
  CASE 
    WHEN pa.total_orders = 0 THEN NULL
    ELSE CAST(pa.total_sales / CAST(pa.total_orders AS decimal(18,2)) AS decimal(18,2))
  END AS AOR,
  CASE 
    WHEN pa.lifespan > 0 
      THEN CAST(pa.total_sales / CAST(pa.lifespan AS decimal(18,2)) AS decimal(18,2))
    ELSE CAST(pa.total_sales AS decimal(18,2))
  END AS avg_monthly_revenue,

  -- 7) Segmentation
  CASE NTILE(5) OVER (ORDER BY pa.total_sales DESC)
    WHEN 1 THEN 'Top 20% - Excellent'
    WHEN 2 THEN 'High'
    WHEN 3 THEN 'Medium'
    WHEN 4 THEN 'Low'
    ELSE 'Bottom 20% - Poor'
  END AS revenue_segment

FROM product_aggregation pa

