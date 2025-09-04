---
name: Sales Data Analysis (SQL)
tools: [SQL, Database]
image: /assets/images/sql-project-thumbnail.png
description: Sales data analysis using SQL – customer & product reports, RFM segmentation, and ABC/Pareto classification
---

# Sales Data Analysis (SQL Only)

## 📊 Overview
- **Stack**: SQL Server (Azure SQL)
- **Goal**: Build consolidated **customer** and **product** reports via SQL, then derive actionable insights with **RFM** and **ABC/Pareto**.

## 🗃️ Data
- **gold.fact_sales** — orders, sales_amount, quantity, order_date, keys  
- **gold.dim_customers** — customer attributes  
- **gold.dim_products** — product attributes, category, cost

---

## 🧩 Customer Report (View)
Key metrics: total_orders, total_sales, total_quantity, **recency** (months since last order), **lifespan**, **AOV**, **avg_monthly_spending**, **age_group**, **customer_segment** (VIP/Regular/New).

**View:** `gold.report_customers`  
**SQL:** [12.Report_Customer.sql](/sql/12.Report_Customer.sql)

```sql
-- excerpt
CREATE VIEW gold.report_customers AS
WITH base_query AS (
  SELECT f.order_number, f.product_key, f.order_date, f.sales_amount, f.quantity,
         c.customer_key, c.customer_number,
         CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
         DATEDIFF(year, c.birthdate, GETDATE()) AS age
  FROM gold.fact_sales f
  LEFT JOIN gold.dim_customers c ON c.customer_key = f.customer_key
  WHERE f.order_date IS NOT NULL
),
customer_aggregation AS (
  SELECT customer_key, customer_number, customer_name, age,
         COUNT(DISTINCT order_number) AS total_orders,
         SUM(sales_amount) AS total_sales,
         SUM(quantity) AS total_quantity,
         COUNT(DISTINCT product_key) AS total_product,
         MAX(order_date) AS last_order_date,
         DATEDIFF(month, MIN(order_date), MAX(order_date)) AS lifespan
  FROM base_query
  GROUP BY customer_key, customer_number, customer_name, age
)
SELECT ..., -- (see full SQL)
       CASE WHEN total_orders=0 THEN 0 ELSE total_sales/total_orders END AS AOV,
       CASE WHEN lifespan=0 THEN total_sales ELSE total_sales/lifespan END AS avg_monthly_spending
FROM customer_aggregation;
