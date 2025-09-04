
-------------------------------------------------------------------------------------------------------
/*
============================================================================
Customer Report
============================================================================
Purpose: 
	- This report concolidates key customer mertrics and behaviors

Highlights:
	1. Gathers essential fields such as names, ages, and transaction details.
	2. Segments customer into categories (VIP, Regular, New) and age groups.
	3. Aggregates customer-level metrics:
		- total orders
		- total sale
		- total quantity purchased
		- total products
		- lifespan (in months)
	4. calculates valuable KPIs:
		- recency (months since last order)
		- average order value
		- average monthly spend
==================================================================================
*/
CREATE VIEW gold.report_customers AS
WITH base_quary AS (
/*
	Gathers essential fields such as names, ages, and transaction details.*/
SELECT 
f.order_number,
f.product_key,
f.order_date,
f.sales_amount,
f.quantity,
c.customer_key,
c.customer_number,
CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
DATEDIFF(year, c.birthdate,GETDATE()) AS age
FROM gold.fact_sales f
LEFT JOIN gold.dim_customers c
ON c.customer_key= f.customer_key
WHERE f.order_date IS NOT NULL )

,customer_aggregation AS ( 
/*
 Segments customer into categories (VIP, Regular, New) and age groups.*/
SELECT 
customer_key,
customer_number,
customer_name,
age,
COUNT(DISTINCT order_number) AS total_orders,
SUM(sales_amount) AS total_sales,
SUM(quantity) AS total_quantity,
COUNT(DISTINCT product_key) AS total_product,
MAX(order_date) AS last_order_date,
DATEDIFF(month, MIN(order_date), MAX(order_date)) AS lifespan
FROM base_quary
GROUP BY customer_key,
customer_number,
customer_name,
age)

SELECT 
customer_key,
customer_number,
customer_name,
age,
CASE WHEN age < 20 THEN 'under 20'
	WHEN age BETWEEN 20 AND 29 THEN '20s'
	WHEN age BETWEEN 30 AND 39 THEN '30s'
	WHEN age BETWEEN 40 AND 49 THEN '40s'
	WHEN age BETWEEN 50 AND 59 THEN '50s'
	WHEN age BETWEEN 60 AND 69 THEN '60s'
ELSE 'over 70'
END AS age_group,

CASE WHEN total_sales >= 5000 AND lifespan >= 12 THEN 'VIP'
	WHEN total_sales < 5000 AND lifespan >=12 THEN 'Regular'
	ELSE 'NEW'
END AS customer_segment,
last_order_date,
DATEDIFF(month,last_order_date,GETDATE()) AS recency,
total_orders,
total_sales,
total_quantity,
total_product,
lifespan,
-- Coumpute average order value(AOV) 
CASE WHEN total_sales = 0 THEN 0
	ELSE total_sales / total_orders  END AS AOV,
-- Average Monthly Spending = Total Sales / Nr.of Months
CASE WHEN lifespan =0 THEN total_sales
	ELSE total_sales / lifespan
END AS avg_monthly_spending
FROM customer_aggregation
