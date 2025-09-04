--11. Data Segmentaiton
-- Group the data based on a specific range.
-- Helps understand the correlation between two measures

-- 11 .1 Segment products into cost ranges and count how many products fall into each segment.
WITH product_segment AS ( 
SELECT
product_key,
product_name,
cost,
CASE WHEN cost < 100 THEN 'Below 100'
     WHEN cost BETWEEN 100 AND 500 THEN '100-500'
	 WHEN cost BETWEEN 500 AND 1000 THEN '500-1000'
	 ELSE 'Above 1000'
END cost_range
FROM gold.dim_products)

SELECT 
cost_range,
COUNT(product_key) AS total_products
FROM product_segment
GROUP BY cost_range
ORDER BY total_products DESC;

/*- group customer into three segments based on their spending behavior
 vip: at least 12 months of history and spending more than 5000 euro
 regualr : at least 12 months of history but spending 5000 euro or less
 new: lifespan less then 12 months 
*/
WITH segment_info AS(
SELECT 
c.customer_key,
SUM(f.sales_amount) AS total_spending,
MIN(f.order_date) AS first_order,
MAX(f.order_date) AS last_order_date,
DATEDIFF(month,MIN(order_date),MAX(order_date)) AS lifespan
FROM gold.dim_customers c
LEFT JOIN gold.fact_sales f
ON c.customer_key = f.customer_key
WHERE f.order_date IS NOT NULL
GROUP BY c.customer_key)

SELECT
segment,
COUNT(customer_key) total_customer
FROM (
SELECT 
customer_key,
CASE WHEN total_spending >= 5000 AND lifespan >= 12 THEN 'VIP'
	WHEN total_spending < 5000 AND lifespan >=12 THEN 'Regular'
	ELSE 'NEW'
END AS segment
FROM segment_info)t
GROUP BY segment
ORDER BY total_customer DESC
