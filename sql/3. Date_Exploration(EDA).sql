-- Find the date of the first and last order

SELECT 
MIN(order_date) AS first_order_date,
MAX(order_date) AS last_order_date,
DATEDIFF(year, MIN(order_date),MAX(order_date))
 AS order_range_years,
DATEDIFF(month, MIN(order_date),MAX(order_date))
 AS order_range_years

FROM gold.fact_sales;

-- find the youngest and the oldest customer
SELECT
MIN(birthdate) as yougest_birhtdate,
DATEDIFF(year, MIN(birthdate), GETDATE()) AS oldest_age,
MAX(birthdate) AS oldest_birthdate,
DATEDIFF(year, MAX(birthdate), GETDATE()) AS youngest_age
FROM gold.dim_customers