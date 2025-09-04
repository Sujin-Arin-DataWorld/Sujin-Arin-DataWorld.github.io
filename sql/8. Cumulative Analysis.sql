-- 2. Cumulative Analysis . Aggregate the data progressively over time. 
-- help to understand whether our business is gorwoing or declining. 

-- 2-1. Calcultate the total seles per month
-- and the running total of sales over time.
SELECT
order_date,
total_sales,
SUM(total_sales) OVER (PARTITION BY order_date ORDER BY order_date) AS running_total_sales
--window funtion
FROM
(
SELECT
DATETRUNC(month,order_date) AS order_date,
SUM(sales_amount) AS total_sales
FROM gold.fact_sales
WHERE order_date IS NOT NULL
GROUP BY DATETRUNC(month,order_date)

)t
-- adding each row's value toe the sum of all the previoes row's values.

--2-2 yearly.

SELECT
order_date,
total_sales,
SUM(total_sales) OVER (ORDER BY order_date) AS running_total_sales,
AVG(avg_price) OVER (ORDER BY order_date )AS moving_avg_price
--window funtion
FROM
(
SELECT
DATETRUNC(year,order_date) AS order_date,
SUM(sales_amount) AS total_sales,
AVG(price) AS avg_price
FROM gold.fact_sales
WHERE order_date IS NOT NULL
GROUP BY DATETRUNC(year,order_date)

)t