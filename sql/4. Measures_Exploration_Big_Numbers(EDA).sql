-- total sales
SELECT SUM(sales_amount) AS total_sales FROM gold.fact_sales
-- how many items are sold
SELECT SUM(quantity) AS total_qunatity FROM gold.fact_sales
-- the avrage selling price
SELECT AVG(price) AS total_price FROM gold.fact_sales
-- the total number of orders
SELECT COUNT(order_number) AS total_orders FROM gold.fact_sales
SELECT COUNT(DISTINCT order_number) AS total_orders FROM gold.fact_sales
-- the total number of products 
SELECT COUNT(DISTINCT product_key) AS total_products FROM gold.dim_products
-- the total number of customers
SELECT COUNT(DISTINCT customer_key) AS total_customers FROM gold.dim_customers;
-- the total number of customers that has placed an order
SELECT COUNT(DISTINCT customer_key) AS total_customers FROM gold.fact_sales;

-- Generate a report that show all key metrics of the besiness

SELECT 'Total Sales'as measure_name, SUM(sales_amount) AS measer_value FROM gold.fact_sales
UNION ALL 
SELECT 'Total Quntity ',  SUM(quantity) AS total_qunatity FROM gold.fact_sales
UNION ALL
SELECT 'Average Price' , AVG(price) AS total_price FROM gold.fact_sales
UNION ALL
SELECT 'Total Orders', COUNT(order_number) AS total_orders FROM gold.fact_sales
UNION ALL
SELECT 'Total Nr. Orders', COUNT(DISTINCT product_key) AS total_products FROM gold.dim_products
UNION ALL
SELECT 'Total Nr. Products', COUNT(DISTINCT customer_key) AS total_customers FROM gold.dim_customers
UNION ALL
SELECT 'Total Nr. Customers', COUNT(DISTINCT customer_key) AS total_customers FROM gold.fact_sales