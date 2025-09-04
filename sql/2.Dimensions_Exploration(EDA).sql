-- Explore All Countries our customers come from.

SELECT DISTINCT country FROM gold.dim_customers

-- Explore All Category "The major Divistions"
SELECT DISTINCT category, subcategory, product_name From gold.dim_products
ORDER BY 1,2,3
