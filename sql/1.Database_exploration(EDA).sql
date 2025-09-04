-- Explore ALL Objects in the Database
SELECT *FROM INFORMATION_SCHEMA.TABLES

--Explore All columns in the Database
SELECT*FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'dim_customers'