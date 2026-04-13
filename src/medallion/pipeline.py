"""
メダリオンパイプラインのエントリポイント。
Databricks Job から呼び出される。
"""
from pyspark.sql import SparkSession
from medallion import bronze, silver, gold

BRONZE_SALES_PATH   = "dbfs:/databricks-datasets/retail-org/sales_orders/"
BRONZE_CUSTOMER_PATH = "dbfs:/databricks-datasets/retail-org/customers/"
CATALOG = "medallion_learning"


def run(spark: SparkSession) -> None:
    # Bronze → Silver
    raw_orders    = bronze.read_sales_orders(spark, BRONZE_SALES_PATH)
    raw_customers = bronze.read_customers(spark, BRONZE_CUSTOMER_PATH)

    silver_orders    = silver.transform_sales_orders(raw_orders)
    silver_customers = silver.transform_customers(raw_customers)

    silver_orders.write.format("delta").mode("overwrite") \
        .saveAsTable(f"{CATALOG}.silver.sales_orders")
    silver_customers.write.format("delta").mode("overwrite") \
        .saveAsTable(f"{CATALOG}.silver.customers")

    print(f"silver.sales_orders:  {silver_orders.count()} rows")
    print(f"silver.customers:     {silver_customers.count()} rows")

    # Silver → Gold
    gold_summary = gold.build_sales_summary(silver_orders, silver_customers)
    gold_ranking = gold.build_product_ranking(silver_orders)

    gold_summary.write.format("delta").mode("overwrite") \
        .saveAsTable(f"{CATALOG}.gold.sales_summary")
    gold_ranking.write.format("delta").mode("overwrite") \
        .saveAsTable(f"{CATALOG}.gold.product_ranking")

    print(f"gold.sales_summary:   {gold_summary.count()} rows")
    print(f"gold.product_ranking: {gold_ranking.count()} rows")


if __name__ == "__main__":
    spark = SparkSession.builder.appName("medallion_pipeline").getOrCreate()
    run(spark)
