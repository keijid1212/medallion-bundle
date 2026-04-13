from pyspark.sql import DataFrame, SparkSession


def read_sales_orders(spark: SparkSession, path: str) -> DataFrame:
    """DBFS の JSON から売上注文を読み込む（生データそのまま）"""
    return spark.read.json(path)


def read_customers(spark: SparkSession, path: str) -> DataFrame:
    """DBFS の CSV から顧客マスタを読み込む（生データそのまま）"""
    return (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv(path)
    )
