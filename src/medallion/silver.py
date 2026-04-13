from pyspark.sql import DataFrame
from pyspark.sql.functions import col, trim, expr, current_timestamp


def transform_sales_orders(df: DataFrame) -> DataFrame:
    """
    Bronze の売上注文をクレンジングして Silver 用に変換する。

    処理内容:
    - customer_id が NULL のレコードを除去
    - order_datetime が 0 以下のレコードを除去
    - number_of_line_items が数値でないレコードを除去
    - 重複レコード（order_number）を除去
    - order_datetime を Unixタイムスタンプ → TIMESTAMP 型に変換
    - number_of_line_items を STRING → INT に変換
    - customer_name の前後スペースを除去
    """
    return (
        df
        .filter(col("customer_id").isNotNull())
        .filter(expr("TRY_CAST(order_datetime AS BIGINT) > 0"))
        .filter(expr("TRY_CAST(number_of_line_items AS INT) IS NOT NULL"))
        .dropDuplicates(["order_number"])
        .withColumn("order_datetime",
            expr("TIMESTAMP_SECONDS(CAST(order_datetime AS BIGINT))"))
        .withColumn("number_of_line_items",
            expr("CAST(number_of_line_items AS INT)"))
        .withColumn("customer_name", trim(col("customer_name")))
        .withColumn("silver_loaded_at", current_timestamp())
    )


def transform_customers(df: DataFrame) -> DataFrame:
    """
    Bronze の顧客マスタを SCD Type 2 形式に変換する。

    追加カラム:
    - surrogate_key      : customer_id + valid_from の MD5
    - effective_start_date: レコード有効開始日時
    - effective_end_date  : レコード有効終了日時（現在レコードは 9999-12-31）
    - is_current          : 最新レコードかどうか
    """
    return (
        df
        .filter(col("customer_id").isNotNull())
        .filter(col("valid_from").isNotNull())
        .withColumn("surrogate_key",
            expr("MD5(CONCAT(customer_id, COALESCE(valid_from, '0')))"))
        .withColumn("customer_name", trim(col("customer_name")))
        .withColumn("postcode", trim(col("postcode")))
        .withColumn("units_purchased",
            expr("CAST(CAST(units_purchased AS DOUBLE) AS INT)"))
        .withColumn("loyalty_segment",
            expr("CAST(loyalty_segment AS INT)"))
        .withColumn("effective_start_date",
            expr("TIMESTAMP_SECONDS(CAST(CAST(valid_from AS DOUBLE) AS BIGINT))"))
        .withColumn("effective_end_date",
            expr("""
                CASE WHEN valid_to IS NULL OR TRIM(valid_to) = ''
                     THEN CAST('9999-12-31' AS TIMESTAMP)
                     ELSE TIMESTAMP_SECONDS(CAST(CAST(valid_to AS DOUBLE) AS BIGINT))
                END
            """))
        .withColumn("is_current",
            expr("CASE WHEN valid_to IS NULL OR TRIM(valid_to) = '' THEN TRUE ELSE FALSE END"))
        .withColumn("silver_loaded_at", current_timestamp())
        .drop("tax_id", "tax_code", "street", "number", "unit",
              "region", "district", "lon", "lat", "valid_from", "valid_to")
    )
