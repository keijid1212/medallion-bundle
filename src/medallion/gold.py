from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr, count, sum, min, max, current_timestamp


def build_sales_summary(orders: DataFrame, customers: DataFrame) -> DataFrame:
    """
    Silver の売上注文 × 顧客（現在レコード）を結合して顧客別売上サマリーを作る。
    """
    current_customers = customers.filter(col("is_current") == True)

    orders_rev = orders.withColumn(
        "order_revenue",
        expr("AGGREGATE(ordered_products, 0L, (acc, p) -> acc + p.price * p.qty)")
    )

    return (
        orders_rev
        .join(current_customers,
              orders_rev.customer_id == current_customers.customer_id, "left")
        .groupBy(
            current_customers.customer_id,
            current_customers.customer_name,
            current_customers.state,
            current_customers.loyalty_segment,
        )
        .agg(
            count(orders_rev.order_number).alias("total_orders"),
            sum(orders_rev.number_of_line_items).alias("total_line_items"),
            sum("order_revenue").alias("total_revenue"),
            min(orders_rev.order_datetime).alias("first_order_date"),
            max(orders_rev.order_datetime).alias("last_order_date"),
            current_timestamp().alias("gold_loaded_at"),
        )
    )


def build_product_ranking(orders: DataFrame) -> DataFrame:
    """
    Silver の売上注文から ordered_products を展開して商品別売上ランキングを作る。
    """
    return (
        orders
        .selectExpr("order_number", "EXPLODE(ordered_products) AS p")
        .selectExpr(
            "p.id            AS product_id",
            "p.name          AS product_name",
            "p.qty           AS qty",
            "p.price * p.qty AS revenue",
            "order_number",
        )
        .groupBy("product_id", "product_name")
        .agg(
            sum("qty").alias("total_qty_sold"),
            sum("revenue").alias("total_revenue"),
            count("order_number").alias("order_count"),
            current_timestamp().alias("gold_loaded_at"),
        )
    )
