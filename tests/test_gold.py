import pytest
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType,
    ArrayType, IntegerType, TimestampType, BooleanType
)
from medallion.gold import build_product_ranking, build_sales_summary
from datetime import datetime


def make_orders(spark):
    schema = StructType([
        StructField("order_number", LongType()),
        StructField("customer_id",  StringType()),
        StructField("customer_name", StringType()),
        StructField("order_datetime", TimestampType()),
        StructField("number_of_line_items", IntegerType()),
        StructField("ordered_products", ArrayType(StructType([
            StructField("id",    StringType()),
            StructField("name",  StringType()),
            StructField("price", LongType()),
            StructField("qty",   LongType()),
            StructField("curr",  StringType()),
            StructField("unit",  StringType()),
            StructField("promotion_info", StringType()),
        ]))),
        StructField("promo_info", ArrayType(StringType())),
        StructField("silver_loaded_at", TimestampType()),
    ])
    now = datetime.now()
    return spark.createDataFrame([
        Row(order_number=1, customer_id="111", customer_name="Alice",
            order_datetime=now, number_of_line_items=2,
            ordered_products=[
                Row(id="P1", name="Widget A", price=100, qty=2,
                    curr="USD", unit="pcs", promotion_info=None),
                Row(id="P2", name="Widget B", price=50,  qty=1,
                    curr="USD", unit="pcs", promotion_info=None),
            ],
            promo_info=[], silver_loaded_at=now),
        Row(order_number=2, customer_id="222", customer_name="Bob",
            order_datetime=now, number_of_line_items=1,
            ordered_products=[
                Row(id="P1", name="Widget A", price=100, qty=1,
                    curr="USD", unit="pcs", promotion_info=None),
            ],
            promo_info=[], silver_loaded_at=now),
    ], schema=schema)


def make_customers(spark):
    now = datetime.now()
    schema = StructType([
        StructField("customer_id",    StringType()),
        StructField("customer_name",  StringType()),
        StructField("state",          StringType()),
        StructField("loyalty_segment", IntegerType()),
        StructField("is_current",     BooleanType()),
        StructField("surrogate_key",  StringType()),
        StructField("city",           StringType()),
        StructField("postcode",       StringType()),
        StructField("ship_to_address", StringType()),
        StructField("units_purchased", IntegerType()),
        StructField("effective_start_date", TimestampType()),
        StructField("effective_end_date",   TimestampType()),
        StructField("silver_loaded_at",     TimestampType()),
    ])
    return spark.createDataFrame([
        Row(customer_id="111", customer_name="Alice", state="CA",
            loyalty_segment=2, is_current=True, surrogate_key="sk1",
            city="LA", postcode="90001", ship_to_address="CA",
            units_purchased=10, effective_start_date=now,
            effective_end_date=now, silver_loaded_at=now),
        Row(customer_id="222", customer_name="Bob", state="NY",
            loyalty_segment=1, is_current=True, surrogate_key="sk2",
            city="NYC", postcode="10001", ship_to_address="NY",
            units_purchased=5, effective_start_date=now,
            effective_end_date=now, silver_loaded_at=now),
    ], schema=schema)


class TestBuildProductRanking:

    def test_explodes_ordered_products(self, spark):
        """ordered_products が商品単位に展開される"""
        orders = make_orders(spark)
        result = build_product_ranking(orders)
        product_ids = {r.product_id for r in result.collect()}
        assert "P1" in product_ids
        assert "P2" in product_ids

    def test_aggregates_qty_and_revenue(self, spark):
        """P1 は2注文合計: qty=3, revenue=300"""
        orders = make_orders(spark)
        result = build_product_ranking(orders)
        p1 = result.filter(result.product_id == "P1").first()
        assert p1.total_qty_sold == 3
        assert p1.total_revenue  == 300
        assert p1.order_count    == 2


class TestBuildSalesSummary:

    def test_joins_with_current_customers(self, spark):
        """is_current=True の顧客のみ結合される"""
        orders    = make_orders(spark)
        customers = make_customers(spark)
        result    = build_sales_summary(orders, customers)
        assert result.count() == 2

    def test_calculates_total_revenue(self, spark):
        """Alice の合計売上: 100*2 + 50*1 = 250"""
        orders    = make_orders(spark)
        customers = make_customers(spark)
        result    = build_sales_summary(orders, customers)
        alice = result.filter(result.customer_name == "Alice").first()
        assert alice.total_revenue == 250
        assert alice.total_orders  == 1
