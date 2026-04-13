import pytest
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, ArrayType
)
from medallion.silver import transform_sales_orders, transform_customers

# NULL を含む Row を createDataFrame に渡すとスキーマ推論に失敗するため
# 明示的にスキーマを定義する

ORDER_SCHEMA = StructType([
    StructField("order_number",        LongType(),             nullable=False),
    StructField("customer_id",         StringType(),           nullable=True),
    StructField("customer_name",       StringType(),           nullable=True),
    StructField("order_datetime",      StringType(),           nullable=True),
    StructField("number_of_line_items", StringType(),          nullable=True),
    StructField("ordered_products",    ArrayType(StringType()), nullable=True),
    StructField("promo_info",          ArrayType(StringType()), nullable=True),
])

CUSTOMER_SCHEMA = StructType([
    StructField("customer_id",     StringType(), nullable=True),
    StructField("tax_id",          StringType(), nullable=True),
    StructField("tax_code",        StringType(), nullable=True),
    StructField("customer_name",   StringType(), nullable=True),
    StructField("state",           StringType(), nullable=True),
    StructField("city",            StringType(), nullable=True),
    StructField("postcode",        StringType(), nullable=True),
    StructField("street",          StringType(), nullable=True),
    StructField("number",          StringType(), nullable=True),
    StructField("unit",            StringType(), nullable=True),
    StructField("region",          StringType(), nullable=True),
    StructField("district",        StringType(), nullable=True),
    StructField("lon",             StringType(), nullable=True),
    StructField("lat",             StringType(), nullable=True),
    StructField("ship_to_address", StringType(), nullable=True),
    StructField("valid_from",      StringType(), nullable=True),
    StructField("valid_to",        StringType(), nullable=True),
    StructField("units_purchased", StringType(), nullable=True),
    StructField("loyalty_segment", StringType(), nullable=True),
])


class TestTransformSalesOrders:

    def _make_order(self, spark, rows):
        return spark.createDataFrame(rows, schema=ORDER_SCHEMA)

    def test_removes_null_customer_id(self, spark):
        """customer_id が NULL のレコードは除去される"""
        df = self._make_order(spark, [
            Row(order_number=1, customer_id=None,  customer_name="Alice",
                order_datetime="1564627663", number_of_line_items="2",
                ordered_products=[], promo_info=[]),
            Row(order_number=2, customer_id="123", customer_name="Bob",
                order_datetime="1564627663", number_of_line_items="1",
                ordered_products=[], promo_info=[]),
        ])
        result = transform_sales_orders(df)
        assert result.count() == 1
        assert result.first().customer_id == "123"

    def test_removes_invalid_datetime(self, spark):
        """order_datetime が 0 以下のレコードは除去される"""
        df = self._make_order(spark, [
            Row(order_number=1, customer_id="111", customer_name="Alice",
                order_datetime="0",          number_of_line_items="2",
                ordered_products=[], promo_info=[]),
            Row(order_number=2, customer_id="222", customer_name="Bob",
                order_datetime="-1",         number_of_line_items="1",
                ordered_products=[], promo_info=[]),
            Row(order_number=3, customer_id="333", customer_name="Carol",
                order_datetime="1564627663", number_of_line_items="1",
                ordered_products=[], promo_info=[]),
        ])
        result = transform_sales_orders(df)
        assert result.count() == 1
        assert result.first().customer_id == "333"

    def test_removes_non_numeric_line_items(self, spark):
        """number_of_line_items が数値でないレコードは除去される"""
        df = self._make_order(spark, [
            Row(order_number=1, customer_id="111", customer_name="Alice",
                order_datetime="1564627663", number_of_line_items="abc",
                ordered_products=[], promo_info=[]),
            Row(order_number=2, customer_id="222", customer_name="Bob",
                order_datetime="1564627663", number_of_line_items="3",
                ordered_products=[], promo_info=[]),
        ])
        result = transform_sales_orders(df)
        assert result.count() == 1

    def test_deduplicates_by_order_number(self, spark):
        """同じ order_number の重複は1件に絞られる"""
        df = self._make_order(spark, [
            Row(order_number=99, customer_id="111", customer_name="Alice",
                order_datetime="1564627663", number_of_line_items="2",
                ordered_products=[], promo_info=[]),
            Row(order_number=99, customer_id="111", customer_name="Alice",
                order_datetime="1564627663", number_of_line_items="2",
                ordered_products=[], promo_info=[]),
        ])
        result = transform_sales_orders(df)
        assert result.count() == 1

    def test_trims_customer_name(self, spark):
        """customer_name の前後スペースが除去される"""
        df = self._make_order(spark, [
            Row(order_number=1, customer_id="111", customer_name="  Alice  ",
                order_datetime="1564627663", number_of_line_items="1",
                ordered_products=[], promo_info=[]),
        ])
        result = transform_sales_orders(df)
        assert result.first().customer_name == "Alice"


class TestTransformCustomers:

    def _base_row(self, **kwargs):
        defaults = dict(
            customer_id="111", tax_id=None, tax_code=None,
            customer_name="Alice", state="CA", city="LA",
            postcode=" 90001 ", street="Main St", number="1",
            unit=None, region="West", district="1",
            lon="-118.0", lat="34.0",
            ship_to_address="CA, 90001", valid_from="1532824233",
            valid_to=None, units_purchased="10.0", loyalty_segment="2",
        )
        defaults.update(kwargs)
        return Row(**defaults)

    def _make_customers(self, spark, rows):
        return spark.createDataFrame(rows, schema=CUSTOMER_SCHEMA)

    def test_adds_scd_columns(self, spark):
        """SCD Type 2 用カラムが追加される"""
        df = self._make_customers(spark, [self._base_row()])
        result = transform_customers(df)
        cols = result.columns
        assert "surrogate_key"        in cols
        assert "effective_start_date" in cols
        assert "effective_end_date"   in cols
        assert "is_current"           in cols

    def test_current_record_has_max_end_date(self, spark):
        """valid_to が NULL のレコードは is_current=True かつ effective_end_date が 9999-12-31"""
        df = self._make_customers(spark, [self._base_row(valid_to=None)])
        result = transform_customers(df)
        row = result.first()
        assert row.is_current is True
        assert row.effective_end_date.year == 9999

    def test_historical_record_is_not_current(self, spark):
        """valid_to がある（過去の）レコードは is_current=False"""
        df = self._make_customers(spark, [self._base_row(valid_to="1548137353.0")])
        result = transform_customers(df)
        assert result.first().is_current is False

    def test_removes_null_customer_id(self, spark):
        """customer_id が NULL のレコードは除去される"""
        df = self._make_customers(spark, [
            self._base_row(customer_id=None),
            self._base_row(customer_id="222"),
        ])
        result = transform_customers(df)
        assert result.count() == 1
