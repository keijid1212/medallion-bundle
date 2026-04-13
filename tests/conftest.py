import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """テスト用 SparkSession（ローカルモード）"""
    return (
        SparkSession.builder
        .master("local[1]")
        .appName("medallion_test")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "1")
        .getOrCreate()
    )
