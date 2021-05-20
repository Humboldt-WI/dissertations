from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame


def _create_spark_session() -> SparkSession:
    """
    Start spark session
    """
    spark = (
        SparkSession
        .builder
        .appName('engagement')
        .getOrCreate()
    )
    return spark


def _read_file(spark: SparkSession, filename: str) -> DataFrame:
    """
    Read parquet file and save it as spark df
    """
    df = spark.read.parquet('engagement/preparation/data/{}'.format(filename))
    return df
