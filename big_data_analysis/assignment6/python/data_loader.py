import logging
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
    LongType,
)


def get_ais_schema():
    """
    Define the schema for AIS data.
    """
    return StructType(
        [
            StructField("Timestamp", StringType(), True),
            StructField("TypeOfMobile", StringType(), True),
            StructField("MMSI", LongType(), True),
            StructField("Latitude", DoubleType(), True),
            StructField("Longitude", DoubleType(), True),
            StructField("NavigationalStatus", StringType(), True),
            StructField("ROT", DoubleType(), True),  # Rate of Turn
            StructField("SOG", DoubleType(), True),  # Speed over Ground
        ]
    )


def load_data(spark, file_path, schema):
    """
    Load AIS data from a CSV file into a Spark DataFrame with the specified schema.
    """
    df = (
        spark.read.format("csv")
        .option("header", "false")
        .option("delimiter", ",")
        .option("comment", "#")  # skip original header
        .option("inferSchema", "false")  # use defined schema
        .schema(schema)
        .load(file_path)
    )

    logging.info(f"Successfully loaded data from {file_path}")
    logging.info(f"Initial row count: {df.count()}")
    return df
