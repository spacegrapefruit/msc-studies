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
    return StructType(
        [
            StructField(
                "Timestamp", StringType(), True
            ),  # Or TimestampType if directly parsable
            StructField("TypeOfMobile", StringType(), True),
            StructField("MMSI", LongType(), True),
            StructField("Latitude", DoubleType(), True),
            StructField("Longitude", DoubleType(), True),
            StructField("NavigationalStatus", StringType(), True),
            StructField("ROT", DoubleType(), True),  # Rate of Turn
            StructField("SOG", DoubleType(), True),  # Speed Over Ground
            StructField("COG", DoubleType(), True),  # Course Over Ground
            StructField("Heading", LongType(), True),
            # Add more fields if available and needed, e.g., IMO, Name, ShipType
            StructField("IMO", StringType(), True),
            StructField("CallSign", StringType(), True),
            StructField("Name", StringType(), True),
            StructField("ShipType", StringType(), True),
            StructField("CargoType", StringType(), True),
            StructField("Width", StringType(), True),
            StructField("Length", StringType(), True),
            StructField("TypeOfPositionFixingDevice", StringType(), True),
            StructField("Draught", StringType(), True),
            StructField("Destination", StringType(), True),
            StructField("ETA", StringType(), True),
            StructField("DataSourceType", StringType(), True),
            StructField("A", StringType(), True),
            StructField("B", StringType(), True),
            StructField("C", StringType(), True),
            StructField("D", StringType(), True),
        ]
    )


def load_data(spark, file_path, schema):
    df = (
        spark.read.format("csv")
        .option("header", "false")
        .option("delimiter", ",")
        .option("comment", "#")  # skip original header
        .option(
            "inferSchema", "false"
        )  # Use defined schema for performance and correctness
        .schema(schema)
        .load(file_path)
    )

    logging.info(f"Successfully loaded data from {file_path}")
    logging.info(f"Initial row count: {df.count()}")
    return df
