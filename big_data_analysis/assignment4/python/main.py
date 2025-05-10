import logging
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, udf, sum as spark_sum, desc
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, TimestampType

from config import Config
# from utils import haversine_distance


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Compute the great-circle distance between two points
    on the Earth specified in decimal degrees.
    Returns distance in kilometers.
    """
    if lon1 is None or lat1 is None or lon2 is None or lat2 is None:
        return None

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # load configuration
    config_path = Path(__file__).parent.parent / "config.yml"
    config = Config(config_path, "main")

    # Initialize Spark
    spark = SparkSession.builder.appName("LongestVesselRoute").getOrCreate()
    logging.info("Spark session started.")

    # Load CSV (adjust header/inferSchema as necessary)
    df = spark.read.csv(config.input_file, header=True, inferSchema=True)
    logging.info(f"Data loaded from {config.input_file}, total records: {df.count()}")

    # Rename Timestamp, ensure correct column types
    df = (
        df.withColumnRenamed("# Timestamp", "Timestamp")
        .withColumn("Timestamp", col("Timestamp").cast(TimestampType()))
        .withColumn("Latitude", col("Latitude").cast(DoubleType()))
        .withColumn("Longitude", col("Longitude").cast(DoubleType()))
    )
    logging.info("Data types corrected.")

    # Define a UDF for haversine distance
    haversine_udf = udf(haversine_distance, DoubleType())

    # Window to get previous point per vessel ordered by time
    vessel_window = Window.partitionBy("MMSI").orderBy("Timestamp")

    df_dist = (
        df.withColumn("prev_lat", lag("Latitude").over(vessel_window))
        .withColumn("prev_lon", lag("Longitude").over(vessel_window))
        .withColumn(
            "segment_km",
            haversine_udf(
                col("prev_lon"), col("prev_lat"), col("Longitude"), col("Latitude")
            ),
        )
    )
    logging.info("Distance segments calculated.")

    # Sum distances per vessel (skip first null segment)
    total_dist = (
        df_dist.na.fill({"segment_km": 0.0})
        .groupBy("mmsi")
        .agg(spark_sum("segment_km").alias("total_km"))
    )

    # Find the vessel with the maximum total distance
    longest = total_dist.orderBy(desc("total_km")).limit(1).collect()

    if longest:
        mmsi, dist = longest[0]["mmsi"], longest[0]["total_km"]
        print(
            f"Vessel with longest route on 2024-05-04: MMSI={mmsi}, Distance={dist:.2f} km"
        )
    else:
        print("No data found for that date.")

    spark.stop()
