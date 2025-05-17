import logging
import math
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, udf, sum as spark_sum, desc
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, TimestampType

from config import Config

EARTH_RADIUS = 6_378  # in kilometers


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Haversine distance between two (lat, lon) points in kilometers."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None

    raw_lat1 = lat1
    raw_lon1 = lon1
    raw_lat2 = lat2
    raw_lon2 = lon2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    if c * EARTH_RADIUS > 2000:
        logging.warning(
            f"Distance too large: {c * EARTH_RADIUS:.2f} km, "
            f"lat1={raw_lat1}, lon1={raw_lon1}, lat2={raw_lat2}, lon2={raw_lon2}"
        )
    return c * EARTH_RADIUS


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

    # Load CSV data
    df = spark.read.csv(
        config.input_file,
        header=True,
        inferSchema=True,
    )
    logging.info(f"Data loaded from {config.input_file}, total records: {df.count()}")

    # Rename Timestamp, ensure correct column types
    df = (
        df.withColumnRenamed("# Timestamp", "Timestamp")
        .withColumn("Timestamp", col("Timestamp").cast(TimestampType()))
        .withColumn("Latitude", col("Latitude").cast(DoubleType()))
        .withColumn("Longitude", col("Longitude").cast(DoubleType()))
        # remove invalid coordinates
        .filter((col("Latitude").between(-90, 90)) & (col("Longitude").between(-180, 180)))
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
                col("prev_lat"), col("prev_lon"), col("Latitude"), col("Longitude")
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
    longest = total_dist.orderBy(desc("total_km")).limit(5).collect()

    if longest:
        logging.info("Vessels with longest routes on 2024-05-04:")
        for i, row in enumerate(longest):
            mmsi, dist = row["mmsi"], row["total_km"]
            logging.info(f"  Rank {i + 1}: MMSI={mmsi}, Distance={dist:.2f} km")
    else:
        logging.warning("No data found for that date.")

    spark.stop()
