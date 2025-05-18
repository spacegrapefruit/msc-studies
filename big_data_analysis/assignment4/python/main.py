import logging
import math
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    greatest,
    lag,
    lit,
    udf,
    sum as spark_sum,
    to_timestamp,
    desc,
    when,
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, TimestampType

from config import Config

EARTH_RADIUS = 6_378  # in kilometers
TIMESTAMP_FORMAT = "dd/MM/yyyy HH:mm:ss"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Haversine distance between two (lat, lon) points in kilometers."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance_km = c * EARTH_RADIUS
    return distance_km


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # load configuration
    config_path = Path(__file__).parent.parent / "config.yml"
    config = Config(config_path, "main")

    # initialize Spark
    spark = (
        SparkSession.builder.appName("LongestVesselRoute")
        # give it 4GB of memory
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    logging.info("Spark session started.")

    # load CSV data
    df = spark.read.csv(
        config.input_file,
        header=True,
        inferSchema=True,
    )
    logging.info(f"Data loaded from {config.input_file}, total records: {df.count()}")

    # rename Timestamp, ensure correct column types
    df = (
        df.withColumnRenamed("# Timestamp", "Timestamp")
        .withColumn("Timestamp", to_timestamp(col("Timestamp"), TIMESTAMP_FORMAT))
        .withColumn("Latitude", col("Latitude").cast(DoubleType()))
        .withColumn("Longitude", col("Longitude").cast(DoubleType()))
        .withColumn("MMSI", col("MMSI").cast("string"))
        # filter out invalid coordinates
        .filter(
            (col("Latitude").between(-90, 90)) & (col("Longitude").between(-180, 180))
        )
    )
    logging.info("Data types corrected and invalid coordinates filtered.")

    # define a UDF for haversine distance
    haversine_udf = udf(haversine_distance, DoubleType())

    # window to get previous point per vessel ordered by time
    vessel_window = Window.partitionBy("MMSI").orderBy("Timestamp")

    # phase 1: ban vessels with excessive speed
    logging.info("Identifying vessels with speeds over 200 m/s...")
    df_with_prev_data = (
        df.withColumn("prev_lat", lag("Latitude").over(vessel_window))
        .withColumn("prev_lon", lag("Longitude").over(vessel_window))
        .withColumn("prev_timestamp", lag("Timestamp").over(vessel_window))
    )

    # calculate distance segments and time difference
    df_segments_for_speed_check = df_with_prev_data.withColumn(
        "segment_km_check",
        haversine_udf(
            col("prev_lat"), col("prev_lon"), col("Latitude"), col("Longitude")
        ),
    ).withColumn(
        "time_diff_seconds",
        col("Timestamp").cast("long") - col("prev_timestamp").cast("long"),
    )

    # condition for banning: speed > 200 m/s
    condition_speed_ban = (
        col("segment_km_check").isNotNull()
        & col("time_diff_seconds").isNotNull()
        & (
            (
                (col("segment_km_check") * 1000)
                # max(time_diff, 1 second)
                / greatest(col("time_diff_seconds"), lit(1))
            )
            > 200.0
        )
    )

    # filter vessels to ban
    banned_mmsi_df = (
        df_segments_for_speed_check.filter(condition_speed_ban)
        .select("MMSI")
        .distinct()
    )

    banned_count = banned_mmsi_df.count()
    if banned_count > 0:
        logging.info(f"Found {banned_count} vessels to ban due to excessive speed.")
    else:
        logging.info("No vessels found exceeding the speed limit of 200 m/s.")

    # phase 2: process valid vessels
    df_valid_vessels = df.join(banned_mmsi_df, "MMSI", "left_anti")

    original_record_count = df.count()
    valid_record_count = df_valid_vessels.count()
    logging.info(
        f"Original records: {original_record_count}. Records after removing banned vessels: {valid_record_count}."
    )

    if valid_record_count == 0:
        logging.warning("No valid vessel data remains after filtering. Exiting.")
        spark.stop()
        exit(0)

    # calculate distance segments for valid vessels
    df_dist = (
        df_valid_vessels.withColumn("prev_lat", lag("Latitude").over(vessel_window))
        .withColumn("prev_lon", lag("Longitude").over(vessel_window))
        .withColumn(
            "segment_km",
            haversine_udf(
                col("prev_lat"), col("prev_lon"), col("Latitude"), col("Longitude")
            ),
        )
    )
    logging.info("Distance segments calculated for valid vessels.")

    # sum distances per vessel
    total_dist = (
        df_dist.na.fill({"segment_km": 0.0})  # Fill first row with 0.0
        .groupBy("MMSI")
        .agg(spark_sum("segment_km").alias("total_km"))
    )

    # find the vessels with the maximum total distance
    longest = total_dist.orderBy(desc("total_km")).limit(5).collect()

    logging.info("Top Vessels with longest routes (excluding speed-banned vessels):")
    for i, row in enumerate(longest):
        mmsi, dist = row["MMSI"], row["total_km"]
        logging.info(f"  Rank {i + 1}: MMSI={mmsi}, Distance={dist:.2f} km")

    spark.stop()
    logging.info("Spark session stopped.")
