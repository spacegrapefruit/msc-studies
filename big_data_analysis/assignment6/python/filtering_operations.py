import logging
from pyspark.sql.functions import col, to_timestamp

SPEED_THRESHOLD_KNOTS = 2.5  # slower vessels might be in port
MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0
TIMESTAMP_FORMAT = "dd/MM/yyyy HH:mm:ss"


def filter_and_prepare_data(df):
    """
    Filter and prepare AIS data for port detection.
    """
    logging.info("Starting data filtering and preparation...")

    # convert Timestamp
    df_ts = df.withColumn(
        "Timestamp", to_timestamp(col("Timestamp"), TIMESTAMP_FORMAT)
    ).na.drop(subset=["Timestamp"])  # drop rows where timestamp conversion failed
    logging.info(f"Rows after timestamp conversion and drop NA: {df_ts.count()}")

    # filter by valid coordinates
    df_valid_coords = df_ts.filter(
        (col("Latitude").isNotNull())
        & (col("Latitude") >= MIN_LATITUDE)
        & (col("Latitude") <= MAX_LATITUDE)
        & (col("Longitude").isNotNull())
        & (col("Longitude") >= MIN_LONGITUDE)
        & (col("Longitude") <= MAX_LONGITUDE)
    )
    logging.info(f"Rows after filtering invalid coordinates: {df_valid_coords.count()}")

    # filter by SOG
    df_slow_vessels = df_valid_coords.filter(col("SOG") <= SPEED_THRESHOLD_KNOTS)
    logging.info(
        f"Rows after SOG filter (<= {SPEED_THRESHOLD_KNOTS} knots): {df_slow_vessels.count()}"
    )

    # filter or prioritize by Navigational Status
    df_slow_stationary_vessels = df_slow_vessels.filter(
        (col("NavigationalStatus").isNotNull())
        & (col("NavigationalStatus").isin("Moored", "At anchor"))
    )
    logging.info(
        f"Rows after filtering by Navigational Status (Moored/At anchor): {df_slow_stationary_vessels.count()}"
    )

    # select necessary columns
    final_df = df_slow_stationary_vessels.select(
        "MMSI", "Latitude", "Longitude"
    ).dropDuplicates()

    logging.info(f"Rows after final selection and deduplication: {final_df.count()}")
    return final_df
