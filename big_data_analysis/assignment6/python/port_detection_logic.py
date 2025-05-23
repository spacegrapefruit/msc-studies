import logging
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    ArrayType,
    LongType,
)
from utils import EARTH_RADIUS_KM

# Parameters for DBSCAN
DBSCAN_EPS_KM = 1  # 1000 meters
DBSCAN_MIN_SAMPLES = 15
MIN_UNIQUE_VESSELS_PER_PORT = 5


def convert_km_to_radians(km):
    return km / EARTH_RADIUS_KM


def detect_ports_dbscan(filtered_df, spark_session):
    logging.info("Starting port detection using DBSCAN...")

    # Select necessary columns and collect to Pandas DataFrame on the driver
    logging.info(
        f"Collecting filtered data to driver for DBSCAN... (count: {filtered_df.count()})"
    )
    # Include MMSI for counting unique vessels per port later
    points_pd = filtered_df.select("Latitude", "Longitude", "MMSI").toPandas()

    if points_pd.empty:
        logging.warning("No data points to cluster. Returning empty DataFrame.")
        # Define an empty schema for consistency
        empty_port_schema = StructType(
            [
                StructField("port_id", IntegerType(), True),
                StructField("center_lat", DoubleType(), True),
                StructField("center_lon", DoubleType(), True),
                StructField("num_signals", IntegerType(), True),
                StructField("num_unique_vessels", IntegerType(), True),
                # Consider adding 'points_in_cluster' (Array of Structs with lat/lon)
                # or 'convex_hull_area' if needed for advanced sizing/visualization
            ]
        )
        return spark_session.createDataFrame([], schema=empty_port_schema)

    # Prepare data for DBSCAN: convert to radians for Haversine metric
    # scikit-learn's Haversine expects [latitude, longitude] in radians
    points_pd["Latitude_rad"] = np.radians(points_pd["Latitude"])
    points_pd["Longitude_rad"] = np.radians(points_pd["Longitude"])

    # Coordinates array for DBSCAN
    X = points_pd[["Latitude_rad", "Longitude_rad"]].values

    # Convert eps from km to radians
    eps_rad = convert_km_to_radians(DBSCAN_EPS_KM)
    logging.info(
        f"Running DBSCAN with eps={DBSCAN_EPS_KM}km ({eps_rad:.6f} radians) and min_samples={DBSCAN_MIN_SAMPLES}"
    )

    # Initialize and run DBSCAN
    db = DBSCAN(
        eps=eps_rad,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="haversine",
        algorithm="ball_tree",
    )
    db.fit(X)

    # Get cluster labels (-1 for noise)
    points_pd["cluster_label"] = db.labels_
    num_clusters_found = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    num_noise_points = (db.labels_ == -1).sum()
    logging.info(
        f"DBSCAN complete. Found {num_clusters_found} clusters and {num_noise_points} noise points."
    )

    # Filter out noise points
    clustered_points_pd = points_pd[points_pd["cluster_label"] != -1]

    if clustered_points_pd.empty:
        logging.warning(
            "No actual clusters formed (only noise). Returning empty DataFrame."
        )
        empty_port_schema = StructType(
            [
                StructField("port_id", IntegerType(), True),
                StructField("center_lat", DoubleType(), True),
                StructField("center_lon", DoubleType(), True),
                StructField("num_signals", IntegerType(), True),
                StructField("num_unique_vessels", IntegerType(), True),
            ]
        )
        return spark_session.createDataFrame([], schema=empty_port_schema)

    # Process each cluster to define a port
    port_summary_list = []
    skipped_clusters_count = 0
    for label in clustered_points_pd["cluster_label"].unique():
        port_id = int(label)  # DBSCAN labels are integers
        current_cluster_points = clustered_points_pd[
            clustered_points_pd["cluster_label"] == label
        ]

        center_lat = float(current_cluster_points["Latitude"].mean())
        center_lon = float(current_cluster_points["Longitude"].mean())
        num_signals = len(current_cluster_points)
        num_unique_vessels = current_cluster_points["MMSI"].nunique()

        if num_unique_vessels < MIN_UNIQUE_VESSELS_PER_PORT:
            skipped_clusters_count += 1
            continue  # Skip to the next cluster

        # We can store the points themselves if needed for visualization (e.g., convex hull).
        # For sizing, we'll rely on num_signals and num_unique_vessels.

        port_summary_list.append(
            {
                "port_id": port_id,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "num_signals": num_signals,
                "num_unique_vessels": num_unique_vessels,
                # "member_points": current_cluster_points[['Latitude', 'Longitude']].to_dict(orient='records') # Optional
            }
        )

    logging.info(
        f"Skipped {skipped_clusters_count} clusters due to not meeting unique vessel threshold."
    )

    # Define port DataFrame schema
    port_schema = StructType(
        [
            StructField("port_id", IntegerType(), True),
            StructField("center_lat", DoubleType(), True),
            StructField("center_lon", DoubleType(), True),
            StructField("num_signals", IntegerType(), True),
            StructField("num_unique_vessels", LongType(), True),
        ]
    )

    if port_summary_list:
        logging.info(
            f"Detected {len(port_summary_list)} distinct port clusters using DBSCAN."
        )
    else:
        logging.warning("No valid port summaries generated. Returning empty DataFrame.")

    final_ports_df = spark_session.createDataFrame(
        port_summary_list,
        schema=port_schema,
    )
    return final_ports_df
