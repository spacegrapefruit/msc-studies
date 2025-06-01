import logging
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from pyspark.sql.functions import col, log1p, min, max, greatest, lit
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    ArrayType,
)

EARTH_RADIUS_KM = 6_371


def convert_km_to_radians(km):
    return km / EARTH_RADIUS_KM


def detect_ports_dbscan(filtered_df, spark_session, config):
    """
    Detect ports using DBSCAN clustering algorithm on filtered AIS data.
    """
    eps_km = config["eps_km"]
    min_samples = config["min_samples"]
    min_vessels_per_port = config["min_vessels_per_port"]

    # collect to Pandas
    logging.info(
        f"Collecting filtered data for DBSCAN... (count: {filtered_df.count()})"
    )
    # include MMSI for counting unique vessels per port later
    points_pd = filtered_df.select("Latitude", "Longitude", "MMSI").toPandas()

    # convert to radians for Haversine metric
    # scikit-learn's Haversine expects [latitude, longitude] in radians
    points_pd["Latitude_rad"] = np.radians(points_pd["Latitude"])
    points_pd["Longitude_rad"] = np.radians(points_pd["Longitude"])

    # coordinates array for DBSCAN
    X = points_pd[["Latitude_rad", "Longitude_rad"]].values

    # convert eps from km to radians
    eps_rad = convert_km_to_radians(eps_km)
    logging.info(
        f"Running DBSCAN with eps={eps_km}km ({eps_rad:.6f} radians) and min_samples={min_samples}"
    )

    # run DBSCAN
    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    db.fit(X)

    # get cluster labels (-1 for noise)
    points_pd["cluster_label"] = db.labels_
    num_clusters_found = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    num_noise_points = (db.labels_ == -1).sum()
    logging.info(
        f"DBSCAN complete. Found {num_clusters_found} clusters and {num_noise_points} noise points."
    )

    # filter out noise points
    clustered_points_pd = points_pd[points_pd["cluster_label"] != -1]

    # process each cluster to define a port
    port_summary_list = []
    skipped_clusters_count = 0
    for label in clustered_points_pd["cluster_label"].unique():
        port_id = int(label)
        current_cluster_points = clustered_points_pd[
            clustered_points_pd["cluster_label"] == label
        ]

        center_lat = float(current_cluster_points["Latitude"].mean())
        center_lon = float(current_cluster_points["Longitude"].mean())
        num_signals = len(current_cluster_points)
        num_unique_vessels = current_cluster_points["MMSI"].nunique()

        # skip clusters under the minimum unique vessel threshold
        if num_unique_vessels < min_vessels_per_port:
            skipped_clusters_count += 1
            continue

        # select member points for the port, round to deduplicate somewhat
        # these will be used for visualization and convex hull creation
        member_points = (
            current_cluster_points[["Latitude", "Longitude"]].round(4).drop_duplicates()
        )

        # for port sizing, we'll rely on num_unique_vessels
        port_summary_list.append(
            {
                "port_id": port_id,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "num_signals": num_signals,
                "num_unique_vessels": num_unique_vessels,
                "member_points": member_points.to_dict(orient="records"),
            }
        )

    logging.info(
        f"Skipped {skipped_clusters_count} clusters due to not meeting unique vessel threshold."
    )

    # define port DataFrame schema
    port_schema = StructType(
        [
            StructField("port_id", IntegerType(), True),
            StructField("center_lat", DoubleType(), True),
            StructField("center_lon", DoubleType(), True),
            StructField("num_signals", IntegerType(), True),
            StructField("num_unique_vessels", IntegerType(), True),
            StructField(
                "member_points",
                ArrayType(
                    StructType(
                        [
                            StructField("Latitude", DoubleType(), True),
                            StructField("Longitude", DoubleType(), True),
                        ]
                    )
                ),
                True,
            ),
        ]
    )

    logging.info(
        f"Detected {len(port_summary_list)} distinct port clusters using DBSCAN."
    )

    final_ports_df = spark_session.createDataFrame(
        port_summary_list,
        schema=port_schema,
    )
    return final_ports_df


def evaluate_relative_port_size(ports_df):
    """
    Evaluate relative port sizes based on the number of unique vessels.
    """
    if ports_df.count() == 0:
        logging.warning("No ports to evaluate for size.")
        return ports_df

    logging.info("Evaluating relative port sizes...")

    min_max_vessels = ports_df.select(
        min("num_unique_vessels").alias("min_v"),
        max("num_unique_vessels").alias("max_v"),
    ).first()
    min_v, max_v = (min_max_vessels["min_v"], min_max_vessels["max_v"])

    ports_with_size = ports_df.withColumn(
        "rel_size_vessels_log", log1p(col("num_unique_vessels"))
    )

    min_max_log_v_row = ports_with_size.select(
        min("rel_size_vessels_log").alias("min_val"),
        max("rel_size_vessels_log").alias("max_val"),
    ).first()
    min_log_v, max_log_v = (
        min_max_log_v_row["min_val"],
        min_max_log_v_row["max_val"],
    )
    denominator_v = max_log_v - min_log_v
    ports_with_size = ports_with_size.withColumn(
        "norm_size_vessels",
        (col("rel_size_vessels_log") - lit(min_log_v))
        / greatest(lit(1e-6), lit(denominator_v))
        if denominator_v > 1e-6
        else lit(0.5),
    )

    logging.info(f"Added relative size metrics. Total ports: {ports_with_size.count()}")
    return ports_with_size
