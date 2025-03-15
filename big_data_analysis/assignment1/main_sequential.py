import argparse
import math
import time
import logging
import numpy as np
import pandas as pd
from collections import namedtuple
from dataclasses import dataclass
from sklearn.cluster import DBSCAN

EARTH_RADIUS = 6_378_137  # in meters
MIN_JAMMING_CLUSTER_SIZE = 4


parser = argparse.ArgumentParser(
    description="AIS data analysis for GPS jamming detection"
)
parser.add_argument(
    "--file-path",
    type=str,
    default="data/aisdk-2025-02-24.csv",
    help="Path to AIS data CSV file",
)
parser.add_argument(
    "--log-level",
    type=str,
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    default="INFO",
    help="Set logging level",
)
parser.add_argument(
    "--chunk-size",
    type=int,
    default=100_000,
    help="Size of data chunks for processing",
)


GpsPointDelta = namedtuple("GpsPointDelta", ["distance", "time_delta"])


@dataclass
class GpsPoint:
    timestamp: pd.Timestamp
    latitude: float
    longitude: float

    def __sub__(self, other: "GpsPoint") -> float:
        distance = haversine_distance(
            self.latitude, self.longitude, other.latitude, other.longitude
        )
        time_delta = abs((self.timestamp - other.timestamp).total_seconds())

        return GpsPointDelta(distance=distance, time_delta=time_delta)


class VesselState:
    def __init__(self, gps_point: GpsPoint):
        self.last_point = gps_point
        self.possible_locations = [gps_point]

    def update(self, gps_point: GpsPoint):
        self.last_point = gps_point

        for idx, other_location in enumerate(self.possible_locations):
            if (gps_point - other_location).distance <= 20_000:
                self.possible_locations[idx] = gps_point
                break
        else:
            self.possible_locations.append(gps_point)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Haversine distance between two (lat, lon) points in meters."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return c * EARTH_RADIUS


def parse_chunk(chunk_df: pd.DataFrame, vessel_states: dict) -> list[dict]:
    """Process a chunk of data, updating vessel states and collecting alerts."""
    flagged_coordinates = []

    for row in chunk_df.itertuples(index=False):
        # skip invalid coordinates
        if abs(row.Latitude) > 90 or abs(row.Longitude) > 180:
            continue

        mmsi = row.MMSI
        gps_point = GpsPoint(
            timestamp=row.Timestamp,
            latitude=row.Latitude,
            longitude=row.Longitude,
        )

        if mmsi not in vessel_states:
            vessel_states[mmsi] = VesselState(gps_point)
            continue

        gps_delta = gps_point - vessel_states[mmsi].last_point
        speed = gps_delta.distance / max(gps_delta.time_delta, 1)

        flag_coordinates = False
        if gps_delta.distance > 20_000:
            flag_coordinates = True
            logging.warning(f"Location jump of {gps_delta.distance:.2f}m")

        if speed > 200:
            flag_coordinates = True
            logging.warning(f"Speed {speed:.2f} m/s > 200 m/s")

        # saving both before and after coordinates
        if flag_coordinates:
            coordinates = {
                "mmsi": mmsi,
                "gps_point": vessel_states[mmsi].last_point,
            }
            flagged_coordinates.append(coordinates)
            coordinates = {
                "mmsi": mmsi,
                "gps_point": gps_point,
            }
            flagged_coordinates.append(coordinates)

        vessel_states[mmsi].update(gps_point)

    return flagged_coordinates


def reader_process(file_path: str, chunk_size: int) -> list[pd.DataFrame]:
    """Read CSV data in chunks."""
    for chunk_df in pd.read_csv(
        file_path,
        usecols=["# Timestamp", "MMSI", "Latitude", "Longitude"],
        parse_dates=["# Timestamp"],
        dayfirst=True,
        chunksize=chunk_size,
    ):
        chunk_df.rename(columns={"# Timestamp": "Timestamp"}, inplace=True)

        # assert sorted by Timestamp
        assert chunk_df.Timestamp.is_monotonic_increasing

        yield chunk_df


def worker_process(chunk_df, vessel_states) -> None:
    """Worker process to parse data chunks and report alerts."""
    alerts = parse_chunk(chunk_df, vessel_states)
    return alerts


def find_cluster_centroids(
    distance_matrix,
    slice_positions_df,
    radius=50_000,
    min_cluster_size=MIN_JAMMING_CLUSTER_SIZE,
):
    # perform DBSCAN clustering
    db = DBSCAN(eps=radius, min_samples=min_cluster_size, metric="precomputed")
    labels = db.fit_predict(distance_matrix)

    clusters = []
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue  # ignore noise points
        cluster_indices = np.where(labels == label)[0]

        mmsis = sorted(set(slice_positions_df["mmsi"][cluster_indices]))
        if len(mmsis) < min_cluster_size:
            continue  # skip clusters with < min_cluster_size MMSI

        # compute the centroid as the mean of the coordinates in the cluster
        cluster_coordinates = np.array(
            slice_positions_df["coordinates"][cluster_indices]
        )
        centroid = np.mean(cluster_coordinates, axis=0)
        clusters.append((centroid, label, mmsis))

    return clusters


def main(args: argparse.Namespace) -> None:
    file_path = args.file_path

    all_alerts = []
    vessel_states = {}
    for chunk_df in reader_process(args.file_path, args.chunk_size):
        all_alerts += worker_process(chunk_df, vessel_states)
    for mmsi, state in vessel_states.items():
        if len(state.possible_locations) > 1:
            logging.warning(
                f"Suspected {len(state.possible_locations)} ships under one MMSI"
            )

    logging.info(f"Total suspicious movements: {len(all_alerts)}")
    alerts_df = pd.DataFrame(all_alerts)
    alerts_df["timestamp"] = alerts_df["gps_point"].apply(lambda x: x.timestamp)
    alerts_df.sort_values("timestamp", inplace=True)
    alerts_df["time_slice"] = alerts_df["timestamp"].dt.floor("15min")

    # every 15 minutes, check for suspicious clusters of anomalies
    for time_slice, group_df in alerts_df.groupby("time_slice"):
        vessel_states = {}
        for row in group_df.itertuples(index=False):
            mmsi = row.mmsi
            gps_point = row.gps_point

            if mmsi not in vessel_states:
                vessel_states[mmsi] = VesselState(gps_point)
            else:
                vessel_states[mmsi].update(gps_point)

        # construct final positions and mmsis within this slice
        slice_positions_df = pd.DataFrame(
            [
                {"mmsi": mmsi, "coordinates": (gps_point.latitude, gps_point.longitude)}
                for mmsi, state in vessel_states.items()
                for gps_point in state.possible_locations
            ]
        )

        n_points = len(slice_positions_df)
        if n_points < MIN_JAMMING_CLUSTER_SIZE:
            continue

        distance_matrix = np.zeros((n_points, n_points), dtype=int)
        for i, p1 in enumerate(slice_positions_df["coordinates"]):
            for j, p2 in enumerate(slice_positions_df["coordinates"]):
                if i == j:
                    break
                dist = haversine_distance(*p1, *p2)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        clusters = find_cluster_centroids(distance_matrix, slice_positions_df)
        if clusters:
            logging.warning(f"Suspected jamming at {time_slice}:")
        for centroid, label, mmsis in clusters:
            logging.warning(
                f"  Incident #{label}: ({centroid[0]:.5f}, {centroid[1]:.5f}), {len(mmsis)} MMSIS"
            )


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(process)d-%(levelname)s: %(message)s",
    )
    main(args)
