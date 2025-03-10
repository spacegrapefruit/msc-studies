import math
import time
import logging
import multiprocessing as mp
from dataclasses import dataclass
import numpy as np
import pandas as pd

CHUNK_SIZE = 1_000_000
N_PARSING_PROCESSES = max(mp.cpu_count() - 2, 1)
COLUMNS = ["# Timestamp", "MMSI", "Latitude", "Longitude", "SOG", "COG"]


@dataclass
class VesselState:
    mmsi: int
    timestamp: pd.Timestamp
    latitude: float
    longitude: float
    sog: float
    cog: float

    def update(
        self,
        timestamp: pd.Timestamp,
        latitude: float,
        longitude: float,
        sog: float,
        cog: float,
    ) -> None:
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.sog = sog
        self.cog = cog


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
    earth_radius = 6_378_137  # in meters
    return c * earth_radius


def check_location_anomaly(distance: float, time_diff: float) -> bool:
    """Return True if the distance exceeds 150 km in under 1 minute."""
    return time_diff < 60 and distance > 150_000


def detect_duplicate_mmsi(
    df: pd.DataFrame,
    eps: float = 2000.0,
) -> pd.DataFrame:
    """
    Detect duplicate MMSI records within a given time window.

    Args:
        df: DataFrame containing columns ["Timestamp", "MMSI", "Latitude", "Longitude"].
        eps: The maximum distance for connecting the positions.

    Returns:
        TODO
    """
    suspicious_records = []
    for mmsi, group_df in df.groupby("MMSI"):
        if len(group_df) < 2:
            continue

        coords = list(zip(group_df["Latitude"], group_df["Longitude"]))
        last_locations = []
        for row in group_df.itertuples(index=False):
            latitude, longitude = row.Latitude, row.Longitude
            for idx, last_location in enumerate(last_locations):
                dist = haversine_distance(
                    latitude, longitude, last_location[0], last_location[1]
                )
                if dist <= eps:
                    last_locations[idx] = (latitude, longitude)
                    break
            else:
                last_locations.append((latitude, longitude))

        if len(last_locations) > 1:
            logging.warning(f"Found {len(last_locations)} locations for MMSI {mmsi}")

            suspicious_records.append(
                {
                    "latitude": None,
                    "longitude": None,
                    "mmsi": mmsi,
                    "timestamp": None,
                    "reason": f"Suspected {len(last_locations)} ships under one MMSI",
                }
            )

    return suspicious_records


def parse_chunk(chunk_df: pd.DataFrame, vessel_states: dict) -> list[dict]:
    """Process a chunk of data, updating vessel states and collecting alerts."""
    alerts = []
    suspicious_mmsi = set()

    for row in chunk_df.itertuples(index=False):
        mmsi = row.MMSI
        if mmsi not in vessel_states:
            vessel_states[mmsi] = VesselState(
                mmsi=mmsi,
                timestamp=row.Timestamp,
                latitude=row.Latitude,
                longitude=row.Longitude,
                sog=row.SOG,
                cog=row.COG,
            )
            continue

        state = vessel_states[mmsi]
        time_diff = max((row.Timestamp - state.timestamp).total_seconds(), 1)
        distance = haversine_distance(
            state.latitude, state.longitude, row.Latitude, row.Longitude
        )
        speed = distance / time_diff

        if check_location_anomaly(distance, time_diff):
            suspicious_mmsi.add(mmsi)
            logging.warning(f"Location anomaly: {distance:.2f}m in {time_diff:.2f}s")
            alerts.append(
                {
                    "latitude": row.Latitude,
                    "longitude": row.Longitude,
                    "mmsi": mmsi,
                    "timestamp": row.Timestamp,
                    "reason": f"Location jump of {distance:.2f}m in {time_diff:.2f}s",
                }
            )

        if speed > 200:
            suspicious_mmsi.add(mmsi)
            logging.warning(f"Speed anomaly: {speed:.2f} m/s")
            alerts.append(
                {
                    "latitude": row.Latitude,
                    "longitude": row.Longitude,
                    "mmsi": mmsi,
                    "timestamp": row.Timestamp,
                    "reason": f"Speed {speed:.2f} m/s > 200 m/s",
                }
            )

        state.update(row.Timestamp, row.Latitude, row.Longitude, row.SOG, row.COG)

    # Check for duplicate MMSI entries within a 10-minute window.
    duplicate_alerts = detect_duplicate_mmsi(
        df=chunk_df[chunk_df["MMSI"].isin(suspicious_mmsi)],
        eps=2000.0,
    )
    alerts.extend(duplicate_alerts)
    return alerts


def reader_process(
    file_path: str, chunk_size: int, task_queues: list[mp.Queue]
) -> None:
    """Read CSV data in chunks and distribute tasks to worker queues."""
    for chunk in pd.read_csv(
        file_path,
        usecols=COLUMNS,
        parse_dates=["# Timestamp"],
        dayfirst=True,
        chunksize=chunk_size,
        nrows=CHUNK_SIZE * 2,
    ):
        chunk.rename(columns={"# Timestamp": "Timestamp"}, inplace=True)
        # Assign a worker based on a hash of MMSI.
        chunk["worker_id"] = chunk["MMSI"] % 31337 % N_PARSING_PROCESSES

        for worker_id, group in chunk.groupby("worker_id"):
            queue = task_queues[worker_id]
            queue.put(group.drop("worker_id", axis=1))
            logging.info(
                f"Queued {len(group)} rows to worker {worker_id} (queue size: {queue.qsize()})"
            )

    # Signal workers to exit.
    for queue in task_queues:
        queue.put(None)


def worker_process(task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Worker process to parse data chunks and report alerts."""
    vessel_states = {}
    while True:
        start_time = time.time()
        chunk = task_queue.get()
        wait_time = time.time() - start_time
        if wait_time > 0.1:
            logging.warning(f"Queue wait time: {wait_time:.2f}s")

        if chunk is None:
            result_queue.put(None)
            break

        alerts = parse_chunk(chunk, vessel_states)
        result_queue.put(alerts)


def main() -> None:
    file_path = "data/aisdk-2025-02-24.csv"
    task_queues = [mp.Queue(maxsize=10) for _ in range(N_PARSING_PROCESSES)]
    result_queue = mp.Queue()

    reader = mp.Process(
        target=reader_process, args=(file_path, CHUNK_SIZE, task_queues)
    )
    workers = [
        mp.Process(target=worker_process, args=(queue, result_queue))
        for queue in task_queues
    ]

    reader.start()
    for w in workers:
        w.start()

    reader.join()

    all_alerts = []
    alive_workers = len(workers)
    while alive_workers:
        result = result_queue.get()
        if result is None:
            alive_workers -= 1
        else:
            all_alerts.extend(result)

    logging.info(f"Total alerts detected: {len(all_alerts)}")
    alerts_df = pd.DataFrame(all_alerts)

    for w in workers:
        w.join()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(process)d-%(levelname)s: %(message)s",
    )
    main()
