import math
import time
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
from collections import namedtuple
from dataclasses import dataclass

CHUNK_SIZE = 100_000
N_PARSING_PROCESSES = max(mp.cpu_count() - 1, 1)
COLUMNS = ["# Timestamp", "MMSI", "Latitude", "Longitude"]


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
    earth_radius = 6_378_137  # in meters
    return c * earth_radius


def parse_chunk(
    chunk_df: pd.DataFrame, vessel_states: dict, suspicious_mmsi: set
) -> list[dict]:
    """Process a chunk of data, updating vessel states and collecting alerts."""
    alerts = []

    for row in chunk_df.itertuples(index=False):
        mmsi = row.MMSI
        gps_point = GpsPoint(
            timestamp=row.Timestamp,
            latitude=row.Latitude,
            longitude=row.Longitude,
        )

        if mmsi not in vessel_states:
            vessel_states[mmsi] = VesselState(gps_point)
            continue

        point_delta = gps_point - vessel_states[mmsi].last_point
        speed = point_delta.distance / max(point_delta.time_delta, 1)

        if point_delta.distance > 20_000:
            alert = {
                "mmsi": mmsi,
                "latitude": row.Latitude,
                "longitude": row.Longitude,
                "timestamp": row.Timestamp,
                "reason": f"Location jump of {point_delta.distance:.2f}m",
            }
            logging.warning(alert["reason"])
            alerts.append(alert)

            suspicious_mmsi.add(mmsi)

        if speed > 200:
            alert = {
                "mmsi": mmsi,
                "latitude": row.Latitude,
                "longitude": row.Longitude,
                "timestamp": row.Timestamp,
                "reason": f"Speed {speed:.2f} m/s > 200 m/s",
            }
            logging.warning(alert["reason"])
            alerts.append(alert)

        vessel_states[mmsi].update(gps_point)

    return alerts


def reader_process(file_path: str, task_queues: list[mp.Queue]) -> None:
    """Read CSV data in chunks and distribute tasks to worker queues."""
    for chunk_df in pd.read_csv(
        file_path,
        usecols=COLUMNS,
        parse_dates=["# Timestamp"],
        dayfirst=True,
        chunksize=CHUNK_SIZE,
        # nrows=CHUNK_SIZE * 5,
    ):
        chunk_df.rename(columns={"# Timestamp": "Timestamp"}, inplace=True)
        # assign worker based on a deterministic hash of MMSI
        chunk_df["worker_id"] = chunk_df["MMSI"] % 31337 % N_PARSING_PROCESSES

        for worker_id, group_df in chunk_df.groupby("worker_id"):
            queue = task_queues[worker_id]
            queue.put(group_df.drop("worker_id", axis=1))
            logging.debug(
                f"Queued {len(group_df)} rows to worker {worker_id} (queue size: {queue.qsize()})"
            )

    # Signal workers to exit.
    for queue in task_queues:
        queue.put(None)


def worker_process(task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Worker process to parse data chunks and report alerts."""
    vessel_states = {}
    suspicious_mmsi = set()
    while True:
        start_time = time.time()
        chunk_df = task_queue.get()
        wait_time = time.time() - start_time

        if wait_time > 0.5:
            logging.warning(f"Task queue wait time: {wait_time:.2f}s")

        if chunk_df is None:
            alerts = []
            for mmsi, state in vessel_states.items():
                if len(state.possible_locations) > 1:
                    alert = {
                        "mmsi": mmsi,
                        "latitude": None,
                        "longitude": None,
                        "timestamp": None,
                        "reason": f"Suspected {len(state.possible_locations)} ships under one MMSI",
                    }
                    logging.warning(alert["reason"])

            result_queue.put(alerts)
            result_queue.put(None)
            break

        alerts = parse_chunk(chunk_df, vessel_states, suspicious_mmsi)
        result_queue.put(alerts)


def main() -> None:
    file_path = "data/aisdk-2025-02-24.csv"
    task_queues = [mp.Queue(maxsize=10) for _ in range(N_PARSING_PROCESSES)]
    result_queue = mp.Queue()

    reader = mp.Process(target=reader_process, args=(file_path, task_queues))
    workers = [
        mp.Process(target=worker_process, args=(task_queue, result_queue))
        for task_queue in task_queues
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
