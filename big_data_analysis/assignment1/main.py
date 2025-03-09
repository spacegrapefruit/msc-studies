import csv
import math
import logging
import itertools
import time
import multiprocessing as mp
import pandas as pd

CHUNK_SIZE = 1_000_000
N_PARSING_PROCESSES = mp.cpu_count() - 2
COLUMNS = ["# Timestamp", "MMSI", "Latitude", "Longitude", "SOG", "COG"]


class VesselState:
    def __init__(self, mmsi, timestamp, latitude, longitude, sog, cog):
        self.mmsi = mmsi
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.sog = sog
        self.cog = cog

    def update(self, timestamp, latitude, longitude, sog, cog):
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.sog = sog
        self.cog = cog


def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    r = 6_371_000  # radius of earth in meters
    return c * r


def parse_chunk(chunk_df, vessel_states):
    alerts = []

    for row in chunk_df.itertuples(index=False):
        if row.MMSI not in vessel_states:
            vessel_states[row.MMSI] = VesselState(
                timestamp=row.Timestamp,
                mmsi=row.MMSI,
                latitude=row.Latitude,
                longitude=row.Longitude,
                sog=row.SOG,
                cog=row.COG,
            )
            continue

        vessel_state = vessel_states[row.MMSI]
        time_diff = max((row.Timestamp - vessel_state.timestamp).total_seconds(), 1)

        distance = haversine_distance(
            vessel_state.latitude,
            vessel_state.longitude,
            row.Latitude,
            row.Longitude,
        )
        speed = distance / time_diff
        if speed > 200:
            logging.warning(
                f"Vessel {vessel_state.mmsi} has speed {speed:.2f} m/s at {row.Timestamp}"
            )
            alerts.append(
                {
                    "latitude": row.Latitude,
                    "longitude": row.Longitude,
                    "mmsi": vessel_state.mmsi,
                    "timestamp": row.Timestamp,
                    "reason": f"speed {speed} > 200 m/s",
                }
            )

        vessel_state.update(
            timestamp=row.Timestamp,
            latitude=row.Latitude,
            longitude=row.Longitude,
            sog=row.SOG,
            cog=row.COG,
        )

    logging.info(f"Parsed {len(chunk_df)} rows")
    return alerts


def reader_process(file_path, chunk_size, task_queues):
    for chunk_df in pd.read_csv(
        file_path,
        usecols=COLUMNS,
        parse_dates=["# Timestamp"],
        dayfirst=True,
        chunksize=chunk_size,
    ):
        chunk_df["worker_id"] = chunk_df["MMSI"] % 31337 % N_PARSING_PROCESSES
        chunk_df.rename(columns={"# Timestamp": "Timestamp"}, inplace=True)

        for worker_id, worker_df in chunk_df.groupby("worker_id"):
            task_queue = task_queues[worker_id]

            task_queue.put(worker_df.drop("worker_id", axis=1))
            logging.info(f"Read chunk, queue length: {task_queue.qsize()}")

    for task_queue in task_queues:
        task_queue.put(None)


def worker(task_queue, result_queue):
    vessel_states = {}
    while True:
        start_time = time.time()
        chunk_df = task_queue.get()
        end_time = time.time()
        if end_time - start_time > 0.1:
            logging.warning(f"Queue wait time: {end_time - start_time}")

        if chunk_df is None:  # Termination signal
            break
        alerts = parse_chunk(chunk_df, vessel_states)
        result_queue.put(alerts)


def main():
    file_path = "data/aisdk-2025-02-24.csv"

    task_queues = [mp.Queue(maxsize=10) for _ in range(N_PARSING_PROCESSES)]
    result_queue = mp.Queue()

    reader_proc = mp.Process(
        target=reader_process,
        args=(file_path, CHUNK_SIZE, task_queues),
    )

    workers = [
        mp.Process(
            target=worker,
            args=(task_queue, result_queue),
        )
        for task_queue in task_queues
    ]

    reader_proc.start()
    for w in workers:
        w.start()

    reader_proc.join()
    for w in workers:
        w.join()

    processed_alerts = []
    while not result_queue.empty():
        processed_alerts.extend(result_queue.get())
    logging.info(f"Total alerts: {len(processed_alerts)}")

    alerts_df = pd.DataFrame(processed_alerts)

    logging.info(f"Total rows processed: {len(processed_data)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(process)d-%(levelname)s: %(message)s",
    )
    main()
