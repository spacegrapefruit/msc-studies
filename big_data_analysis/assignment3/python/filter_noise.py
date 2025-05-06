import logging
import threading

from pathlib import Path
from pymongo import MongoClient, ASCENDING
from queue import Queue
from tqdm import tqdm

from config import Config


REQUIRED_FIELDS = [
    "NavigationalStatus",
    "MMSI",
    "Latitude",
    "Longitude",
    "ROT",
    "SOG",
    "COG",
    "Heading",
]
MIN_POINTS = 100


def worker(q: Queue, worker_id: int, config: Config):
    """
    Worker function to process batches of data and insert them into MongoDB.
    """
    client = MongoClient(config.mongo_uri, connectTimeoutMS=1000, socketTimeoutMS=1000)
    logging.info(f"Worker {worker_id} connected to MongoDB")

    raw_coll = client[config.db_name][config.raw_collection]
    clean_coll = client[config.db_name][config.clean_collection]
    while True:
        mmsi = q.get()
        if mmsi is None:
            break
        docs = list(raw_coll.find({"MMSI": mmsi}))
        if len(docs) < MIN_POINTS:
            q.task_done()
            continue
        filtered = [
            d for d in docs if all(d.get(f) not in (None, "") for f in REQUIRED_FIELDS)
        ]
        if filtered:
            clean_coll.insert_many(filtered)
        q.task_done()
    client.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # load configuration
    config_path = Path(__file__).parent.parent / "config.yml"
    config = Config(config_path, "filter_noise")

    client = MongoClient(config.mongo_uri)
    raw_coll = client[config.db_name][config.raw_collection]
    clean_coll = client[config.db_name][config.clean_collection]

    # index for fast grouping
    raw_coll.create_index([("MMSI", ASCENDING)])
    clean_coll.create_index([("MMSI", ASCENDING)])

    # enqueue distinct vessels
    q = Queue(maxsize=config.num_workers * 2)
    threads = []
    for worker_id in range(1, config.num_workers + 1):
        t = threading.Thread(
            target=worker,
            args=(q, worker_id, config),
        )
        t.start()
        threads.append(t)

    mmsis = raw_coll.distinct("MMSI")
    for m in tqdm(mmsis):
        q.put(m)
    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()
    client.close()
