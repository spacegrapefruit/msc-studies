import logging
import threading

import pandas as pd
from pathlib import Path
from pymongo import MongoClient
from queue import Queue
from tqdm import tqdm

from config import Config


def worker(q: Queue, worker_id: int, config: Config):
    """
    Worker function to process batches of data and insert them into MongoDB.
    """
    logging.info(f"Worker {worker_id} starting")

    client = MongoClient(config.mongo_uri)
    logging.info(f"Worker {worker_id} connected to MongoDB")

    raw_coll = client[config.db_name][config.raw_collection]
    while True:
        batch = q.get()
        if batch is None:
            logging.info(f"Worker {worker_id} stopping")
            break
        raw_coll.insert_many(batch)
        logging.info(f"Worker {worker_id} inserted {len(batch)} records")
        q.task_done()
    client.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # load configuration
    config_path = Path(__file__).parent.parent / "config.yml"
    config = Config(config_path, "ingest")

    q = Queue(maxsize=config.num_workers * 2)
    threads = []
    for worker_id in range(1, config.num_workers + 1):
        t = threading.Thread(
            target=worker,
            args=(q, worker_id, config),
        )
        t.start()
        threads.append(t)

    for chunk_df in tqdm(
        pd.read_csv(
            config.input_file, chunksize=config.batch_size, nrows=config.max_rows
        ),
    ):
        batch = chunk_df.to_dict(orient="records")
        q.put(batch)

    # stop workers
    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()
