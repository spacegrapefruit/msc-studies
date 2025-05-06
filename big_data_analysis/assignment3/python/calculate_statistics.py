import logging

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pymongo import MongoClient, ASCENDING

from config import Config


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # load configuration
    config_path = Path(__file__).parent.parent / "config.yml"
    config = Config(config_path, "calculate_statistics")
    output_dir = Path(__file__).parent.parent / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # connect to MongoDB
    client = MongoClient(config.mongo_uri)
    clean_coll = client[config.db_name][config.clean_collection]

    # ensure sorted by time
    cursor = clean_coll.find({}, {"MMSI": 1, "Timestamp": 1}).sort(
        [("MMSI", ASCENDING), ("Timestamp", ASCENDING)]
    )

    prev = {}
    deltas = []
    for doc in cursor:
        m = doc["MMSI"]
        t = pd.to_datetime(doc["Timestamp"])
        if m in prev:
            deltas.append((t - prev[m]).total_seconds() * 1000)
        prev[m] = t

    client.close()

    df = pd.DataFrame({"delta_t_ms": deltas})
    csv_path = output_dir / "delta_t_values.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved delta t values to {csv_path}")

    plt.figure()
    plt.hist(df["delta_t_ms"], bins=100)
    plt.xlabel("Delta t (ms)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Inter-message Intervals")
    img_path = output_dir / "delta_t_histogram.png"
    plt.savefig(img_path)
    logging.info(f"Saved histogram to {img_path}")
