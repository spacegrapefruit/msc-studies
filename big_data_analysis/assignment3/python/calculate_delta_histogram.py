import logging

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pymongo import MongoClient, ASCENDING

from config import Config

OUTPUT_IMG = "delta_t_histogram.png"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # load configuration
    config_path = Path(__file__).parent.parent / "config.yml"
    config = Config(config_path, "calculate_delta_histogram")

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
    df.to_csv("delta_t_values.csv", index=False)

    plt.figure()
    plt.hist(df["delta_t_ms"], bins=100)
    plt.xlabel("Delta t (ms)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Inter-message Intervals")
    plt.savefig(OUTPUT_IMG)
    print(f"Saved histogram to {OUTPUT_IMG}")
