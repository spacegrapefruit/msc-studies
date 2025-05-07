import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pymongo import MongoClient, ASCENDING
from tqdm import tqdm

from config import Config

delta_t_pipeline = [
    # add prevTs = timestamp of the previous doc in the same MMSI
    {
        "$setWindowFields": {
            "partitionBy": "$MMSI",
            "sortBy": {"Timestamp": 1},
            "output": {
                "prevTs": {
                    "$shift": {
                        "output": "$Timestamp",
                        "by": -1,  # shift backwards by 1
                        "default": None,
                    }
                }
            },
        }
    },
    # compute the delta, in ms
    {
        "$addFields": {
            "delta_t_ms": {
                "$multiply": [
                    {"$subtract": ["$Timestamp", "$prevTs"]},
                    1,  # already in ms
                ]
            }
        }
    },
    # drop the first row per MMSI
    {"$match": {"prevTs": {"$ne": None}}},
    # project only the delta
    {"$project": {"_id": 0, "delta_t_ms": 1}},
]


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
    logging.info("Connected to MongoDB")

    clean_coll = client[config.db_name][config.clean_collection]

    # run the aggregation and pull back a flat list of numbers
    cursor = clean_coll.aggregate(delta_t_pipeline)
    deltas = [doc["delta_t_ms"] for doc in tqdm(cursor)]
    logging.info(f"Retrieved {len(deltas)} delta t values")

    client.close()

    df = pd.DataFrame({"delta_t_ms": deltas})
    csv_path = output_dir / "delta_t_values.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved delta t values to {csv_path}")

    data = df["delta_t_ms"][df["delta_t_ms"] > 0]

    # make bins spaced evenly in log-space between lowest and highest values
    min_val, max_val = data.min(), data.max()
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)

    plt.figure()
    plt.hist(data, bins=bins, log=True)
    plt.xscale("log")
    plt.xlabel("Delta t (ms)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Inter-message Intervals")

    img_path = output_dir / "delta_t_histogram.png"
    plt.savefig(img_path)
    logging.info(f"Saved histogram to {img_path}")
