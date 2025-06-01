import logging
import os
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession

from config import Config
from data_loader import get_ais_schema, load_data
from filtering_operations import filter_and_prepare_data
from port_detection import detect_ports_dbscan, evaluate_relative_port_size
from visualize_ports import create_port_visualization_map


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # load configuration
    config_path = Path(__file__).parent.parent / "config.yml"
    config = Config(config_path, "main")

    # initialize Spark
    spark = (
        SparkSession.builder.appName("AISPortDetection")
        .config("spark.driver.memory", "4g")  # give it 4GB of memory
        .master("local[*]")  # use all available cores
        .getOrCreate()
    )
    logging.info("Spark session started.")

    try:
        ais_schema = get_ais_schema()
        raw_df = load_data(spark, config.input_file, ais_schema)

        filtered_df = filter_and_prepare_data(raw_df)
        if filtered_df.count() == 0:
            logging.warning(
                "No data after filtering. No ports can be detected. Exiting."
            )
            exit(1)

        # cache for performance
        filtered_df.persist()

        # detect ports using DBSCAN
        ports_df = detect_ports_dbscan(filtered_df, spark, config=config.dbscan)
        if ports_df.count() == 0:
            logging.warning("No ports detected. No visualization will be generated.")
            filtered_df.unpersist()
            exit(1)

        # evaluate relative port size
        sized_ports_df = evaluate_relative_port_size(ports_df)

        # collect data to Pandas for visualization
        logging.info("Collecting data for visualization...")
        ports_for_viz_pd = sized_ports_df.toPandas()

        # sample filtered (slow) points for heatmap background
        logging.info("Collecting all filtered points for heatmap background...")
        filtered_points_for_heatmap_pd = filtered_df.select(
            "Latitude", "Longitude"
        ).toPandas()

        create_port_visualization_map(
            ports_for_viz_pd,
            filtered_points_for_heatmap_pd,
            config.viz_output_file,
        )

        # clean up resources
        filtered_df.unpersist()

    except Exception as e:
        logging.error(
            f"An error occurred during processing: {e}",
            exc_info=True,
        )

    finally:
        spark.stop()
