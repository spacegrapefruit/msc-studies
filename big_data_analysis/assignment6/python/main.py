import logging
from pathlib import Path

import pandas as pd
from config import Config
from spark_session_manager import get_spark_session
from data_loader_and_schema import get_ais_schema, load_data
from filtering_operations import filter_and_prepare_data
from port_detection_logic import detect_ports_dbscan
from port_sizing_logic import evaluate_relative_port_size
from visualize_ports import create_port_visualization_map

# For collecting limited data for visualization (set to None to collect all)
# This is useful if the filtered_df is very large, to avoid OOM on driver for heatmap
MAX_POINTS_FOR_HEATMAP = 200000


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # load configuration
    config_path = Path(__file__).parent.parent / "config.yml"
    config = Config(config_path, "main")

    # initialize Spark
    spark = get_spark_session("AISPortDetectionPipeline")

    try:
        ais_schema = get_ais_schema()
        raw_df = load_data(spark, config.input_file, ais_schema)

        filtered_df = filter_and_prepare_data(raw_df)
        if filtered_df.count() == 0:
            logging.warning(
                "No data after filtering. No ports can be detected. Exiting."
            )
            exit(1)

        # Cache for performance
        filtered_df.persist()

        # The grid-based detection includes merging logic that currently collects to driver.
        # Be mindful of this for very large numbers of potential core cells.
        ports_df = detect_ports_dbscan(filtered_df, spark)
        if ports_df.count() == 0:
            logging.warning("No ports detected. No visualization will be generated.")
            filtered_df.unpersist()
            exit(1)

        # Cache for performance
        ports_df.persist()

        # Evaluate Relative Port Size
        sized_ports_df = evaluate_relative_port_size(ports_df)

        # Visualize Ports
        # Collect data to Pandas for Folium visualization
        logging.info("Collecting data for visualization...")
        ports_for_viz_pd = sized_ports_df.toPandas()

        # Collect a sample of filtered (slow) points for heatmap background
        filtered_points_for_heatmap_pd = pd.DataFrame()
        if filtered_df.count() > 0:
            if MAX_POINTS_FOR_HEATMAP and filtered_df.count() > MAX_POINTS_FOR_HEATMAP:
                logging.info(
                    f"Sampling {MAX_POINTS_FOR_HEATMAP} points from filtered data for heatmap background..."
                )
                sample_fraction = MAX_POINTS_FOR_HEATMAP / filtered_df.count()
                filtered_points_for_heatmap_pd = (
                    filtered_df.select("Latitude", "Longitude")
                    .sample(withReplacement=False, fraction=sample_fraction, seed=42)
                    .toPandas()
                )
            else:
                logging.info("Collecting all filtered points for heatmap background...")
                filtered_points_for_heatmap_pd = filtered_df.select(
                    "Latitude", "Longitude"
                ).toPandas()

        create_port_visualization_map(
            ports_for_viz_pd,
            filtered_points_for_heatmap_pd,
            config.viz_output_file,
        )

        ports_df.unpersist()
        filtered_df.unpersist()

    except Exception as e:
        logging.error(f"An error occurred during the pipeline: {e}")
        import traceback

        traceback.print_exc()
    finally:
        spark.stop()
