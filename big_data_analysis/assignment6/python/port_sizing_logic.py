import logging
from pyspark.sql.functions import col, log1p, min, max, greatest, lit


def evaluate_relative_port_size(ports_df):
    if ports_df.count() == 0:
        logging.warning("No ports to evaluate for size.")
        return ports_df

    logging.info("Evaluating relative port sizes (DBSCAN version)...")

    # Ensure required columns exist
    required_cols = ["num_signals", "num_unique_vessels"]
    for r_col in required_cols:
        if r_col not in ports_df.columns:
            logging.error(
                f"Error: Column '{r_col}' not found in ports_df for sizing. Columns are: {ports_df.columns}"
            )
            return ports_df  # Or raise an error

    min_max_signals = ports_df.select(
        min("num_signals").alias("min_s"), max("num_signals").alias("max_s")
    ).first()
    min_s, max_s = (
        min_max_signals["min_s"] if min_max_signals else None,
        min_max_signals["max_s"] if min_max_signals else None,
    )

    min_max_vessels = ports_df.select(
        min("num_unique_vessels").alias("min_v"),
        max("num_unique_vessels").alias("max_v"),
    ).first()
    min_v, max_v = (
        min_max_vessels["min_v"] if min_max_vessels else None,
        min_max_vessels["max_v"] if min_max_vessels else None,
    )

    ports_with_size = ports_df.withColumn(
        "rel_size_signals_log",
        log1p(col("num_signals")) if min_s is not None else lit(0.0),
    ).withColumn(
        "rel_size_vessels_log",
        log1p(col("num_unique_vessels")) if min_v is not None else lit(0.0),
    )

    if min_s is not None:
        min_max_log_s_row = ports_with_size.select(
            min("rel_size_signals_log").alias("min_val"),
            max("rel_size_signals_log").alias("max_val"),
        ).first()
        min_log_s, max_log_s = (
            min_max_log_s_row["min_val"],
            min_max_log_s_row["max_val"],
        )
        denominator_s = max_log_s - min_log_s
        ports_with_size = ports_with_size.withColumn(
            "norm_size_signals",
            (col("rel_size_signals_log") - lit(min_log_s))
            / greatest(lit(1e-6), lit(denominator_s))
            if denominator_s > 1e-6
            else lit(0.5),
        )
    else:
        ports_with_size = ports_with_size.withColumn("norm_size_signals", lit(0.0))

    if min_v is not None:
        min_max_log_v_row = ports_with_size.select(
            min("rel_size_vessels_log").alias("min_val"),
            max("rel_size_vessels_log").alias("max_val"),
        ).first()
        min_log_v, max_log_v = (
            min_max_log_v_row["min_val"],
            min_max_log_v_row["max_val"],
        )
        denominator_v = max_log_v - min_log_v
        ports_with_size = ports_with_size.withColumn(
            "norm_size_vessels",
            (col("rel_size_vessels_log") - lit(min_log_v))
            / greatest(lit(1e-6), lit(denominator_v))
            if denominator_v > 1e-6
            else lit(0.5),
        )
    else:
        ports_with_size = ports_with_size.withColumn("norm_size_vessels", lit(0.0))

    # Composite score based on two factors now
    ports_with_size = ports_with_size.withColumn(
        "composite_size_score",
        (col("norm_size_signals") + col("norm_size_vessels")) / 2.0,
    )

    logging.info(f"Added relative size metrics. Total ports: {ports_with_size.count()}")
    return ports_with_size
