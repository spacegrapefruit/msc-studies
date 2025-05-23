from pyspark.sql import SparkSession
import logging
import os
import glob


def get_spark_session(app_name="AISPortDetection"):
    spark_builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.shuffle.partitions", "50")
        .master("local[*]")
    )  # Use all available cores locally

    spark = spark_builder.getOrCreate()
    logging.info("SparkSession created successfully.")

    # Add project Python files to SparkContext
    project_root = os.path.dirname(os.path.abspath(__file__))

    for py_file in glob.glob(os.path.join(project_root, "*.py")):
        if os.path.basename(py_file) == "main.py":
            continue

        # Using addFile for non-zip/egg modules
        spark.sparkContext.addFile(py_file)
        logging.info(f"Added file to SparkContext: {py_file}")

    return spark
