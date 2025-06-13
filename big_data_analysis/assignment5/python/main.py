import logging
from pathlib import Path
from pyspark.sql import SparkSession

from config import Config
from data_loader import load_preprocess_data
from topic_modeling_trends import analyze_topic_trends, describe_topics, train_lda_model
from visualization import plot_topic_trends, plot_topic_wordclouds


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # load configuration
    config_path = Path(__file__).parent.parent / "config.yml"
    config = Config(config_path)

    output_plot_path = Path(config.output_plot_path)
    output_wordcloud_dir = Path(config.output_wordcloud_dir)

    # create output directories if they don't exist
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    output_wordcloud_dir.mkdir(parents=True, exist_ok=True)

    # initialize Spark
    spark = (
        SparkSession.builder.appName("MediumTopicTrends")
        .config("spark.driver.memory", "8g")  # give it 8GB of memory
        .master("local[*]")  # use all available cores
        .getOrCreate()
    )
    logging.info("Spark session started.")

    try:
        # preprocess data
        processed_data, cv_model = load_preprocess_data(spark, config.input_path)
        # cache the processed data
        processed_data.persist()

        # train LDA model
        lda_model, df_with_topics = train_lda_model(
            processed_data,
            num_topics=config.num_topics,
        )

        # print the topics
        describe_topics(lda_model, cv_model, num_top_words=5)

        # analyze topic trends
        trends_df = analyze_topic_trends(df_with_topics, num_topics=config.num_topics)

        # visualize trends and word clouds
        plot_topic_trends(trends_df, output_plot_path)
        plot_topic_wordclouds(lda_model, cv_model, output_wordcloud_dir)
        logging.info("Topic trends analysis completed successfully.")

    except Exception as e:
        logging.error(
            f"An error occurred during processing: {e}",
            exc_info=True,
        )
    finally:
        logging.info("Stopping Spark session.")
        spark.stop()
