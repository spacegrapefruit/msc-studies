import logging

import pandas as pd
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.ml.feature import CountVectorizerModel
from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, udf, col, month, year
from pyspark.sql.types import ArrayType, DoubleType


def train_lda_model(
    processed_data: DataFrame, num_topics: int = 10, max_iter: int = 20
) -> tuple[LDAModel, DataFrame]:
    """
    Trains a Latent Dirichlet Allocation (LDA) model.
    """
    logging.info(f"Training LDA model with {num_topics} topics...")

    # initialize LDA
    lda = LDA(k=num_topics, maxIter=max_iter, featuresCol="features")

    # train the model
    lda_model = lda.fit(processed_data)

    # add topic distributions to the DataFrame
    df_with_topics = lda_model.transform(processed_data)

    logging.info("LDA model training complete.")
    return lda_model, df_with_topics


def get_topic_keywords(
    lda_model: LDAModel,
    cv_model: CountVectorizerModel,
    num_top_words: int = 3,
    verbose: bool = False,
) -> list[str]:
    """
    Returns a list of top keywords for each topic. If verbose is True, prints the keywords.
    """
    topics = lda_model.describeTopics(maxTermsPerTopic=num_top_words)
    vocabulary = cv_model.vocabulary

    if verbose:
        logging.info("\n--- Discovered Topics ---")

    topic_keywords = []
    for i, topic in enumerate(topics.collect()):
        word_indices = topic["termIndices"]
        topic_words = [vocabulary[idx] for idx in word_indices]
        topic_keywords.append(", ".join(topic_words))

        if verbose:
            logging.info(f"Topic {i}:")
            word_weights = topic["termWeights"]

            for word, weight in zip(topic_words, word_weights):
                logging.info(f"  - {word} (weight: {weight:.4f})")

    if verbose:
        logging.info("-" * 25)

    return topic_keywords


def analyze_topic_trends(df_with_topics, num_topics):
    """
    Aggregates topic prevalence over time (monthly).
    """
    logging.info("Analyzing topic trends over time...")

    # UDF to extract a single topic's weight
    def get_topic_weight(topic_index):
        return udf(lambda v: float(v[topic_index]), DoubleType())

    select_exprs = [col("*")]  # keep all existing columns
    for i in range(num_topics):
        select_exprs.append(
            get_topic_weight(i)(col("topicDistribution")).alias(f"topic_{i}_weight")
        )

    # apply all transformations in a single select statement
    df_with_topics = df_with_topics.select(select_exprs)
    logging.info("Extracted topic weights from topic distribution.")

    # aggregate by year and month
    df_trends = df_with_topics.withColumn("year", year(col("date"))).withColumn(
        "month", month(col("date"))
    )

    # create aggregation expressions
    agg_exprs = [col("year"), col("month")]
    for i in range(num_topics):
        agg_exprs.append(f"avg(topic_{i}_weight) as avg_weight_topic_{i}")

    # group by month, year and calculate average topic weights
    monthly_trends = (
        df_trends.groupBy("year", "month")
        .agg(
            *[
                avg(col(f"topic_{i}_weight")).alias(f"topic_{i}")
                for i in range(num_topics)
            ]
        )
        .orderBy("year", "month")
    )

    logging.info("Converting trend data to Pandas DataFrame.")
    # convert to Pandas DataFrame
    trends_df = monthly_trends.toPandas()

    # create a 'date' column for the x-axis
    trends_df["date"] = pd.to_datetime(trends_df[["year", "month"]].assign(day=1))
    trends_df = trends_df.set_index("date")

    # reindex the DataFrame to ensure all months are present
    full_date_range = pd.date_range(
        start=trends_df.index.min(), end=trends_df.index.max(), freq="MS"
    )
    trends_df = trends_df.reindex(full_date_range)

    return trends_df
