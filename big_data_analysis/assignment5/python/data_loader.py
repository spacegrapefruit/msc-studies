import logging

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    CountVectorizer,
    CountVectorizerModel,
    IDF,
    StopWordsRemover,
    Tokenizer,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, concat_ws, lower, regexp_replace, to_timestamp


def load_preprocess_data(
    spark: SparkSession, input_path: str
) -> tuple[DataFrame, CountVectorizerModel]:
    """
    Loads data and creates a Spark ML Pipeline to preprocess the text.
    """
    logging.info("Starting data preprocessing...")

    # load the dataset
    df = spark.read.csv(
        input_path,
        header=True,
        inferSchema=True,
        multiLine=True,
        escape='"',
        quote='"',
    )
    logging.info(
        f"Loaded dataset with {df.count()} articles and {len(df.columns)} columns."
    )

    # select relevant columns, drop rows with null text, parse timestamp
    df_clean = (
        df.select(["title", "text", "timestamp", "tags"])
        .dropna(subset=["text"])
        .withColumn("date", to_timestamp(col("timestamp")))
    )

    # convert to lowercase and remove non-alphabetic characters
    df_clean = df_clean.withColumn(
        "text_clean",
        lower(concat_ws(" ", col("title"), col("tags"), col("text"))),
    )
    df_clean = df_clean.withColumn(
        "text_clean", regexp_replace(col("text_clean"), r"[^a-z\s]", "")
    )

    # feature engineering pipeline
    # tokenizer to split text into words
    tokenizer = Tokenizer(
        inputCol="text_clean",
        outputCol="words",
    )

    # StopWordsRemover to remove common stop words
    stopwords_remover = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words",
    )

    # CountVectorizer to convert words into feature vectors
    cv = CountVectorizer(
        inputCol="filtered_words",
        outputCol="raw_features",
        vocabSize=5000,
        minDF=5,
    )

    # IDF to scale down the importance of common words
    idf = IDF(
        inputCol="raw_features",
        outputCol="features",
    )

    # finally, create a pipeline to chain the stages together
    pipeline = Pipeline(
        stages=[
            tokenizer,
            stopwords_remover,
            cv,
            idf,
        ]
    )

    # fit the pipeline to the data
    logging.info("Fitting the preprocessing pipeline...")
    pipeline_model = pipeline.fit(df_clean)

    # transform the data
    processed_data = pipeline_model.transform(df_clean)

    # extract the CountVectorizer model
    cv_model = pipeline_model.stages[2]

    logging.info("Preprocessing complete.")
    return processed_data, cv_model
