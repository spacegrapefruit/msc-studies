import logging
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.clustering import LDAModel
from pyspark.ml.feature import CountVectorizerModel
from scipy.ndimage import gaussian_filter
from wordcloud import WordCloud


def plot_topic_trends(
    trends_df: pd.DataFrame, topic_keywords: list[str], output_path: Path
) -> None:
    """
    Plots the topic prevalence over time and saves it as a PNG file.
    """
    logging.info(f"Generating and saving topic trends plot to {output_path}...")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 9))
    colors = plt.get_cmap("tab20").colors

    # plot each topic's trend line
    plot_columns = [col for col in trends_df.columns if col.startswith("topic_")]
    for idx, (column, keywords) in enumerate(zip(plot_columns, topic_keywords)):
        if column.startswith("topic_"):
            # smooth the trend line
            trends_df[column] = gaussian_filter(trends_df[column], 2)

            # plot the trend line
            ax.plot(
                trends_df.index,
                trends_df[column],
                label=f"Topic {idx}: {keywords}",
                color=colors[idx],
            )

    # formatting the plot
    ax.set_title("Topic Popularity Over Time on Medium", fontsize=20, pad=20)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Average Topic Prevalence", fontsize=14)
    ax.legend(title="Topics & Keywords", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.ylim(-0.005, 0.255)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend

    # save the figure
    fig.savefig(output_path)
    logging.info("Trends plot saved successfully.")
    plt.close(fig)


def plot_topic_wordclouds(
    lda_model: LDAModel, cv_model: CountVectorizerModel, output_dir: Path
) -> None:
    """
    Generates and saves a word cloud for each topic.
    """
    logging.info(f"Generating and saving word clouds to {output_dir}...")

    topics = lda_model.describeTopics(maxTermsPerTopic=50)
    vocabulary = cv_model.vocabulary

    for i, topic in enumerate(topics.collect()):
        # create a dictionary of word: weight for the word cloud
        term_indices = topic["termIndices"]
        term_weights = topic["termWeights"]
        word_weights_dict = {
            vocabulary[idx]: weight for idx, weight in zip(term_indices, term_weights)
        }

        wc = WordCloud(
            background_color="white", width=800, height=600, max_words=50
        ).generate_from_frequencies(word_weights_dict)

        plt.figure(figsize=(10, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.title(f"Topic {i}", fontsize=20)
        plt.axis("off")

        # save the figure
        output_path = output_dir / f"topic_{i}.png"
        plt.savefig(output_path)
        plt.close()

    logging.info("Word clouds saved successfully.")
