import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)

plt.rcParams.update({"font.size": 15})

# constants
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
PLOT_DIR = Path("data/plots")

STOP_WORDS = set(stopwords.words("english"))


def clean_text_column(column: pd.Series) -> pd.Series:
    # remove twitter handles
    column = column.str.replace(r"@[a-zA-Z0-9_]+", "")
    # remove URLs
    column = column.str.replace(r"http\S+", "")
    # remove special characters
    column = column.str.replace(r"[^a-zA-Z0-9\s]", "")
    # remove extra whitespaces
    column = column.str.replace(r"\s+", " ")
    # convert to lowercase
    column = column.str.lower()
    # tokenize the text
    column = column.str.strip().str.split()
    # remove stopwords
    column = column.apply(lambda x: [word for word in x if word not in STOP_WORDS])

    return column


def plot_word_clouds(df_data: pd.DataFrame, text_column: str) -> None:
    logging.info("Plotting word clouds...")

    # plot word clouds for each sentiment class
    for sentiment in df_data.airline_sentiment.unique():
        text = " ".join(
            df_data[df_data.airline_sentiment == sentiment][text_column].str.join(" ")
        )
        wordcloud = WordCloud(
            background_color="white",
            max_words=100,
            width=800,
            height=400,
            random_state=42,
        ).generate(text)

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(PLOT_DIR / f"{sentiment}_wordcloud.png", bbox_inches="tight")
        logging.info(f"Word cloud for {sentiment} sentiment saved.")


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    classes: list,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    filename = title.lower().replace("- ", "").replace(" ", "_")
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(PLOT_DIR / filename, bbox_inches="tight")
    logging.info(f"Confusion matrix saved as {filename}.png")


def fit_logistic_regression_model(
    df_data: pd.DataFrame,
    X_column: str,
    y_column: str,
    classes: list,
    title: str,
) -> None:
    X = df_data[X_column].str.join(" ")
    y = df_data[y_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # applying tf-idf vectorization
    tfidf = CountVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
    )
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    # split the data into training and testing sets
    logging.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
    logging.info(f"Testing data shape: {X_test.shape}, {y_test.shape}")

    # train a classifier
    classifier = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )
    classifier.fit(X_train, y_train)

    # evaluate the classifier
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.2f}")

    # per-class classification report
    logging.info(classification_report(y_test, y_pred, digits=3))

    # plot the confusion matrix
    plot_confusion_matrix(
        y_test,
        y_pred,
        classes=df_data.airline_sentiment.unique(),
        title=f"Confusion Matrix - {title}",
    )


def process_file(
    filename: Path,
) -> None:
    # load the csv file
    df_data = pd.read_csv(filename, parse_dates=True)
    logging.info(f"Data shape: {df_data.shape}")
    logging.info(f"Columns: {df_data.columns}")

    logging.info(f"Class distribution:")
    logging.info(df_data.airline_sentiment.value_counts())

    logging.info(f"Tweet length statistics:")
    logging.info(df_data.text.str.len().describe())

    # removing irrelevant columns
    df_data = df_data[["airline_sentiment", "text"]]

    # removing rows with missing values
    logging.info(f"Rows with missing values:")
    logging.info(f"{df_data.isnull().sum(axis=0)}")
    df_data = df_data.dropna()

    # cleaning text data
    df_data["text_cleaned"] = clean_text_column(df_data["text"])

    # plotting the word clouds
    # plot_word_clouds(df_data, text_column="text_cleaned")

    # lemmatizing the text data
    lemmatizer = WordNetLemmatizer()
    df_data["text_cleaned_lemmatized"] = df_data["text_cleaned"].apply(
        lambda x: [lemmatizer.lemmatize(word) for word in x]
    )

    # fitting a logistic regression model without lemmatization
    logging.info("Fitting a logistic regression model without lemmatization...")
    fit_logistic_regression_model(
        df_data,
        X_column="text_cleaned",
        y_column="airline_sentiment",
        classes=df_data.airline_sentiment.unique(),
        title="Without Lemmatization",
    )

    # fitting a logistic regression model with lemmatization
    logging.info("Fitting a logistic regression model with lemmatization...")
    fit_logistic_regression_model(
        df_data,
        X_column="text_cleaned_lemmatized",
        y_column="airline_sentiment",
        classes=df_data.airline_sentiment.unique(),
        title="With Lemmatization",
    )


if __name__ == "__main__":
    # create directories if they don't exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    filenames = list(INPUT_DIR.glob("*.csv"))

    # validate input files
    assert len(filenames) > 0, "No input files provided."

    # process each csv file
    for filename in filenames:
        logging.info(f"Processing {filename}")
        process_file(
            filename,
        )
        logging.info("\n")
