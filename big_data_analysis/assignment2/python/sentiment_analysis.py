from textblob import TextBlob


def get_sentiment(text: str) -> str:
    """Return the sentiment category for a given text."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


def analyze_tweets(tweets: list) -> tuple:
    """
    Analyze a list of reviews.

    Each review is expected to have a 'text' key with the review content.
    Returns:
        - The list of reviews with an added 'sentiment' key
        - A dictionary with sentiment distribution counts
    """
    sentiment_distribution = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for tweet in tweets:
        sentiment = get_sentiment(tweet["text"])
        tweet["sentiment"] = sentiment
        sentiment_distribution[sentiment] += 1

    return tweets, sentiment_distribution
