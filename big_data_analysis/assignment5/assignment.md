# Medium Articles: Topic Modeling and Trend Analysis over Time

Objective: analyze a dataset of Medium articles using Apache Spark. The main goal is to discover the hidden topics within the articles and track how their popularity changes over time.

Steps:
* Preprocess data: Clean and prepare the article text for analysis.
* Apply topic modeling: Use an algorithm like to identify key themes.
* Detect trends: Link the topics to the article timestamps to find trends, bursts in popularity, or shifts in focus.
* Visualize results: Create charts and graphs to clearly present the findings.

Input data: `data/input/medium_articles.csv`

Columns:
* `title` - Title of the article.
* `text` - Full text of the article.
* `url` - URL of the article.
* `authors` - Authors of the article.
* `timestamp` - Publication date and time of the article.
* `tags` - Tags associated with the article.
