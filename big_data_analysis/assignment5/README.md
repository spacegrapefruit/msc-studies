# Medium Articles: Topic Modeling and Trend Analysis over Time

TODO

This project analyzes a dataset of Medium articles to identify latent topics and track their popularity over time using Apache Spark.

## Methodology

TODO

## Project Structure

* `data/`: Stores the input and output files.
  * `input/`: Contains the input CSV file with Medium articles.
  * `output/`: Stores the generated visualizations.
* `python/`: Contains Python scripts for the analysis.
  * `main.py`: The main script to run the analysis pipeline.
  * `config.py`: Configuration class for loading parameters from `config.yml`.
  * `data_loader.py`: Functions for loading and preprocessing the dataset.
  * `topic_modeling_trends.py`: Contains functions for training the LDA model and analyzing topic trends.
  * `visualization.py`: Generates visualizations for topic trends and word clouds.
* `config.yml`: Configuration file for file paths and parameters.
* `Makefile`: Defines the build process and tasks for the project.
* `pyproject.toml`: Defines the project dependencies and configuration for Poetry.
* `README.md`: This file, providing an overview of the project.
* `requirements.txt`: Lists Python package dependencies.

## Requirements

- Apache Spark
- Python 3.10 or 3.11
- Poetry or pip (for Python package management)

## Setup

1. Clone or copy the repo.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and unzip the dataset of Medium articles. Place the CSV file in the `data/input/` directory. The expected file name is:

   ```
   medium_articles.csv
   ```

4. Alter the `config.yml` file to point to the correct input file path. The default is set to `data/input/medium_articles.csv`.

## Running the pipeline

To run the topic modeling pipeline, execute the following command in the terminal:

```bash
[poetry run] python main.py
```

This will start a Spark session, load the data, train the LDA model, analyze topic trends, and generate visualizations.

Sample output:
```
<timestamp> - INFO - Spark session started.
...
TODO
...
<timestamp> - INFO - Closing down clientserver connection
```

The generated visualizations will be saved in the `data/output/` directory:
* `topic_trends.png`
* `wordclouds/topic*.png`

## Findings

Running on the dataset of Medium articles, the LDA model identified several topics, each represented by a set of top words. The trends over time showed how the popularity of these topics evolved, with some topics gaining traction while others declined.
