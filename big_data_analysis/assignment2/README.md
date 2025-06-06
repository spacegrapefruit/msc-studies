# Movie Reviews Sentiment Analysis Dashboard

This Dash application retrieves movie reviews using the TMDB API and performs sentiment analysis on the review content. Users can enter a movie title keyword to search for a specific movie and see its reviews, and a pie chart of the sentiment breakdown.

## Project Structure

```
assignment2/
└── python/
    ├── app.py                # Main Dash application
    ├── data_loader.py        # Module for fetching movie reviews from the TMDB API
    └── sentiment_analysis.py # Module for performing sentiment analysis
├── Dockerfile                # Docker configuration
├── Makefile                  # Project-specific build and run commands
├── pyproject.toml            # Project configuration file for Poetry
├── poetry.lock               # Lockfile for project dependencies
└── README.md                 # This file
```

## Setup

1. **Obtain a TMDB API Key**:
   - Sign up at [TMDB](https://www.themoviedb.org/documentation/api) and obtain a free API key.
   - Set your TMDB API key as an environment variable:

        ```bash
        export TMDB_API_KEY=your_tmdb_api_key
        ```

2. **Install dependencies locally** (optional):
    - Install Poetry:

        ```bash
        pip install poetry
        ```

    - Install the dependencies using Poetry:
     
        ```bash
        make install
        ```

## Running Locally

1. Ensure the **TMDB_API_KEY** environment variable is set.
2. Run the app:
   ```bash
   make run
   ```
3. Open your browser and navigate to [http://localhost:8050](http://localhost:8050).

## Running with Docker

1.
   a) Build the Docker image:
   ```bash
   docker build -t sentiment-dashboard .
   ```

   b) Alternatively, download the pre-built image from Docker Hub:
   ```bash
   docker pull ghcr.io/spacegrapefruit/sentiment-dashboard:latest
   ```
2. Run the container (passing the TMDB API key environment variable):
   ```bash
   docker run -p 8050:8050 -e TMDB_API_KEY=your_tmdb_api_key sentiment-dashboard
   ```
3. Open your browser and navigate to [http://localhost:8050](http://localhost:8050).

## Usage

- The dashboard displays a pie chart showing the sentiment distribution of movie reviews.
- Enter a movie title keyword in the input field and click **Search** to fetch reviews for that movie.
- If no keyword is entered, reviews for a popular movie are displayed.
