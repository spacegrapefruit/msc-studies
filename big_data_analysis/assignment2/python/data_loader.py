import os
import requests

from functools import lru_cache


@lru_cache(maxsize=None)
def load_reviews(keyword=None):
    """
    Load movie reviews using TMDB API.

    If a keyword is provided, search for a movie by title.
    If not, select a popular movie from the trending list.

    Returns:
        A list of review dictionaries with keys:
          'text' (mapped from review content) and 'author'.
    """
    api_key = os.environ["TMDB_API_KEY"]

    # determine which movie to fetch reviews for.
    if keyword:
        # search for a movie using the keyword.
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": api_key, "query": keyword, "language": "en-US"}
        search_response = requests.get(search_url, params=params)
        if search_response.status_code == 200:
            search_data = search_response.json()
            results = search_data.get("results", [])
            if not results:
                raise Exception("No movie found for the given keyword.")
            movie_id = results[0]["id"]
        else:
            raise Exception(
                f"Error searching movie: {search_response.status_code} - {search_response.text}"
            )
    else:
        # if no keyword is provided, fetch popular movies.
        popular_url = "https://api.themoviedb.org/3/movie/popular"
        params = {"api_key": api_key, "language": "en-US"}
        popular_response = requests.get(popular_url, params=params)
        if popular_response.status_code == 200:
            popular_data = popular_response.json()
            results = popular_data.get("results", [])
            if not results:
                raise Exception("No popular movies found.")
            movie_id = results[0]["id"]
        else:
            raise Exception(
                f"Error fetching popular movies: {popular_response.status_code} - {popular_response.text}"
            )

    # fetch reviews for the selected movie.
    reviews_url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews"
    params = {"api_key": api_key, "language": "en-US"}
    reviews_response = requests.get(reviews_url, params=params)
    if reviews_response.status_code == 200:
        reviews_data = reviews_response.json()
        reviews_list = []
        for review in reviews_data.get("results", []):
            reviews_list.append(
                {
                    "text": review.get("content"),
                    "created_at": review.get("created_at"),
                }
            )
        return reviews_list
    else:
        raise Exception(
            f"Error fetching reviews: {reviews_response.status_code} - {reviews_response.text}"
        )
