import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

from dash import dash_table, dcc, html, Input, Output, State

from data_loader import load_reviews
from sentiment_analysis import analyze_tweets

# initial fetch for the first load (without filtering)
initial_reviews = load_reviews()
analyzed_reviews, sentiment_dist = analyze_tweets(initial_reviews)
df_reviews = pd.DataFrame(analyzed_reviews)

# initialize the Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server  # for deployment

app.title = "Movie Reviews Sentiment Analysis Dashboard"
app.layout = html.Div(
    [
        html.H1("Movie Reviews Sentiment Analysis Dashboard"),
        html.Div(
            [
                dcc.Input(
                    id="keyword-input",
                    type="text",
                    placeholder="Enter movie title keyword to search",
                    style={
                        "marginRight": "10px",
                        "width": "300px",
                    },
                ),
                html.Button("Search", id="search-button", n_clicks=0),
            ],
            style={
                "marginTop": "20px",
                "display": "flex",
                "justifyContent": "center",
            },
        ),
        html.Br(),
        html.Div(
            [
                dcc.Graph(
                    id="sentiment-pie-chart",
                    figure=px.pie(
                        names=list(sentiment_dist.keys()),
                        values=list(sentiment_dist.values()),
                        title="Sentiment Distribution",
                    ),
                    style={
                        "height": "50vh",
                    },
                )
            ]
        ),
        html.Br(),
        html.Div(
            [
                dash_table.DataTable(
                    id="reviews-table",
                    columns=[{"name": col, "id": col} for col in df_reviews.columns],
                    data=df_reviews.to_dict("records"),
                    page_size=10,
                    style_cell={
                        "textAlign": "left",
                        "maxWidth": "360px",
                    },
                )
            ]
        ),
    ],
    style={
        "padding": "20px",
        "backgroundColor": "#f8f9fa",
        "fontFamily": "Arial, sans-serif",
    },
)


# callback to fetch and update reviews based on movie keyword
@app.callback(
    [Output("reviews-table", "data"), Output("sentiment-pie-chart", "figure")],
    [Input("search-button", "n_clicks"), Input("keyword-input", "n_submit")],
    [State("keyword-input", "value")],
)
def update_output(n_clicks: int, n_submit: int, keyword: str) -> tuple:
    """Fetch and update reviews based on the keyword input."""
    # if no keyword is provided, use the initial reviews
    # otherwise, fetch reviews based on the keyword
    if not keyword:
        reviews = initial_reviews
    else:
        reviews = load_reviews(keyword)

    analyzed, sentiment_dist = analyze_tweets(reviews)
    df_filtered = pd.DataFrame(analyzed)
    fig = px.pie(
        names=list(sentiment_dist.keys()),
        values=list(sentiment_dist.values()),
        title="Sentiment Distribution",
        color=list(sentiment_dist.keys()),
        color_discrete_map={
            "Positive": "royalblue",
            "Neutral": "grey",
            "Negative": "red",
        },
    )
    return df_filtered.to_dict("records"), fig


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
