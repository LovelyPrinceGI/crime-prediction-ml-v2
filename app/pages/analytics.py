import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path="/analytics", name="Analytics")

df = pd.read_csv("datasets/updated_hate_crime.csv")

# Clean year column if needed
df["year"] = pd.to_numeric(df["year"], errors="coerce")
top_offenses = df["offense_name"].value_counts().nlargest(10).reset_index()
top_offenses.columns = ["offense_name", "count"]

layout = html.Div([
    html.H2("Analytics"),

    dcc.Dropdown(
        df["bias_desc"].unique(),
        id="bias_filter",
        placeholder="Select a Bias Type"
    ),

    dcc.Graph(id="trend_graph"),

    dcc.Graph(
        figure=px.bar(
            top_offenses,
            x="offense_name", y="count",
            labels={"offense_name": "Offense", "count": "Count"},
            title="Top 10 Offense Types"
        )
    )

])

@dash.callback(
    dash.Output("trend_graph", "figure"),
    dash.Input("bias_filter", "value")
)
def update_trend(bias):
    filtered = df if not bias else df[df["bias_desc"] == bias]
    trend = filtered.groupby("year").size().reset_index(name="count")
    return px.line(trend, x="year", y="count", title=f"Hate Crimes Over Time for {bias or 'All Biases'}")
