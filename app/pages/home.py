import dash
from dash import html, dcc
import pandas as pd

dash.register_page(__name__, path="/", name="Home")

df = pd.read_csv("datasets/updated_hate_crime.csv")

layout = html.Div([
    html.H2("Summary Statistics"),

    html.Div([
        html.P(f"Total Incidents: {df.shape[0]}"),
        html.P(f"Total Victims: {df['victim_count'].sum()}"),
        html.P(f"Most Common Bias: {df['bias_desc'].mode()[0]}"),
        html.P(f"Top State: {df['state_name'].value_counts().idxmax()}"),
        html.P(f"Top Offense: {df['offense_name'].value_counts().idxmax()}"),
    ])
])
