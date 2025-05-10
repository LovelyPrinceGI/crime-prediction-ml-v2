import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/analytics", name="Analytics")

# Read in the data
df = pd.read_csv("datasets/updated_hate_crime.csv")

# Clean year column if needed
df["year"] = pd.to_numeric(df["year"], errors="coerce")

trend = df.groupby("year").size().reset_index(name="count")
# Generate top offenses
top_offenses = df["offense_name"].value_counts().nlargest(10).reset_index()
top_offenses.columns = ["offense_name", "count"]

# Group bias types and calculate top 10
bias_counts = df['bias_desc'].value_counts().reset_index()
bias_counts.columns = ['Bias Type', 'Incident Count']

# Group remaining biases as 'Others'
top_10_biases = bias_counts.head(10)  # Select top 10 biases
other_biases = bias_counts.tail(len(bias_counts) - 10)  # Get the remaining biases
others = pd.DataFrame({'Bias Type': ['Others'], 'Incident Count': [other_biases['Incident Count'].sum()]})

# Combine top 10 with "Others"
final_bias_counts = pd.concat([top_10_biases, others])

top_states = df["state_name"].value_counts().nlargest(10).reset_index()
top_states.columns = ["state_name", "incident_count"]

top_location_names = df["location_name"].value_counts().nlargest(10).reset_index()
top_location_names.columns = ["location_name", "incident_count"]

crime_type_trends = df.groupby(["year", "crime_type"]).size().reset_index(name="incident_count")

layout = html.Div([
    html.H2("Analytics", className="text-center mb-4"),

    # Analytics section with two columns
    dbc.Row([
        # Left column with trend chart
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Hate Crimes Over Time", className="card-title text-uppercase font-weight-bold"),
                    dcc.Graph(
                        figure=px.line(
                            trend, x="year", y="count",
                            markers=True,
                            labels={"year": "Year", "count": "Number of Incidents"}
                        ) 
                    )
                ]),
                className="shadow-sm rounded-lg border border-light mb-4",
                style={"minHeight": "300px", "box-shadow": "none"}  # Ensure consistent card style
            ),
            md=6  # Set column to take half the width of the row
        ),

        # Right column with top offenses bar chart
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Top 10 Offense Types", className="card-title text-uppercase font-weight-bold"),
                    dcc.Graph(
                        figure=px.bar(
                            top_offenses,
                            x="offense_name", y="count",
                            labels={"offense_name": "Offense", "count": "Count"}
                        )
                    ),
                ]),
                className="shadow-lg rounded-lg border border-light mb-4",
                style={"minHeight": "300px","box-shadow": "none"}
            ),
            md=6  # Set column to take half the width of the row
        ),
    ], className="mb-5"),

    # Bias distribution pie chart card
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Bias Distribution", className="card-title text-uppercase font-weight-bold"),
                    dcc.Graph(
                        figure=px.pie(final_bias_counts, names='Bias Type', values='Incident Count', 
                                      title="Distribution of Bias Types in Hate Crimes")
                    ),
                ]),
                className="shadow-lg rounded-lg border border-light mb-4",
                style={"minHeight": "300px","box-shadow": "none"}
            ),
            md=12  # Set this to full width, as we want the pie chart to span across
        ),
    ], className="mb-5"),
    dbc.Row([
        # Left column with trend chart
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Top 10 States", className="card-title text-uppercase font-weight-bold"),
                    dcc.Graph(
                        figure=px.bar(
                            top_states,
                            x="state_name", y="incident_count",
                            labels={"state_name": "State", "incident_count": "Number of Incidents"}
                        )
                    ),
                ]),
                className="shadow-lg rounded-lg border border-light mb-4",
                style={"minHeight": "300px","box-shadow": "none"}
            ),
            md=6  # Set column to take half the width of the row
        ),

        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Top 10 Locations", className="card-title text-uppercase font-weight-bold"),
                    dcc.Graph(
                        figure=px.bar(
                            top_location_names,
                            x="location_name", y="incident_count",
                            labels={"location_name": "Location", "incident_count": "Number of Incidents"}
                        )
                    ),
                ]),
                className="shadow-lg rounded-lg border border-light mb-4",
                style={"minHeight": "300px","box-shadow": "none"}
            ),
            md=6  # Set column to take half the width of the row
        ),
    ], className="mb-5"),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Crime Type Trend (Violent vs. Non-Violent)", className="card-title text-uppercase font-weight-bold"),
                    dcc.Graph(
                        figure=px.line(
                            crime_type_trends,
                            x="year",
                            y="incident_count",
                            color="crime_type",
                            markers=True,
                            labels={"year": "Year", "incident_count": "Number of Incidents","crime_type":"Crime Type"}
                        ) 
                    )
                ]),
                className="shadow-sm rounded-lg border border-light mb-4",
                style={"minHeight": "300px", "box-shadow": "none"}
            ),
            md=12  # Set this to full width for the trend graph
        ),
    ], className="mb-5"),


])

