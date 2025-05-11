import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
import numpy as np

# Register prediction page
dash.register_page(__name__, path="/prediction", name="Prediction")

# Load trained model
with open("saved_model/best_multilabel_model_rf.pkl", "rb") as f:
    model = pickle.load(f)

# Define label names (ordered to match model's output)
LABELS = [
    "Disability Hate Crime",
    "Gender Hate Crime",
    "Racial Hate Crime",
    "Religion Hate Crime"
]

layout = dbc.Container([
    html.H2("Predict Hate Crime Bias Type", className="my-4 text-center"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Total Offender Count"),
            dcc.Input(id='offender-count', type='number', min=0, className="form-control")
        ]),
        dbc.Col([
            dbc.Label("Victim Count"),
            dcc.Input(id='victim-count', type='number', min=0, className="form-control")
        ]),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Total Individual Victims"),
            dcc.Input(id='individual-victims', type='number', min=0, className="form-control")
        ]),
        dbc.Col([
            dbc.Label("Crime Type (Violent?)"),
            dcc.Dropdown(
                id='is-violent',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                className="form-control",
                placeholder="Select"
            )
        ]),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Area Type (Urban?)"),
            dcc.Dropdown(
                id='is-urban',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                className="form-control",
                placeholder="Select"
            )
        ]),
        dbc.Col([
            dbc.Label("Offender Ethnicity: Not Hispanic or Latino?"),
            dcc.Dropdown(
                id='is-not-hispanic',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                className="form-control",
                placeholder="Select"
            )
        ]),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(
            dbc.Button("Predict", id="predict-btn", color="dark", style={"backgroundColor": "#4b0082", "border": "none"}),
            width=12, className="d-flex justify-content-center mb-3"
        )
    ]),

    html.Div(id="prediction-output", className="row justify-content-center g-4")
], className="pt-4")


@dash.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("offender-count", "value"),
    State("is-violent", "value"),
    State("victim-count", "value"),
    State("individual-victims", "value"),
    State("is-urban", "value"),
    State("is-not-hispanic", "value")
)
def predict_bias(n_clicks, offender_count, is_violent, victim_count, individual_victims, is_urban, is_not_hispanic):
    if n_clicks is None:
        return ""

    if any(v is None for v in [offender_count, is_violent, victim_count, individual_victims, is_urban, is_not_hispanic]):
        return dbc.Alert("⚠️ All fields are required to get a prediction.", color="danger")

    input_data = pd.DataFrame([{
        'total_offender_count': offender_count,
        'crime_type_Violent': is_violent,
        'victim_count': victim_count,
        'total_individual_victims': individual_victims,
        'area_type_Urban': is_urban,
        'offender_ethnicity_Not Hispanic or Latino': is_not_hispanic
    }])

    proba = model.predict_proba(input_data)
    prediction = model.predict(input_data)[0]

    cards = []
    for i, label in enumerate(LABELS):
        conf = proba[i][0][1] * 100
        is_positive = prediction[i] == 1
        border_color = "#28a745" if is_positive else "#dc3545"
        icon = "✅" if is_positive else "❌"

        card = dbc.Col(
            dbc.Card([
                dbc.CardHeader(label, className="fw-semibold text-center"),
                dbc.CardBody([
                    html.H5(f"{icon} {'Yes' if is_positive else 'No'}", className="card-title text-center mb-1"),
                    html.P(f"Confidence: {conf:.2f}%", className="card-text text-center text-muted")
                ])
            ], style={"backgroundColor": "#ffffff", "border": f"2px solid {border_color}"}, className="rounded-4 shadow-sm"), md=3
        )
        cards.append(card)

    return cards
