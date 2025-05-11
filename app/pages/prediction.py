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

location_options = [
    "Home", "Street", "Transport", "School", "Govt", "Religious",
    "Retail", "Medical", "Outdoor", "Worksite", "Community", "Cyber",
    "Entertainment", "Other"
]

division_options = [
    "East South Central", "Middle Atlantic", "Mountain", "New England",
    "Pacific", "South Atlantic", "U.S. Territories", "West North Central", "West South Central"
]

ethnicity_options = [
    "Multiple", "Not Hispanic or Latino", "Not Specified", "Unknown"
]

layout = dbc.Container([
    html.H2("Predict Hate Crime Bias Type", className="my-4 text-center"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Juvenile Victim Count"),
            dcc.Input(id='juvenile-victim-count', type='number', min=0, className="form-control")
        ]),
        dbc.Col([
            dbc.Label("Adult Victim Count"),
            dcc.Input(id='adult-victim-count', type='number', min=0, className="form-control")
        ]),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Total Individual Victims"),
            dcc.Input(id='individual-victims', type='number', min=0, className="form-control")
        ]),
        dbc.Col([
            dbc.Label("Juvenile Offender Count"),
            dcc.Input(id='juvenile-offender-count', type='number', min=0, className="form-control")
        ]),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Adult Offender Count"),
            dcc.Input(id='adult-offender-count', type='number', min=0, className="form-control")
        ]),
        dbc.Col([
            dbc.Label("Total Offender Count"),
            dcc.Input(id='total-offender-count', type='number', min=0, className="form-control")
        ]),
    ], className="mb-3"),

    dbc.Row([
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
        dbc.Col([
            dbc.Label("Division Name"),
            dcc.Dropdown(
                id='division-name',
                options=[{'label': name, 'value': name} for name in division_options],
                className="form-control",
                placeholder="Select"
            )
        ])
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Offender Ethnicity"),
            dcc.Dropdown(
                id='offender-ethnicity',
                options=[{'label': name, 'value': name} for name in ethnicity_options],
                className="form-control",
                placeholder="Select"
            )
        ]),
        dbc.Col([
            dbc.Label("Location Group"),
            dcc.Dropdown(
                id='location-group',
                options=[{"label": name, "value": name} for name in location_options],
                className="form-control",
                placeholder="Select"
            )
        ])
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
    State("juvenile-victim-count", "value"),
    State("adult-victim-count", "value"),
    State("individual-victims", "value"),
    State("juvenile-offender-count", "value"),
    State("adult-offender-count", "value"),
    State("total-offender-count", "value"),
    State("is-violent", "value"),
    State("division-name", "value"),
    State("offender-ethnicity", "value"),
    State("location-group", "value")
)
def predict_bias(n_clicks, jvc, avc, tiv, joc, aoc, toc, is_violent, division, ethnicity, location_group):
    if n_clicks is None:
        return ""

    if any(v is None for v in [jvc, avc, tiv, joc, aoc, toc, is_violent, division, ethnicity, location_group]):
        return dbc.Alert("⚠️ All fields are required to get a prediction.", color="danger")

    input_data = {
        'juvenile_victim_count': jvc,
        'adult_victim_count': avc,
        'total_individual_victims': tiv,
        'juvenile_offender_count': joc,
        'adult_offender_count': aoc,
        'total_offender_count': toc,
        'crime_type_Violent': is_violent,
    }

    # One-hot encode division
    for div in division_options:
        input_data[f'division_name_{div}'] = 1 if division == div else 0

    # One-hot encode ethnicity
    for eth in ethnicity_options:
        input_data[f'offender_ethnicity_{eth}'] = 1 if ethnicity == eth else 0

    # One-hot encode location group
    for loc in location_options:
        input_data[f"location_group_{loc}"] = 1 if location_group == loc else 0

    input_df = pd.DataFrame([input_data])
    proba = model.predict_proba(input_df)
    prediction = model.predict(input_df)[0]

    print("Proba", proba, "roba", prediction)
    cards = []
    for i, label in enumerate(LABELS):
        conf = proba[i][0, 1] * 100  # Correct indexing
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
            ], style={"backgroundColor": "#ffffff", "border": f"2px solid {border_color}"}, className="rounded-4 shadow-sm"), 
            md=3
        )
        cards.append(card)
    return cards

