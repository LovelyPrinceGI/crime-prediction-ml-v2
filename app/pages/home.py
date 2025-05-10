import dash
import pandas as pd
import plotly.express as px
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc


dash.register_page(__name__, path="/")

df = pd.read_csv("datasets/updated_hate_crime.csv")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["state_abbr"] = df["state_abbr"].replace("NB", "NE")

year_options = [{"label": str(int(y)), "value": int(y)} for y in sorted(df["year"].dropna().unique())]
bias_options = [{"label": b, "value": b} for b in sorted(df["bias_desc"].dropna().unique())]

# layout = html.Div([
#     html.H2("🗺️ U.S. Hate Crime Map", className="mb-3 text-center"),

#     html.Div([
#         html.Div([
#             html.Label("Filter by Year:"),
#             dcc.Dropdown(
#                 options=year_options,
#                 value=None,
#                 id="year-dropdown",
#                 placeholder="Select a year (optional)",
#                 clearable=True,
#             )
#         ], className="col-md-6"),

#         html.Div([
#             html.Label("Filter by Bias Type:"),
#             dcc.Dropdown(
#                 options=bias_options,
#                 value=None,
#                 id="bias-dropdown",
#                 placeholder="Select a bias type (optional)",
#                 clearable=True,
#             )
#         ], className="col-md-6")
#     ], className="row g-3 mb-4"),

#     # 🧠 You missed a comma here 👇
#     dcc.Graph(id="us-choropleth"),

#     html.Hr(),

#     dbc.Row([
#         dbc.Col(
#             dbc.Card(
#                 dbc.CardBody([
#                     html.H5("Total Incidents", className="card-title"),
#                     html.H2(id="kpi-total", className="card-text")
#                 ]),
#             className="shadow-sm text-center h-100"),
#         md=4),
        
#         dbc.Col(
#             dbc.Card(
#                 dbc.CardBody([
#                     html.H5("Most Targeted Bias", className="card-title"),
#                     html.H4(id="kpi-bias", className="card-text")
#                 ]),
#             className="shadow-sm text-center h-100"),
#         md=4),

#         dbc.Col(
#             dbc.Card(
#                 dbc.CardBody([
#                     html.H5("Peak Year", className="card-title"),
#                     html.H4(id="kpi-year", className="card-text")
#                 ]),
#             className="shadow-sm text-center h-100"),
#         md=4),
#     ], className="mb-5"),

#     html.Hr(),

#     html.Div([
#         html.H4("Top 5 Offense Types", className="mb-3 text-center"),
#         dcc.Graph(id="top-offenses-bar")
#     ], className="mb-5")


# ], className="container")

layout = html.Div([
    html.H2("📊 U.S. Hate Crime Dashboard", className="mb-2 text-center fw-bold"),

    # 🔹 SECTION 1: Map
    html.Div([
        html.H4("🗺️ Hate Crime Distribution Map", className="text-dark mb-4"),
        html.Div([
            html.Div([
                html.Label("Filter by Year:", className="text-dark"),
                dcc.Dropdown(
                    options=year_options,
                    value=None,
                    id="year-dropdown",
                    placeholder="Select a year (optional)",
                    clearable=True,
                )
            ], className="col-md-6"),

            html.Div([
                html.Label("Filter by Bias Type:", className="text-dark"),
                dcc.Dropdown(
                    options=bias_options,
                    value=None,
                    id="bias-dropdown",
                    placeholder="Select a bias type (optional)",
                    clearable=True,
                )
            ], className="col-md-6")
        ], className="row g-3 mb-4"),

        dcc.Graph(id="us-choropleth")
    ], className="mb-5"),

    html.Hr(),

    # 🔹 SECTION 2: KPI Cards
    html.Div([
        html.H4("📌 Key Metrics", className="text-dark mb-4"),    
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Total Incidents", className="card-title text-uppercase font-weight-bold"),
                        html.H2(id="kpi-total", className="card-text display-3 "),
                    ]),
                    className="shadow-lg rounded-lg border border-light mb-4",
                    style={"minHeight": "200px"}
                ),
            md=4),

            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Most Targeted Bias", className="card-title text-uppercase font-weight-bold"),
                        html.H4(id="kpi-bias", className="card-text display-3 ", style={"word-wrap": "break-word", "font-size": "2.2rem"}),
                    ]),
                    className="shadow-lg rounded-lg border border-light mb-4",
                    style={"minHeight": "200px"}
                ),
            md=4),

            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Peak Year", className="card-title text-uppercase font-weight-bold"),
                        html.H4(id="kpi-year", className="card-text display-4 "),
                    ]),
                    className="shadow-lg rounded-lg border border-light mb-4",
                    style={"minHeight": "200px"}
                ),
            md=4),
        ], className="mb-5 row g-4"),
    ]),
    

    html.Hr(),

    # 🔹 SECTION 3: Top Offenses Bar Chart
    html.Div([
        html.H4("🔎 Top 5 Offense Types", className="text-dark mb-4 mt-4"),
        dcc.Graph(id="top-offenses-bar")
    ], className="mb-5")

], className="container")


from dash import callback

@callback(
    Output("us-choropleth", "figure"),
    Input("year-dropdown", "value"),
    Input("bias-dropdown", "value")
)
def update_map(selected_year, selected_bias):
    dff = df.copy()

    if selected_year:
        dff = dff[dff["year"] == selected_year]
    if selected_bias:
        dff = dff[dff["bias_desc"] == selected_bias]

    # Group by state
    state_counts = dff["state_abbr"].value_counts().reset_index()
    state_counts.columns = ["state_abbr", "incident_count"]
    state_counts = state_counts.merge(df[['state_abbr', 'state_name']].drop_duplicates(), on='state_abbr', how='left')


    # Plotly choropleth
    fig = px.choropleth(
        state_counts,
        locations="state_abbr",
        locationmode="USA-states",
        color="incident_count",
        hover_name="state_name",
        hover_data={"incident_count": True, "state_abbr": False}, 
        scope="usa",
        color_continuous_scale="OrRd",
        title="Hate Crime Incidents by State",
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>" +  
                    "Incident Count: %{customdata[1]}<br>" +  
                    "<extra></extra>",  
        customdata=state_counts[['state_name', 'incident_count']].values, 
    )

    fig.update_layout(
        paper_bgcolor="#f8f9fa",   
        plot_bgcolor="#f8f9fa",  
        geo=dict(bgcolor="#f8f9fa"), 
        margin=dict(l=0, r=0, t=50, b=0),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            bordercolor="#000",
            align="left",
        )
    )

    return fig

@callback(
    Output("kpi-total", "children"),
    Output("kpi-bias", "children"),
    Output("kpi-year", "children"),
    Input("year-dropdown", "value"),
    Input("bias-dropdown", "value")
)
def update_kpis(selected_year, selected_bias):
    dff = df.copy()

    if selected_year:
        dff = dff[dff["year"] == selected_year]
    if selected_bias:
        dff = dff[dff["bias_desc"] == selected_bias]

    total = len(dff)

    most_common_bias = (
        dff["bias_desc"].mode().iloc[0]
        if not dff["bias_desc"].dropna().empty
        else "N/A"
    )

    peak_year = (
        df["year"].value_counts().idxmax()
        if not df["year"].dropna().empty
        else "N/A"
    )

    return f"{total:,}", most_common_bias, int(peak_year)

@callback(
    Output("top-offenses-bar", "figure"),
    Input("year-dropdown", "value"),
    Input("bias-dropdown", "value")
)
def update_top_offenses(selected_year, selected_bias):
    dff = df.copy()

    if selected_year:
        dff = dff[dff["year"] == selected_year]
    if selected_bias:
        dff = dff[dff["bias_desc"] == selected_bias]

    top_offenses = (
            dff["offense_name"]
            .value_counts()
            .nlargest(5)
            .reset_index()
        )
    top_offenses.columns = ["offense", "count"]

    import plotly.express as px

    fig = px.bar(
        top_offenses,
        x="offense",
        y="count",
        text="count",
        labels={"offense": "Offense Type", "count": "Number of Incidents"}
        
    )

    fig.update_traces(
        textposition="outside",
        marker_line_width=1.5,
        marker_line_color="black", 
        marker_color="#800000"
    )

    fig.update_layout(
        yaxis_title="Number of Incidents",
        xaxis_title="Offense Type",
        xaxis=dict(
        ticklabelposition="outside",    
        ),
        yaxis=dict(
            title_standoff=20         # ⬅ adds space to the left of y-axis label
        ),
        title_font_size=20,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        margin=dict(t=50, l=30, r=30, b=50),
        font=dict(family="Arial", size=14),
        hoverlabel=dict(
        bgcolor="white",
        font_size=14,
        font_family="Arial",
        bordercolor="#000",
        align="left"
        )
    )

    return fig


