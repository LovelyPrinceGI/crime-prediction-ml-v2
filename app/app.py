import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Hate Crime Dashboard", className="text-center mt-3"),
    dcc.Location(id="url"),
    dash.page_container  # Automatically handles multipage routing
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True)
