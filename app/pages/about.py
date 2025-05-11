import dash
from dash import html

dash.register_page(__name__, path="/about")

layout = html.Div([
    # Section 1: Introduction
    html.Section([
        html.H2("About This App", className="text-center mb-4"),
        html.P("Welcome to the U.S. Hate Crime Analysis dashboard. This tool allows you to explore crime data across various states, view trends, and gain insights into different crime categories.", 
               className="lead text-center"),
    ], className="section p-5"),

    # Section 2: Team
    html.Section([
        html.H3("Meet the Team", className="text-center mb-4"),
        html.Div([
            html.Div([
                html.Img(src="assets/Patsakorn.jpg", className="img-fluid rounded-circle shadow-lg", style={"width": "150px", "height": "150px", "margin-bottom": "10px"}),
                html.P("Patsakorn Tangkachaiyanunt", className="h5 text-center mb-2"),
                html.P("Data Scientist", className="text-muted text-center"),
            ], className="col-md-6 d-flex flex-column align-items-center mb-4"),
            
            html.Div([
                html.Img(src="assets/shree.jpg", className="img-fluid rounded-circle shadow-lg", style={"width": "150px", "height": "150px", "margin-bottom": "10px"}),
                html.P("Shreeyukta Pradhanang", className="h5 text-center mb-2"),
                html.P("Data Analyst & Web Developer", className="text-muted text-center"),
            ], className="col-md-6 d-flex flex-column align-items-center mb-4"),
        ], className="row justify-content-center")
    ], className="section p-5 bg-light"),

    html.Section([
        html.H3("Find the Dataset and GitHub Repository", className="text-center mb-4"),
        html.Div([
            html.A(
                "View the Dataset",
                href="https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads", 
                target="_blank", 
                className="btn btn-outline-primary me-2" 
            ),
            html.A(
                "Visit GitHub Repository",
                href="https://github.com/LovelyPrinceGI/crime-prediction-ml-v2",  
                target="_blank",
                className="btn btn-outline-secondary"  
            ),
        ], className="text-center")
    ], className="section p-5"),
    html.Section([
        html.H3("Additional Information", className="text-center mb-4"),
        html.P("This project aims to provide insights into hate crime trends in the U.S. using a dataset of reported incidents. The data is sourced from public records and provides valuable analysis for policymakers and law enforcement."),
    ], className="section p-5"),
    
], className="container")
