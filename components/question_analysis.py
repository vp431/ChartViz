"""
Component for handling question input and analysis configuration.
"""
import logging
import numpy as np

import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback, no_update
import dash

logger = logging.getLogger(__name__)


# Note: _serialize_attention_data function moved to results_section.py
# to avoid code duplication


def create_question_analysis_layout():
    """Create the layout for the question and analysis configuration area."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Question & Analysis", className="mb-0")
        ]),
        dbc.CardBody([
            # Compact Question Section
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Your Question:", className="form-label fw-bold"),
                        dbc.Textarea(
                            id="question-input",
                            placeholder="Enter your question about the chart...",
                            rows=2,
                            className="mb-2"
                        ),
                    ], width=8),
                    dbc.Col([
                        html.Label("Examples:", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id="example-questions",
                            options=[
                                {"label": "What is the highest bar?", "value": "What is the highest bar?"},
                                {"label": "Which category has the lowest value?", "value": "Which category has the lowest value?"},
                                {"label": "What is the trend over time?", "value": "What is the trend over time?"},
                                {"label": "How many data points are there?", "value": "How many data points are there?"},
                                {"label": "What is the total value?", "value": "What is the total value?"}
                            ],
                            placeholder="Select example...",
                            searchable=False,
                            className="mb-2"
                        ),
                    ], width=4)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-question-circle me-2"), "Ask Question"], 
                            id="ask-question-btn", 
                            color="success", 
                            className="w-100"
                        ),
                    ], width=6),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-expand me-2"), "Advanced Analysis"], 
                            id="open-attention-analysis", 
                            color="primary", 
                            outline=True,
                            className="w-100"
                        ),
                    ], width=6)
                ], className="g-2")
            ], id="question-section"),
            
            # Compact Answer Section (appears after asking question)
            html.Div([
                html.Hr(className="my-3"),
                dbc.Alert([
                    html.H6("Model Answer", className="alert-heading mb-2"),
                    html.Div(id="model-answer-content"),
                    html.Div(id="answer-verification")
                ], color="success", className="mb-0", style={"display": "none"}, id="answer-alert")
            ], id="answer-section", style={"display": "none"}),
            
        ], className="p-3")
    ], className="h-100")


def register_callbacks(app, model_manager):
    """Register callbacks for the question analysis component."""
    @app.callback(
        Output("question-input", "value", allow_duplicate=True),
        Input("example-questions", "value"),
        prevent_initial_call=True
    )
    def update_question_from_example(example_question):
        """Update question input from example selection."""
        return example_question or ""
    
    # Note: The ask_question callback has been moved to results_section.py
    # to avoid duplicate callback registration conflicts.
            
