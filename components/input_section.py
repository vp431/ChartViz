"""
Input Section Component - Handles chart image uploads, dataset selection, and question input.
Extracted from app.py for better code organization.
"""
import base64
import io
import logging
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, no_update
from PIL import Image

from components.chart_image import create_chart_image_layout, register_callbacks as register_chart_image_callbacks, encode_image
from components.question_analysis import create_question_analysis_layout, register_callbacks as register_question_analysis_callbacks
from utils.dataset_scanner import get_samples_for_dataset

logger = logging.getLogger(__name__)


def create_input_section():
    """Create the dynamic input section for chart and question."""
    return html.Div([
        # Main input area (always visible)
        dbc.Row([
            dbc.Col([
                create_chart_image_layout()
            ], width=6),
            
            dbc.Col([
                create_question_analysis_layout()
            ], width=6)
        ], className="g-3 mb-4")
    ])


def register_input_section_callbacks(app):
    """Register all callbacks related to the input section."""
    
    # Register chart image callbacks
    register_chart_image_callbacks(app)
    
    # Data source toggle callback
    @app.callback(
        [Output("upload-area", "style"),
         Output("dataset-image-area", "style"),
         Output("dataset-dropdown", "disabled"),
         Output("dataset-sample-input", "disabled", allow_duplicate=True)],
        Input("data-source-dropdown", "value"),
        prevent_initial_call=True
    )
    def toggle_data_source(data_source):
        """Toggle between custom upload and dataset selection."""
        if data_source == "custom":
            return {"display": "block"}, {"display": "none"}, True, True
        elif data_source == "dataset":
            return {"display": "none"}, {"display": "block"}, False, False
        else:
            return {"display": "none"}, {"display": "none"}, True, True

    @app.callback(
        [Output("dataset-samples-data", "data"),
         Output("dataset-sample-input", "max"),
         Output("dataset-sample-input", "value"),
         Output("dataset-sample-input", "disabled", allow_duplicate=True),
         Output("dataset-sample-count-text", "children"),
         Output("chart-preview", "children", allow_duplicate=True),
         Output("question-input", "value", allow_duplicate=True),
         Output("current-image-data", "data", allow_duplicate=True)],
        Input("dataset-dropdown", "value"),
        State("data-source-dropdown", "value"),
        prevent_initial_call=True
    )
    def load_dataset_and_first_sample(dataset_id, data_source):
        """
        Load dataset samples and display the first sample.
        Extracted from app.py for better organization.
        """
        if not dataset_id or data_source != 'dataset':
            return None, 1, 1, True, "", "", "", None

        try:
            samples = get_samples_for_dataset(dataset_id, limit=250)
            if not samples:
                return None, 1, 1, True, "No samples found.", "", "", None
            
            num_samples = len(samples)
            
            samples_data = [
                {
                    "id": s.id,
                    "question": s.question,
                    "image_path": str(s.image_path) if s.image_path else None
                } for s in samples
            ]
            
            first_sample = samples[0]
            first_question = first_sample.question
            
            preview = ""
            image_data = None
            if first_sample.image_path and Path(first_sample.image_path).exists():
                encoded_image = encode_image(first_sample.image_path)
                if encoded_image:
                    preview = html.Div([
                        html.Img(src=encoded_image, style={"maxWidth": "100%", "maxHeight": "300px", "borderRadius": "8px"}),
                        html.P(f"Image: {Path(first_sample.image_path).name}", className="text-muted mt-2 mb-0")
                    ])
                    image_data = {
                        "contents": encoded_image,
                        "filename": Path(first_sample.image_path).name,
                    }
            
            return samples_data, num_samples, 1, False, f"of {num_samples}", preview, first_question, image_data

        except Exception as e:
            logger.error(f"Error loading dataset samples: {e}")
            return None, 1, 1, True, "Error.", dbc.Alert(f"Error: {e}", color="danger"), "", None

    @app.callback(
        [Output("chart-preview", "children", allow_duplicate=True),
         Output("question-input", "value", allow_duplicate=True),
         Output("current-image-data", "data", allow_duplicate=True)],
        Input("dataset-sample-input", "value"),
        State("dataset-samples-data", "data"),
        prevent_initial_call=True
    )
    def update_sample_view(sample_index, samples_data):
        """
        Update the displayed sample when user changes sample selection.
        Extracted from app.py for better organization.
        """
        if sample_index is None or not samples_data:
            return no_update, no_update, no_update

        try:
            sample_index = int(sample_index) - 1
            if 0 <= sample_index < len(samples_data):
                sample = samples_data[sample_index]
                question = sample["question"]
                
                preview = ""
                image_data = None
                
                # Debug logging
                logger.info(f"Loading sample {sample_index + 1}: {sample}")
                
                if sample["image_path"]:
                    image_path = Path(sample["image_path"])
                    logger.info(f"Checking image path: {image_path}, exists: {image_path.exists()}")
                    
                    if image_path.exists():
                        encoded_image = encode_image(sample["image_path"])
                        if encoded_image:
                            preview = html.Div([
                                html.Div([
                                    html.Img(src=encoded_image, style={"maxWidth": "100%", "maxHeight": "300px", "borderRadius": "8px"}),
                                    dbc.Button(
                                        html.I(className="fas fa-expand"),
                                        id="fullscreen-btn",
                                        color="primary",
                                        size="sm",
                                        className="position-absolute",
                                        style={
                                            "top": "10px",
                                            "right": "10px",
                                            "opacity": "0.8",
                                            "zIndex": "10"
                                        },
                                        title="View Fullscreen"
                                    )
                                ], style={"position": "relative"}),
                                html.P(f"Image: {image_path.name}", className="text-muted mt-2 mb-0 text-center")
                            ])
                            image_data = {
                                "contents": encoded_image,
                                "filename": image_path.name,
                            }
                            logger.info(f"Successfully loaded image for sample {sample_index + 1}")
                        else:
                            logger.error(f"Failed to encode image: {sample['image_path']}")
                            preview = dbc.Alert(f"Failed to load image: {image_path.name}", color="warning")
                    else:
                        logger.error(f"Image file not found: {image_path}")
                        preview = dbc.Alert(f"Image not found: {image_path.name}", color="warning")
                else:
                    logger.warning(f"No image path for sample {sample_index + 1}")
                    preview = dbc.Alert("No image available for this sample", color="info")
                
                return preview, question, image_data
            
            return no_update, no_update, no_update

        except Exception as e:
            logger.error(f"Error updating sample view: {e}")
            import traceback
            traceback.print_exc()
            return dbc.Alert(f"Error loading sample: {str(e)}", color="danger"), "", None


def register_input_callbacks(app, model_manager):
    """
    Register callbacks for input section that depend on model_manager.
    This is a wrapper to maintain backward compatibility.
    """
    # Register the input section callbacks
    register_input_section_callbacks(app)
    
    # Register question analysis callbacks (these need model_manager)
    register_question_analysis_callbacks(app, model_manager)

