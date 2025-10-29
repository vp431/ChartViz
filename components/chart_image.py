"""
Component for handling chart image uploads and display.
"""
import base64
import io
import logging
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback
from PIL import Image

logger = logging.getLogger(__name__)


def encode_image(image_path):
    """Encode image to base64 for display."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None


def create_chart_image_layout():
    """Create the layout for the chart image upload and preview area."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Chart Image", className="mb-0")
        ]),
        dbc.CardBody([
            # Upload area (for custom)
            html.Div([
                dcc.Upload(
                    id="chart-upload",
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt", style={"fontSize": "2rem", "marginBottom": "1rem"}),
                        html.Div("Click to Upload Chart", style={"fontWeight": "500"}),
                        html.Div("PNG, JPG, JPEG supported", style={"fontSize": "0.8rem", "color": "#666"})
                    ], style={"textAlign": "center", "padding": "2rem"}),
                    style={
                        "border": "2px dashed #ccc",
                        "borderRadius": "8px",
                        "cursor": "pointer"
                    },
                    multiple=False,
                    accept="image/*"
                )
            ], id="upload-area", style={"display": "none"}),

            # Chart preview area (image on top)
            html.Div([
                html.Div(id="chart-preview", className="mb-3"),
                
                # Fullscreen modal for image
                dbc.Modal([
                    dbc.ModalHeader([
                        dbc.ModalTitle("Chart Image")
                    ]),
                    dbc.ModalBody([
                        html.Div(id="fullscreen-image-content")
                    ])
                ], id="fullscreen-modal", size="xl", is_open=False, centered=True),
            ]),

            # Dataset controls (centered below image)
            html.Div([
                dbc.Row([
                    dbc.Col(width=2),  # Left spacer
                    dbc.Col([
                        html.Div([
                            html.Label("Select Image:", className="fw-bold text-center d-block mb-2", style={"fontSize": "0.9rem"}),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Input(
                                        id="dataset-sample-input",
                                        type="number",
                                        min=1,
                                        step=1,
                                        value=1,
                                        disabled=True,
                                        className="form-control text-center"
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Div(id="dataset-sample-count-text", className="text-center", style={"paddingTop": "0.5rem", "fontSize": "0.9rem"})
                                ], width=8)
                            ], justify="center", align="center")
                        ], className="text-center")
                    ], width=8),
                    dbc.Col(width=2)  # Right spacer
                ], justify="center")
            ], id="dataset-image-area", style={"display": "none", "marginTop": "1rem"})
        ])
    ])


def register_callbacks(app):
    """Register callbacks for the chart image component."""
    @app.callback(
        [Output("chart-preview", "children"),
         Output("current-image-data", "data"),
         Output("upload-area", "style", allow_duplicate=True)],
        Input("chart-upload", "contents"),
        State("chart-upload", "filename"),
        prevent_initial_call=True
    )
    def handle_chart_upload(contents, filename):
        """Handle chart image upload."""
        if not contents:
            return "", None, {"display": "block"}

        try:
            # Parse uploaded image
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            # Create PIL image
            image = Image.open(io.BytesIO(decoded))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Store image data
            image_data = {
                "contents": contents,
                "filename": filename,
                "size": image.size
            }

            # Create preview with fullscreen button
            preview = html.Div([
                html.Div([
                    html.Img(
                        src=contents,
                        style={
                            "maxWidth": "100%",
                            "maxHeight": "300px",
                            "borderRadius": "8px",
                            "boxShadow": "0 2px 8px rgba(0,0,0,0.1)"
                        }
                    ),
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
                html.P(
                    f"Uploaded: {filename} ({image.size[0]}×{image.size[1]})",
                    className="text-muted mt-2 mb-0 text-center"
                )
            ])

            # Hide upload area after successful upload
            return preview, image_data, {"display": "none"}

        except Exception as e:
            logger.error(f"Error processing uploaded image: {e}")
            return dbc.Alert(f"Error processing image: {str(e)}", color="danger"), None, {"display": "block"}

    @app.callback(
        [Output("fullscreen-modal", "is_open"),
         Output("fullscreen-image-content", "children")],
        Input("fullscreen-btn", "n_clicks"),
        State("current-image-data", "data"),
        prevent_initial_call=True
    )
    def open_fullscreen_modal(fullscreen_clicks, image_data):
        """Open the fullscreen image modal."""
        if fullscreen_clicks and image_data:
            fullscreen_content = html.Img(
                src=image_data["contents"],
                style={
                    "width": "100%",
                    "height": "auto",
                    "maxHeight": "80vh",
                    "objectFit": "contain"
                }
            )
            return True, fullscreen_content
        
        return False, ""
