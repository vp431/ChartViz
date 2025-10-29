"""
ChartViz - Modern Chart QA Explainability Tool
Advanced attention visualization for chart-based question answering models.
"""
import os
import sys
import time
import traceback
import logging
from pathlib import Path

# Set offline mode for transformers to prevent Hub downloads
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
# Additional environment variables to ensure complete offline mode
os.environ['PYTORCH_PRETRAINED_BERT_CACHE'] = './cache'
os.environ['PYTORCH_TRANSFORMERS_CACHE'] = './cache'

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, no_update, ALL
import numpy as np
import pandas as pd
from PIL import Image
import base64
import io

# Import configuration and utilities
from config import config, APP_CONFIG, UI_CONFIG
from models import ModelManager
from components import AttentionVisualizer, InteractiveHeatmap
from components.input_section import create_input_section, register_input_callbacks
from components.results_section import create_results_section, create_model_info_section, register_results_callbacks
from components.analysis_results import create_analysis_modal, register_callbacks as register_analysis_callbacks
from popup.attention_analysis_popup import AttentionAnalysisPopup
from popup import HelpPopup
from utils.model_scanner import scan_local_models, get_model_details
from utils.dataset_scanner import scan_local_datasets, get_dataset_details

# Configure logging
logging.basicConfig(
    level=getattr(logging, APP_CONFIG.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize core components
model_manager = ModelManager()
attention_visualizer = AttentionVisualizer()
interactive_heatmap = InteractiveHeatmap()
attention_analysis_popup = AttentionAnalysisPopup()
help_popup = HelpPopup()

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
        "/assets/chartviz_styles.css",
        "/assets/help_popup_styles.css",
        "/assets/attention_analysis_styles.css",
    ],
    suppress_callback_exceptions=True
)

app.title = UI_CONFIG.brand_name
server = app.server  # For deployment


def create_control_panel():
    """Create the main control panel matching OldCode layout exactly."""
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Model:", className="fw-bold mb-1", style={"fontSize": "0.9rem"}),
                    dcc.Dropdown(
                        id="model-dropdown",
                        placeholder="Select model...",
                        clearable=False,
                        searchable=False,
                        style={"zIndex": "1050", "position": "relative", "width": "100%"}
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Source:", className="fw-bold mb-1", style={"fontSize": "0.9rem"}),
                    dbc.RadioItems(
                        id="data-source-dropdown",
                        options=[
                            {"label": "Custom", "value": "custom"},
                            {"label": "Dataset", "value": "dataset"}
                        ],
                        value="custom",
                        inline=True,
                        style={"fontSize": "0.85rem"}
                    )
                ], width=2),
                
                dbc.Col([
                    html.Label("Dataset:", className="fw-bold mb-1", style={"fontSize": "0.9rem"}),
                    dcc.Dropdown(
                        id="dataset-dropdown",
                        placeholder="Choose dataset...",
                        clearable=False,
                        searchable=False,
                        disabled=True,
                        style={"zIndex": "1049", "position": "relative", "width": "100%"}
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Selected Model:", className="fw-bold mb-1", style={"fontSize": "0.9rem"}),
                    dbc.Tooltip(
                        id="model-tooltip",
                        target="selected-model-display",
                        placement="bottom",
                        style={"fontSize": "0.85rem"}
                    ),
                    html.Div(id="selected-model-display", 
                            children="None", 
                            className="text-muted selected-model-display",
                            style={
                                "fontSize": "0.8rem", 
                                "wordBreak": "break-word",
                                "cursor": "help",
                                "padding": "0.25rem 0.5rem",
                                "borderRadius": "4px",
                                "border": "1px solid transparent"
                            })
                ], width=3),
                
                dbc.Col([
                    html.Label(" ", className="fw-bold mb-1", style={"fontSize": "0.9rem"}),
                    dbc.Button([
                        html.I(className="fas fa-question-circle")
                    ], id="help-btn", color="info", size="sm", style={"width": "auto", "padding": "0.25rem 0.5rem"})
                ], width=1)
            ], align="center", style={"width": "100%"})
        ], className="py-2", style={"width": "100%"})
    ], className="mb-3 control-panel-card", style={"width": "100%"})




# Main app layout
app.layout = html.Div([
    dbc.Container([
        # Storage components
        dcc.Store(id="current-image-data"),
        dcc.Store(id="current-attention-data"),
        dcc.Store(id="model-predictions-data"),
        dcc.Store(id="dataset-samples-data"),
        html.Div(id="attention-export-status"),  # For export status messages
        
        # Interval component for periodic model status updates
        dcc.Interval(
            id="model-status-interval",
            interval=30000,  # Update every 30 seconds
            n_intervals=0
        ),
        
        # Download interface components
        
        # Control panel
        create_control_panel(),
        
        # Input section
        create_input_section(),
        
        # Analysis Results Modal
        create_analysis_modal(),
        
        # Attention Analysis Popup
        attention_analysis_popup.create_popup_modal(),
        attention_analysis_popup.create_settings_tooltip(),
        
        # Help Modal (using new help popup component)
        help_popup.create_modal()
        
    ], fluid=True)
], className="chartviz-container", id="app-container")


# Callbacks

# Data source toggle callback moved to components/input_section.py

@app.callback(
    [Output("model-dropdown", "options"),
     Output("model-dropdown", "value")],
    Input("app-container", "id")  # Trigger on page load
)
def update_model_options(_):
    """Update available models using the model scanner."""
    try:
        # Use model scanner to get available models
        available_model_ids = scan_local_models()
        model_details = get_model_details()
        
        # Create dropdown options with status indicators
        model_options = []
        for model_id in available_model_ids:
            details = model_details.get(model_id, {})
            model_name = details.get('name', model_id.title())
            
            # Check model status through model manager
            status = model_manager.get_model_status(model_id)
            
            # Add status indicators to the label
            if status["available"]:
                label = f"✓ {model_name}"
            elif status["downloaded"] and not status["implemented"]:
                label = f"⚠ {model_name} (Not Implemented)"
            elif status["implemented"] and not status["downloaded"]:
                label = f"○ {model_name} (Not Downloaded)"
            else:
                label = f"✗ {model_name} (Unavailable)"
            
            model_options.append({
                "label": label, 
                "value": model_id,
                "disabled": not status["available"]
            })
        
        if not model_options:
            model_options = [{"label": "No models available", "value": "", "disabled": True}]
        
        # Set default value to the first available model
        available_options = [opt for opt in model_options if not opt.get('disabled', False)]
        default_value = available_options[0]['value'] if available_options else None

        return model_options, default_value
        
    except Exception as e:
        logger.error(f"Error updating model options: {e}")
        error_msg = [{"label": "Error loading models", "value": "", "disabled": True}]
        return error_msg, None


# Selected model display callback with auto-loading and tooltip
@app.callback(
    [Output("selected-model-display", "children"),
     Output("selected-model-display", "style"),
     Output("model-tooltip", "children")],
    Input("model-dropdown", "value")
)
def update_selected_model_display(model_id):
    """Update the selected model display, auto-load if available, and update tooltip."""
    if not model_id:
        base_style = {
            "fontSize": "0.8rem", 
            "wordBreak": "break-word",
            "cursor": "default",
            "padding": "0.25rem 0.5rem",
            "borderRadius": "4px",
            "border": "1px solid transparent"
        }
        return "None", base_style, "No model selected"
    
    try:
        # Get model details to show a nice display name
        model_details = get_model_details()
        if model_id in model_details:
            model_name = model_details[model_id].get('name', model_id.title())
        else:
            model_name = model_id.title()
        
        # Check model status
        status = model_manager.get_model_status(model_id)
        
        # Base style for the display
        base_style = {
            "fontSize": "0.8rem", 
            "wordBreak": "break-word",
            "cursor": "help",
            "padding": "0.25rem 0.5rem",
            "borderRadius": "4px",
            "border": "1px solid transparent"
        }
        
        tooltip_content = ""
        
        if status["available"]:
            # Auto-load the model if it's available
            try:
                # Smart model loading: only clear if switching to a different model
                loaded_models = model_manager.get_loaded_models()
                
                if model_id in loaded_models and len(loaded_models) == 1:
                    # Model is already loaded and is the only one - just reuse it
                    logger.info(f"Model {model_id} is already loaded and ready for use")
                    model_manager._update_model_usage(model_id)
                elif model_id in loaded_models and len(loaded_models) > 1:
                    # Model is loaded but there are others - clear others to free memory
                    logger.info(f"Model {model_id} is loaded but clearing {len(loaded_models)-1} other model(s) to free memory")
                    other_models = [mid for mid in loaded_models if mid != model_id]
                    for other_model in other_models:
                        model_manager.unload_model(other_model)
                    model_manager._clear_cuda_memory()
                    model_manager._update_model_usage(model_id)
                elif model_id not in loaded_models:
                    # Model is not loaded - need to load it
                    if len(loaded_models) > 0:
                        logger.info(f"Switching from {loaded_models} to {model_id} - clearing memory first")
                        model_manager.clear_and_load_model(model_id)
                    else:
                        logger.info(f"Loading model: {model_id}")
                        model_manager.preload_model(model_id)
                    
                # Get usage info for tooltip
                usage_info = model_manager.get_model_usage_info()
                if model_id in usage_info:
                    info = usage_info[model_id]
                    tooltip_content = html.Div([
                        html.Div([
                            html.I(className="fas fa-check-circle text-success me-2"),
                            html.Strong(f"{model_name}")
                        ], className="mb-2"),
                        html.Div([
                            html.Small([
                                "🚀 Status: Loaded & Ready", html.Br(),
                                f"⏰ Last used: {info['last_used']}", html.Br(),
                                f"🔄 Auto-unload in: {info['auto_unload_in']} min", html.Br(),
                                "💡 Will auto-unload after 10 min of inactivity"
                            ])
                        ])
                    ])
                else:
                    tooltip_content = html.Div([
                        html.Div([
                            html.I(className="fas fa-check-circle text-success me-2"),
                            html.Strong(f"{model_name}")
                        ], className="mb-2"),
                        html.Small("🚀 Status: Loaded & Ready")
                    ])
                
                # Style for loaded model
                base_style.update({
                    "backgroundColor": "#d1edff",
                    "border": "1px solid #0ea5e9",
                    "color": "#0369a1"
                })
                
            except Exception as load_error:
                logger.error(f"Failed to auto-load model {model_id}: {load_error}")
                tooltip_content = html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                        html.Strong(f"{model_name}")
                    ], className="mb-2"),
                    html.Small([
                        "❌ Status: Failed to Load", html.Br(),
                        f"Error: {str(load_error)}"
                    ])
                ])
                
                # Style for error
                base_style.update({
                    "backgroundColor": "#fef2f2",
                    "border": "1px solid #f87171",
                    "color": "#dc2626"
                })
        else:
            # Model not available - show appropriate status in tooltip
            if status["downloaded"] and not status["implemented"]:
                tooltip_content = html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                        html.Strong(f"{model_name}")
                    ], className="mb-2"),
                    html.Small([
                        "⚠️ Status: Downloaded but Not Implemented", html.Br(),
                        "Please ensure the model class is properly registered."
                    ])
                ])
                # Style for warning
                base_style.update({
                    "backgroundColor": "#fefce8",
                    "border": "1px solid #eab308",
                    "color": "#a16207"
                })
                
            elif status["implemented"] and not status["downloaded"]:
                tooltip_content = html.Div([
                    html.Div([
                        html.I(className="fas fa-download text-info me-2"),
                        html.Strong(f"{model_name}")
                    ], className="mb-2"),
                    html.Small([
                        "📥 Status: Implemented but Not Downloaded", html.Br(),
                        "Please download the model files first."
                    ])
                ])
                # Style for info
                base_style.update({
                    "backgroundColor": "#f0f9ff",
                    "border": "1px solid #0ea5e9",
                    "color": "#0369a1"
                })
                
            elif not status["implemented"] and not status["downloaded"]:
                tooltip_content = html.Div([
                    html.Div([
                        html.I(className="fas fa-times-circle text-danger me-2"),
                        html.Strong(f"{model_name}")
                    ], className="mb-2"),
                    html.Small([
                        "❌ Status: Not Available", html.Br(),
                        "Model is neither implemented nor downloaded."
                    ])
                ])
                # Style for error
                base_style.update({
                    "backgroundColor": "#fef2f2",
                    "border": "1px solid #f87171",
                    "color": "#dc2626"
                })
        
        return model_name, base_style, tooltip_content
        
    except Exception as e:
        logger.error(f"Error updating selected model display: {e}")
        error_style = base_style.copy()
        error_style.update({
            "backgroundColor": "#fef2f2",
            "border": "1px solid #f87171",
            "color": "#dc2626"
        })
        error_tooltip = html.Div([
            html.I(className="fas fa-exclamation-circle text-danger me-2"),
            html.Small(f"Error: {str(e)}")
        ])
        return model_id if model_id else "None", error_style, error_tooltip


@app.callback(
    [Output("dataset-dropdown", "options"),
     Output("dataset-dropdown", "value")],
    [Input("data-source-dropdown", "value"),
     Input("app-container", "id")]  # Trigger on page load
)
def update_dataset_options(data_source, app_container_id):
    """Update available datasets using the dataset scanner."""
    if data_source != "dataset":
        return [], None
    
    try:
        # Use dataset scanner to get available datasets
        available_dataset_ids = scan_local_datasets()
        dataset_details = get_dataset_details()
        
        # Create dropdown options
        dataset_options = []
        for dataset_id in available_dataset_ids:
            details = dataset_details.get(dataset_id, {})
            dataset_name = details.get('name', dataset_id.title())
            dataset_options.append({"label": dataset_name, "value": dataset_id})
        
        if not dataset_options:
            dataset_options = [{"label": "No datasets available", "value": "", "disabled": True}]
            return dataset_options, None
        
        # Set default value to the first available dataset
        default_value = available_dataset_ids[0] if available_dataset_ids else None
        
        return dataset_options, default_value
        
    except Exception as e:
        logger.error(f"Error updating dataset options: {e}")
        error_msg = [{"label": "Error loading datasets", "value": "", "disabled": True}]
        return error_msg, None

# load_dataset_and_first_sample callback moved to components/input_section.py

# update_sample_view callback moved to components/input_section.py






# Callback registrations moved to end of file


# Clear attention data when model changes (CRITICAL BUG FIX)
@app.callback(
    Output("current-attention-data", "data", allow_duplicate=True),
    Input("model-dropdown", "value"),
    prevent_initial_call=True
)
def clear_attention_on_model_change(model_id):
    """Clear attention data when model changes to prevent showing stale heatmaps."""
    logger.info(f"🔄 Model changed to {model_id}, clearing attention data to prevent stale heatmaps")
    return None

# Periodic tooltip update for real-time model usage info
@app.callback(
    Output("model-tooltip", "children", allow_duplicate=True),
    [Input("model-status-interval", "n_intervals")],
    [State("model-dropdown", "value")],
    prevent_initial_call=True
)
def update_model_tooltip_periodic(n_intervals, model_id):
    """Periodically update the model tooltip with current usage info."""
    if not model_id:
        return no_update
    
    try:
        # Get model details
        model_details = get_model_details()
        if model_id in model_details:
            model_name = model_details[model_id].get('name', model_id.title())
        else:
            model_name = model_id.title()
        
        # Check if model is currently loaded
        if model_id in model_manager.get_loaded_models():
            usage_info = model_manager.get_model_usage_info()
            if model_id in usage_info:
                info = usage_info[model_id]
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-check-circle text-success me-2"),
                        html.Strong(f"{model_name}")
                    ], className="mb-2"),
                    html.Div([
                        html.Small([
                            "🚀 Status: Loaded & Ready", html.Br(),
                            f"⏰ Last used: {info['last_used']}", html.Br(),
                            f"🔄 Auto-unload in: {info['auto_unload_in']} min", html.Br(),
                            "💡 Will auto-unload after 10 min of inactivity"
                        ])
                    ])
                ])
        
        # If not loaded, don't update (preserve original tooltip)
        return no_update
        
    except Exception as e:
        logger.error(f"Error updating tooltip: {e}")
        return no_update




# Register all callbacks
register_input_callbacks(app, model_manager)
register_results_callbacks(app, model_manager) 
register_analysis_callbacks(app, model_manager)
attention_analysis_popup.register_callbacks(app)
help_popup.register_callbacks(app)

if __name__ == "__main__":
    logger.info("Starting ChartViz application...")
    logger.info(f"Debug mode: {APP_CONFIG.debug_mode}")
    logger.info(f"Host: {APP_CONFIG.host}:{APP_CONFIG.port}")
    
    app.run(
        debug=APP_CONFIG.debug_mode,
        host=APP_CONFIG.host,
        port=APP_CONFIG.port
    )




