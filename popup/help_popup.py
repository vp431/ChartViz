"""
Help popup component with models, datasets, and tool information.
Includes integrated model and dataset download functionality.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, ALL
import os
from pathlib import Path
from typing import Dict, List

from config import config


class HelpPopup:
    """Unified help popup with three main sections."""
    
    def __init__(self):
        """Initialize the help popup component."""
        self.modal_id = "help-modal"
        self.tabs_id = "help-tabs"
    
    def create_modal(self):
        """Create the help modal with tabbed content."""
        return dbc.Modal([
            dbc.ModalHeader([
                html.H4("ChartViz Help", className="mb-0")
            ], close_button=True),
            dbc.ModalBody([
                dbc.Tabs([
                    dbc.Tab(
                        label="Models",
                        tab_id="models-tab",
                        children=self._create_models_section()
                    ),
                    dbc.Tab(
                        label="Datasets", 
                        tab_id="datasets-tab",
                        children=self._create_datasets_section()
                    ),
                    dbc.Tab(
                        label="About Tool",
                        tab_id="about-tab", 
                        children=self._create_about_section()
                    )
                ], id=self.tabs_id, active_tab="models-tab")
            ], style={"minHeight": "400px"}),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-help-modal", color="primary")
            ])
        ], id=self.modal_id, size="xl", centered=True)
    
    def _create_models_section(self):
        """Create the models section content with integrated download functionality."""
        return html.Div([
            html.H5("Model Management", className="mb-4"),
            
            # System Requirements Check
            dbc.Card([
                dbc.CardHeader("System Requirements"),
                dbc.CardBody([
                    html.Div(id="system-requirements-status", children=[
                        html.I(className="fas fa-spinner fa-spin me-2"),
                        "Checking system requirements..."
                    ]),
                    dbc.Button([
                        html.I(className="fas fa-sync-alt me-2"),
                        "Recheck"
                    ], id="recheck-system-btn", color="outline-primary", size="sm", className="mt-2")
                ])
            ], className="mb-4"),

            # Models Table
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("Available Models", className="me-2"),
                        dbc.Button([
                            html.I(className="fas fa-sync-alt")
                        ], id="refresh-models-btn", color="outline-secondary", size="sm",
                           title="Refresh model status")
                    ], className="d-flex justify-content-between align-items-center")
                ]),
                dbc.CardBody([
                    html.Div(id="models-table", children=[
                        html.I(className="fas fa-spinner fa-spin me-2"),
                        "Loading available models..."
                    ])
                ])
            ], className="mb-4"),

            # Download Options
            dbc.Card([
                dbc.CardHeader("Download Options"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-download me-2"),
                                "Download Selected Models"
                            ], id="download-selected-models-btn", color="primary", className="w-100 mb-2",
                               disabled=True),
                            dbc.Button([
                                html.I(className="fas fa-trash-alt me-2"),
                                "Cleanup Models"
                            ], id="cleanup-models-btn", color="danger", className="w-100")
                        ], width=12, md=6),

                        dbc.Col([
                            html.Div([
                                html.Label("Download Location:", className="form-label"),
                                dcc.Input(
                                    id="download-location-input",
                                    type="text",
                                    value=str(config.paths.local_models_dir),
                                    className="form-control",
                                    readOnly=True
                                ),
                                html.Small("Models will be saved to this directory",
                                          className="text-muted mt-1 d-block")
                            ])
                        ], width=12, md=6)
                    ])
                ])
            ], className="mb-4"),

            # Download Progress
            dbc.Collapse([
                dbc.Card([
                    dbc.CardHeader("Download Progress"),
                    dbc.CardBody([
                        html.Div(id="model-download-progress", children=[
                            dbc.Progress(value=0, id="model-download-progress-bar", className="mb-2"),
                            html.Div(id="model-download-progress-text", children="Ready to download")
                        ]),
                        html.Div(id="model-download-status", className="mt-2"),
                        html.Pre(id="model-download-log", className="mt-2 p-2 bg-light border",
                                style={"maxHeight": "200px", "overflowY": "auto", "fontSize": "0.8rem"})
                    ])
                ])
            ], id="model-download-progress-collapse", is_open=False),
            
            html.Hr(),
            
            # Custom Models Integration Info
            dbc.Collapse([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Integrating Custom Models", className="mb-2"),
                        html.P([
                            "To integrate your own models:",
                            html.Br(),
                            "1. Place model files in the LocalModels directory",
                            html.Br(), 
                            "2. Ensure your model follows the BaseChartQAModel interface",
                            html.Br(),
                            "3. Add model configuration to config.py",
                            html.Br(),
                            "4. Restart the application to detect new models"
                        ], className="mb-0", style={"fontSize": "0.9rem"})
                    ])
                ], color="light")
            ], id="custom-models-collapse", is_open=False),
            
            dbc.Button([
                html.I(className="fas fa-chevron-down me-2"),
                "Custom Model Integration"
            ], id="toggle-custom-models", color="light", outline=True, size="sm", className="mt-2")
        ])
    
    def _create_datasets_section(self):
        """Create the datasets section content with integrated download functionality."""
        return html.Div([
            html.H5("Dataset Management", className="mb-4"),
            
            # Storage Information
            dbc.Card([
                dbc.CardHeader("Storage Information"),
                dbc.CardBody([
                    html.Div(id="storage-info", children=[
                        html.I(className="fas fa-spinner fa-spin me-2"),
                        "Checking available storage..."
                    ])
                ])
            ], className="mb-4"),

            # Datasets Table
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("Available Datasets", className="me-2"),
                        dbc.Button([
                            html.I(className="fas fa-sync-alt")
                        ], id="refresh-datasets-btn", color="outline-secondary", size="sm",
                           title="Refresh dataset status")
                    ], className="d-flex justify-content-between align-items-center")
                ]),
                dbc.CardBody([
                    html.Div(id="datasets-table", children=[
                        html.I(className="fas fa-spinner fa-spin me-2"),
                        "Loading available datasets..."
                    ])
                ])
            ], className="mb-4"),

            # Download Options
            dbc.Card([
                dbc.CardHeader("Download Options"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-database me-2"),
                                "Download Selected Datasets"
                            ], id="download-selected-datasets-btn", color="primary", className="w-100 mb-2",
                               disabled=True),
                            dbc.Button([
                                html.I(className="fas fa-database me-2"),
                                "Download All Datasets"
                            ], id="download-all-datasets-btn", color="success", className="w-100 mb-2"),
                            dbc.Button([
                                html.I(className="fas fa-trash-alt me-2"),
                                "Cleanup Datasets"
                            ], id="cleanup-datasets-btn", color="danger", className="w-100")
                        ], width=12, md=6),

                        dbc.Col([
                            html.Div([
                                html.Label("Download Location:", className="form-label"),
                                dcc.Input(
                                    id="dataset-download-location-input",
                                    type="text",
                                    value=str(config.paths.local_datasets_dir),
                                    className="form-control",
                                    readOnly=True
                                ),
                                html.Small("Datasets will be saved to this directory",
                                          className="text-muted mt-1 d-block")
                            ])
                        ], width=12, md=6)
                    ])
                ])
            ], className="mb-4"),

            # Download Progress
            dbc.Collapse([
                dbc.Card([
                    dbc.CardHeader("Download Progress"),
                    dbc.CardBody([
                        html.Div(id="dataset-download-progress", children=[
                            dbc.Progress(value=0, id="dataset-download-progress-bar", className="mb-2"),
                            html.Div(id="dataset-download-progress-text", children="Ready to download")
                        ]),
                        html.Div(id="dataset-download-status", className="mt-2"),
                        html.Pre(id="dataset-download-log", className="mt-2 p-2 bg-light border",
                                style={"maxHeight": "200px", "overflowY": "auto", "fontSize": "0.8rem"})
                    ])
                ])
            ], id="dataset-download-progress-collapse", is_open=False),
            
            html.Hr(),
            
            # Custom Datasets Integration Info
            dbc.Collapse([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Adding Custom Datasets", className="mb-2"),
                        html.P([
                            "To add your own datasets:",
                            html.Br(),
                            "1. Create a folder in LocalDatasets directory",
                            html.Br(),
                            "2. Include images and corresponding questions/answers",
                            html.Br(), 
                            "3. Follow the ChartQA format for compatibility",
                            html.Br(),
                            "4. Restart the application to detect new datasets"
                        ], className="mb-0", style={"fontSize": "0.9rem"})
                    ])
                ], color="light")
            ], id="custom-datasets-collapse", is_open=False),
            
            dbc.Button([
                html.I(className="fas fa-chevron-down me-2"),
                "Custom Dataset Integration"
            ], id="toggle-custom-datasets", color="light", outline=True, size="sm", className="mt-2")
        ])
    
    def _create_about_section(self):
        """Create the about tool section (placeholder for now)."""
        return html.Div([
            html.H5("About ChartViz", className="mb-3"),
            html.P("This section will be updated later with tool information.", 
                  className="text-muted",
                  style={"fontStyle": "italic"})
        ])
    
    def _create_models_table(self, models_data):
        """Create a table displaying available models."""
        if not models_data:
            return html.Div([
                html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                "No models configuration found. Please check your config.py file."
            ])

        # Create table header
        table_header = [
            html.Thead([
                html.Tr([
                    html.Th("", style={"width": "50px"}),  # Checkbox column
                    html.Th("Model Name"),
                    html.Th("Type"),
                    html.Th("Size (GB)", style={"width": "100px"}),
                    html.Th("Status", style={"width": "120px"}),
                    html.Th("Actions", style={"width": "150px"})
                ])
            ])
        ]

        # Create table body
        table_rows = []
        for model_id, model_info in models_data.items():
            status_badge = self._get_status_badge(model_info.get('downloaded', False))

            table_rows.append(html.Tr([
                html.Td([
                    dbc.Checkbox(
                        id={"type": "model-checkbox", "index": model_id},
                        value=False
                    )
                ]),
                html.Td([
                    html.Strong(model_info.get('name', model_id)),
                    html.Br(),
                    html.Small(model_info.get('hf_repo', ''), className="text-muted")
                ]),
                html.Td(model_info.get('type', 'Unknown')),
                html.Td(f"{model_info.get('size_gb', 2.5):.1f}"),
                html.Td(status_badge),
                html.Td([
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="fas fa-info-circle")
                        ], id={"type": "model-info-btn", "index": model_id},
                           color="outline-info", size="sm", title="Model Info"),
                        dbc.Button([
                            html.I(className="fas fa-trash-alt")
                        ], id={"type": "model-delete-btn", "index": model_id},
                           color="outline-danger", size="sm", title="Delete Model",
                           disabled=not model_info.get('downloaded', False))
                    ], size="sm")
                ])
            ]))

        table_body = [html.Tbody(table_rows)]

        return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True,
                        className="mb-0")

    def _create_datasets_table(self, datasets_data):
        """Create a table displaying available datasets."""
        if not datasets_data:
            return html.Div([
                html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                "No datasets configuration found. Please check your config.py file."
            ])

        # Create table header
        table_header = [
            html.Thead([
                html.Tr([
                    html.Th("", style={"width": "50px"}),  # Checkbox column
                    html.Th("Dataset Name"),
                    html.Th("Size (GB)", style={"width": "100px"}),
                    html.Th("Status", style={"width": "120px"}),
                    html.Th("Actions", style={"width": "150px"})
                ])
            ])
        ]

        # Create table body
        table_rows = []
        for dataset_id, dataset_info in datasets_data.items():
            status_badge = self._get_status_badge(dataset_info.get('downloaded', False))

            table_rows.append(html.Tr([
                html.Td([
                    dbc.Checkbox(
                        id={"type": "dataset-checkbox", "index": dataset_id},
                        value=False
                    )
                ]),
                html.Td([
                    html.Strong(dataset_info.get('name', dataset_id)),
                    html.Br(),
                    html.Small(dataset_info.get('hf_repo', ''), className="text-muted")
                ]),
                html.Td(f"{dataset_info.get('size_gb', 5.0):.1f}"),
                html.Td(status_badge),
                html.Td([
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="fas fa-info-circle")
                        ], id={"type": "dataset-info-btn", "index": dataset_id},
                           color="outline-info", size="sm", title="Dataset Info"),
                        dbc.Button([
                            html.I(className="fas fa-trash-alt")
                        ], id={"type": "dataset-delete-btn", "index": dataset_id},
                           color="outline-danger", size="sm", title="Delete Dataset",
                           disabled=not dataset_info.get('downloaded', False))
                    ], size="sm")
                ])
            ]))

        table_body = [html.Tbody(table_rows)]

        return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True,
                        className="mb-0")

    def _get_status_badge(self, downloaded):
        """Get status badge for model/dataset."""
        if downloaded:
            return dbc.Badge("✓ Downloaded", color="success", className="p-1")
        else:
            return dbc.Badge("○ Not Downloaded", color="secondary", className="p-1")

    def _estimate_model_size(self, model_config):
        """Estimate model size in GB."""
        size_estimates = {
            "unichart": 3.2,
            "pix2struct": 2.8,
            "donut": 2.1,
        }

        model_name = model_config.get('name', '').lower()
        for key, size in size_estimates.items():
            if key in model_name:
                return size

        return 2.5

    def _estimate_dataset_size(self, dataset_config):
        """Estimate dataset size in GB."""
        size_estimates = {
            "chartqa": 8.5,
            "plotqa": 12.3,
            "figureqa": 6.2,
        }

        dataset_name = dataset_config.get('name', '').lower()
        for key, size in size_estimates.items():
            if key in dataset_name:
                return size

        return 5.0

    def _is_model_downloaded(self, model_config):
        """Check if a model is downloaded."""
        try:
            model_dir = Path(model_config.local_dir)
            # Check for common model files
            model_files = [
                "config.json",
                "pytorch_model.bin",
                "model.safetensors",
                "tokenizer.json"
            ]

            for file in model_files:
                if (model_dir / file).exists():
                    return True

            return False
        except:
            return False

    def _is_dataset_downloaded(self, dataset_config):
        """Check if a dataset is downloaded."""
        try:
            dataset_dir = Path(dataset_config.local_dir)
            # Check for common dataset files
            dataset_files = [
                "data_full.json",
                "data_full.csv",
                "metadata.json"
            ]

            for file in dataset_files:
                if (dataset_dir / file).exists():
                    return True

            return False
        except:
            return False

    def _start_model_download_process(self, model_ids):
        """Start the download process for selected models."""
        progress_bar = dbc.Progress(value=0, id="model-download-progress-bar", className="mb-2")
        progress_text = html.Div("Initializing download...", id="model-download-progress-text")

        # This would normally start a background thread to run the download
        # For now, we'll show a placeholder
        status = html.Div([
            html.I(className="fas fa-info-circle text-info me-2"),
            "Download functionality would be implemented here"
        ], className="text-info")

        log_content = "Download process started...\n" + "\n".join([f"- Preparing to download: {mid}" for mid in model_ids])

        return True, [progress_bar, progress_text], status, log_content

    def _start_dataset_download_process(self, dataset_ids):
        """Start the download process for selected datasets."""
        progress_bar = dbc.Progress(value=0, id="dataset-download-progress-bar", className="mb-2")
        progress_text = html.Div("Initializing download...", id="dataset-download-progress-text")

        # This would normally start a background thread to run the download
        # For now, we'll show a placeholder
        status = html.Div([
            html.I(className="fas fa-info-circle text-info me-2"),
            "Download functionality would be implemented here"
        ], className="text-info")

        log_content = "Download process started...\n" + "\n".join([f"- Preparing to download: {did}" for did in dataset_ids])

        return True, [progress_bar, progress_text], status, log_content

    def register_callbacks(self, app):
        """Register callbacks for the help popup with integrated download functionality."""
        
        # Help modal toggle
        @app.callback(
            Output(self.modal_id, "is_open"),
            [Input("help-btn", "n_clicks"),
             Input("close-help-modal", "n_clicks")],
            State(self.modal_id, "is_open")
        )
        def toggle_help_modal(help_clicks, close_clicks, is_open):
            """Toggle help modal visibility."""
            if help_clicks or close_clicks:
                return not is_open
            return is_open
        
        # Custom sections toggles
        @app.callback(
            Output("custom-models-collapse", "is_open"),
            Input("toggle-custom-models", "n_clicks"),
            State("custom-models-collapse", "is_open")
        )
        def toggle_custom_models(n_clicks, is_open):
            """Toggle custom models section."""
            if n_clicks:
                return not is_open
            return is_open
        
        @app.callback(
            Output("custom-datasets-collapse", "is_open"),
            Input("toggle-custom-datasets", "n_clicks"),
            State("custom-datasets-collapse", "is_open")
        )
        def toggle_custom_datasets(n_clicks, is_open):
            """Toggle custom datasets section."""
            if n_clicks:
                return not is_open
            return is_open

        # MODEL MANAGEMENT CALLBACKS
        
        # System requirements check
        @app.callback(
            Output("system-requirements-status", "children"),
            Input("recheck-system-btn", "n_clicks")
        )
        def check_system_requirements(n_clicks):
            """Check system requirements."""
            try:
                import torch

                # Check PyTorch
                torch_version = torch.__version__
                cuda_available = torch.cuda.is_available()

                gpu_info = ""
                if cuda_available:
                    gpu_info = "CUDA available"

                return [
                    html.I(className="fas fa-check-circle text-success me-2"),
                    html.Span("System requirements check passed", className="text-success"),
                    html.Br(),
                    html.Small(f"PyTorch {torch_version}", className="text-muted"),
                    html.Br() if gpu_info else None,
                    html.Small(gpu_info, className="text-muted") if gpu_info else None
                ]
            except ImportError:
                return [
                    html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                    html.Span("PyTorch not found. Please install PyTorch first.", className="text-danger")
                ]
            except Exception as e:
                return [
                    html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                    html.Span(f"Error checking requirements: {str(e)}", className="text-warning")
                ]

        # Load models data
        @app.callback(
            Output("models-table", "children"),
            [Input("refresh-models-btn", "n_clicks"),
             Input(self.modal_id, "is_open")]
        )
        def load_models_table(n_clicks, is_open):
            """Load and display models table."""
            if not is_open:
                return [html.I(className="fas fa-spinner fa-spin me-2"), "Loading available models..."]
            try:
                models_data = {}

                for model_id, model_config in config.models.items():
                    # Check if model is downloaded
                    downloaded = self._is_model_downloaded(model_config)

                    models_data[model_id] = {
                        'name': model_config.name,
                        'type': model_config.model_type.value,
                        'hf_repo': model_config.hf_repo or 'N/A',
                        'downloaded': downloaded,
                        'size_gb': self._estimate_model_size({
                            'name': model_config.name
                        })
                    }

                return self._create_models_table(models_data)

            except Exception as e:
                return html.Div([
                    html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                    f"Error loading models: {str(e)}"
                ], className="text-danger")

        # Download selected models
        @app.callback(
            [Output("model-download-progress-collapse", "is_open"),
             Output("model-download-progress", "children"),
             Output("model-download-status", "children"),
             Output("model-download-log", "children")],
            Input("download-selected-models-btn", "n_clicks"),
            State({"type": "model-checkbox", "index": ALL}, "value"),
            State({"type": "model-checkbox", "index": ALL}, "id")
        )
        def download_selected_models(n_clicks, checkbox_values, checkbox_ids):
            """Download selected models."""
            if not n_clicks:
                return False, "", "", ""

            # Get selected model IDs
            selected_models = []
            for value, checkbox_id in zip(checkbox_values, checkbox_ids):
                if value:
                    selected_models.append(checkbox_id['index'])

            if not selected_models:
                return True, "No models selected", "Please select at least one model to download", ""

            # Start download process
            return self._start_model_download_process(selected_models)

        # Download all models
        @app.callback(
            [Output("model-download-progress-collapse", "is_open", allow_duplicate=True),
             Output("model-download-progress", "children", allow_duplicate=True),
             Output("model-download-status", "children", allow_duplicate=True),
             Output("model-download-log", "children", allow_duplicate=True)],
            Input("download-all-models-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def download_all_models(n_clicks):
            """Disabled after model set reduction."""
            return False, "", "", ""

        # DATASET MANAGEMENT CALLBACKS

        # Storage information
        @app.callback(
            Output("storage-info", "children"),
            Input(self.modal_id, "is_open")
        )
        def get_storage_info(is_open):
            """Get storage information."""
            if not is_open:
                return ""

            try:
                import shutil

                # Get disk usage for the datasets directory
                datasets_dir = config.paths.local_datasets_dir
                if datasets_dir.exists():
                    usage = shutil.disk_usage(datasets_dir)
                    free_gb = usage.free / (1024**3)
                    total_gb = usage.total / (1024**3)

                    return [
                        html.I(className="fas fa-hdd text-info me-2"),
                        html.Span(f"{free_gb:.1f}", className="fw-bold"),
                        html.Span(" GB free", className="text-muted"),
                        html.Br(),
                        html.Small(f"Total: {total_gb:.1f} GB", className="text-muted")
                    ]
                else:
                    # Get disk usage for parent directory
                    parent_usage = shutil.disk_usage(datasets_dir.parent)
                    free_gb = parent_usage.free / (1024**3)

                    return [
                        html.I(className="fas fa-hdd text-info me-2"),
                        html.Span(f"{free_gb:.1f}", className="fw-bold"),
                        html.Span(" GB free in parent directory", className="text-muted")
                    ]

            except Exception as e:
                return [
                    html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                    html.Span(f"Could not check storage: {str(e)}", className="text-warning")
                ]

        # Load datasets data
        @app.callback(
            Output("datasets-table", "children"),
            [Input("refresh-datasets-btn", "n_clicks"),
             Input(self.modal_id, "is_open")]
        )
        def load_datasets_table(n_clicks, is_open):
            """Load and display datasets table."""
            if not is_open:
                return [html.I(className="fas fa-spinner fa-spin me-2"), "Loading available datasets..."]
            try:
                datasets_data = {}

                for dataset_id, dataset_config in config.datasets.items():
                    # Check if dataset is downloaded
                    downloaded = self._is_dataset_downloaded(dataset_config)

                    datasets_data[dataset_id] = {
                        'name': dataset_config.name,
                        'hf_repo': dataset_config.hf_repo or 'N/A',
                        'downloaded': downloaded,
                        'size_gb': self._estimate_dataset_size({
                            'name': dataset_config.name
                        })
                    }

                return self._create_datasets_table(datasets_data)

            except Exception as e:
                return html.Div([
                    html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                    f"Error loading datasets: {str(e)}"
                ], className="text-danger")

        # Download selected datasets
        @app.callback(
            [Output("dataset-download-progress-collapse", "is_open"),
             Output("dataset-download-progress", "children"),
             Output("dataset-download-status", "children"),
             Output("dataset-download-log", "children")],
            Input("download-selected-datasets-btn", "n_clicks"),
            State({"type": "dataset-checkbox", "index": ALL}, "value"),
            State({"type": "dataset-checkbox", "index": ALL}, "id")
        )
        def download_selected_datasets(n_clicks, checkbox_values, checkbox_ids):
            """Download selected datasets."""
            if not n_clicks:
                return False, "", "", ""

            # Get selected dataset IDs
            selected_datasets = []
            for value, checkbox_id in zip(checkbox_values, checkbox_ids):
                if value:
                    selected_datasets.append(checkbox_id['index'])

            if not selected_datasets:
                return True, "No datasets selected", "Please select at least one dataset to download", ""

            # Start download process
            return self._start_dataset_download_process(selected_datasets)

        # Download all datasets
        @app.callback(
            [Output("dataset-download-progress-collapse", "is_open", allow_duplicate=True),
             Output("dataset-download-progress", "children", allow_duplicate=True),
             Output("dataset-download-status", "children", allow_duplicate=True),
             Output("dataset-download-log", "children", allow_duplicate=True)],
            Input("download-all-datasets-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def download_all_datasets(n_clicks):
            """Download all available datasets."""
            if not n_clicks:
                return False, "", "", ""

            all_dataset_ids = list(config.datasets.keys())
            return self._start_dataset_download_process(all_dataset_ids)

        # Enable/disable download selected buttons based on checkbox selections
        @app.callback(
            Output("download-selected-models-btn", "disabled"),
            Input({"type": "model-checkbox", "index": ALL}, "value"),
            prevent_initial_call=True
        )
        def toggle_model_download_button(checkbox_values):
            """Enable download selected button if any models are selected."""
            if not checkbox_values:
                return True
            return not any(checkbox_values)

        @app.callback(
            Output("download-selected-datasets-btn", "disabled"),
            Input({"type": "dataset-checkbox", "index": ALL}, "value"),
            prevent_initial_call=True
        )
        def toggle_dataset_download_button(checkbox_values):
            """Enable download selected button if any datasets are selected."""
            if not checkbox_values:
                return True
            return not any(checkbox_values)


