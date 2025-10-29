"""
Attention Analysis Popup Component - Comprehensive attention visualization and analysis.
Provides multiple analysis modes for understanding model attention patterns.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, callback_context, ALL
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any
import base64
import io
from PIL import Image
import torch
import numpy as np
import json

from components.attention_heatmap_overlay import AttentionHeatmapOverlay
from components.attention_statistics import AttentionStatistics
import logging

logger = logging.getLogger(__name__)


class MockAttentionOutput:
    """Mock AttentionOutput object for serialized attention data."""
    def __init__(self, data_dict):
        logger.info(f"🔍 MockAttentionOutput init with keys: {list(data_dict.keys())}")
        
        # Convert lists back to PyTorch tensors
        if 'cross_attention' in data_dict and data_dict['cross_attention'] is not None:
            logger.info(f"✅ Converting cross_attention from {type(data_dict['cross_attention'])}")
            self.cross_attention = torch.tensor(data_dict['cross_attention'])
            logger.info(f"✅ Cross-attention tensor created: {self.cross_attention.shape}")
        else:
            logger.warning("❌ No cross_attention data in dict")
            self.cross_attention = None
            
        if 'text_self_attention' in data_dict and data_dict['text_self_attention'] is not None:
            self.text_self_attention = torch.tensor(data_dict['text_self_attention'])
        else:
            self.text_self_attention = None
            
        if 'image_self_attention' in data_dict and data_dict['image_self_attention'] is not None:
            self.image_self_attention = torch.tensor(data_dict['image_self_attention'])
        else:
            self.image_self_attention = None
        
        # Copy simple attributes
        self.text_tokens = data_dict.get('text_tokens')
        self.image_patch_coords = data_dict.get('image_patch_coords')
        self.predicted_answer = data_dict.get('predicted_answer')
        self.confidence_score = data_dict.get('confidence_score')
        self.layer_count = data_dict.get('layer_count')
        self.head_count = data_dict.get('head_count')
        self.image_size = data_dict.get('image_size')
        self.patch_size = data_dict.get('patch_size')


class AttentionAnalysisPopup:
    """Comprehensive attention analysis popup with multiple visualization modes."""
    
    def __init__(self):
        """Initialize the attention analysis popup component."""
        self.modal_id = "attention-analysis-modal"
        self.heatmap_overlay = AttentionHeatmapOverlay()
        self.statistics = AttentionStatistics()
    
    def _prepare_attention_data(self, attention_data):
        """Convert serialized attention data to AttentionOutput-like object."""
        logger.info(f"🔍 _prepare_attention_data called with: {type(attention_data)}")
        
        if attention_data is None:
            logger.warning("❌ attention_data is None")
            return None
        
        # If it's already an object with attributes, return as is
        if hasattr(attention_data, 'cross_attention'):
            logger.info(f"✅ AttentionOutput object detected, cross_attention: {attention_data.cross_attention is not None}")
            return attention_data
        
        # If it's a dictionary, convert to mock object
        if isinstance(attention_data, dict):
            logger.info(f"🔍 Dict detected with keys: {list(attention_data.keys())}")
            if 'cross_attention' in attention_data:
                if attention_data['cross_attention'] is not None:
                    logger.info(f"✅ Cross-attention found in dict: {len(attention_data['cross_attention']) if isinstance(attention_data['cross_attention'], list) else 'tensor'}")
                else:
                    logger.warning("❌ Cross-attention in dict is None")
            else:
                logger.warning("❌ No cross_attention key in dict")
            return MockAttentionOutput(attention_data)
        
        logger.warning(f"❌ Unrecognized attention_data type: {type(attention_data)}")
        return None
    
    def create_popup_modal(self):
        """Create the main attention analysis modal."""
        return dbc.Modal([
            dbc.ModalHeader([
                html.H4([
                    html.I(className="fas fa-brain me-2"),
                    "Attention Analysis"
                ], className="mb-0")
            ], close_button=True),
            dbc.ModalBody([
                self._create_analysis_selector(),
                html.Hr(),
                html.Div(id="attention-analysis-content"),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.Div(id="attention-summary-stats")
                    ], width=6),
                    dbc.Col([
                        html.Div(id="attention-detailed-analysis")
                    ], width=6)
                ])
            ], style={"maxHeight": "80vh", "overflowY": "auto"}),
            dbc.ModalFooter([
                dbc.Button([
                    html.I(className="fas fa-download me-2"),
                    "Export Analysis"
                ], id="export-attention-analysis", color="primary", outline=True),
                dbc.Button([
                    html.I(className="fas fa-times me-2"),
                    "Close"
                ], id="close-attention-popup-footer", color="secondary")
            ])
        ], id=self.modal_id, size="xl", is_open=False, className="attention-modal")
    
    def create_settings_tooltip(self):
        """Create settings tooltip for attention analysis controls."""
        return dbc.Tooltip([
            html.P("Analysis Settings:", className="fw-bold mb-2"),
            html.Ul([
                html.Li("Opacity: Controls heatmap transparency"),
                html.Li("Blur: Smooths attention boundaries"), 
                html.Li("Colors: Changes visualization palette"),
                html.Li("Scale: Adjusts analysis granularity")
            ], className="mb-0")
        ], target="attention-settings-icon", placement="top")
    
    def _create_analysis_selector(self):
        """Create the analysis type selector with mode cards."""
        return html.Div([
            html.H5("Choose Analysis Type", className="mb-3"),
            
            # Basic Analysis Row
            html.H6("Basic Analysis", className="text-muted mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-fire text-danger me-2"),
                                "Focus Heatmap"
                            ], className="card-title"),
                            html.P("Shows exactly where the model focused most while processing your question.", 
                                   className="card-text"),
                            dbc.Button("Analyze", id={"type": "analysis-btn", "mode": "focus_heatmap"}, 
                                     color="primary", size="sm", className="w-100")
                        ])
                    ], className="h-100 analysis-card")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-bullseye text-warning me-2"),
                                "Intensity Map"
                            ], className="card-title"),
                            html.P("Highlights top attention regions with detailed annotations and scores.", 
                                   className="card-text"),
                            dbc.Button("Analyze", id={"type": "analysis-btn", "mode": "intensity_map"}, 
                                     color="warning", size="sm", className="w-100")
                        ])
                    ], className="h-100 analysis-card")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-eye text-info me-2"),
                                "Token Importance"
                            ], className="card-title"),
                            html.P("Analyze the importance of different words in your question.", 
                                   className="card-text"),
                            dbc.Button("Analyze", id={"type": "analysis-btn", "mode": "token_importance"}, 
                                     color="info", size="sm", className="w-100")
                        ])
                    ], className="h-100 analysis-card")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-chart-line text-success me-2"),
                                "Quick Stats"
                            ], className="card-title"),
                            html.P("Quick statistical overview of attention patterns.", 
                                   className="card-text"),
                            dbc.Button("Analyze", id={"type": "analysis-btn", "mode": "quick_stats"}, 
                                     color="success", size="sm", className="w-100")
                        ])
                    ], className="h-100 analysis-card")
                ], width=3)
            ], className="mb-4"),
            
            # Advanced Analysis Row
            html.Hr(),
            html.H6("Advanced Analysis", className="text-muted mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-chart-bar text-primary me-2"),
                                "Statistical Analysis"
                            ], className="card-title"),
                            html.P("Comprehensive statistical metrics including entropy, distribution, and concentration.", 
                                   className="card-text"),
                            dbc.Button("Analyze", id={"type": "analysis-btn", "mode": "statistical"}, 
                                     color="primary", size="sm", className="w-100")
                        ])
                    ], className="h-100 analysis-card")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-stream text-info me-2"),
                                "Attention Flow"
                            ], className="card-title"),
                            html.P("Analysis of how attention flows between text tokens and image regions.", 
                                   className="card-text"),
                            dbc.Button("Analyze", id={"type": "analysis-btn", "mode": "attention_flow"}, 
                                     color="info", size="sm", className="w-100")
                        ])
                    ], className="h-100 analysis-card")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-layer-group text-success me-2"),
                                "Layer Comparison"
                            ], className="card-title"),
                            html.P("Compare attention patterns across different model layers.", 
                                   className="card-text"),
                            dbc.Button("Analyze", id={"type": "analysis-btn", "mode": "layer_comparison"}, 
                                     color="success", size="sm", className="w-100")
                        ])
                    ], className="h-100 analysis-card")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-sitemap text-warning me-2"),
                                "Multi-Head Analysis"
                            ], className="card-title"),
                            html.P("Analyze attention patterns across different attention heads.", 
                                   className="card-text"),
                            dbc.Button("Analyze", id={"type": "analysis-btn", "mode": "multi_head"}, 
                                     color="warning", size="sm", className="w-100")
                        ])
                    ], className="h-100 analysis-card")
                ], width=3)
            ], className="mb-3"),
            
            # Second row of advanced analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-crosshairs text-danger me-2"),
                                "Cross-Attention Explorer"
                            ], className="card-title"),
                            html.P("Select tokens from your question to see which image patches they attend to. Interactive & dynamic!", 
                                   className="card-text"),
                            dbc.Button("Explore", id={"type": "analysis-btn", "mode": "cross_attention_explorer"}, 
                                     color="danger", size="sm", className="w-100")
                        ])
                    ], className="h-100 analysis-card")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className="fas fa-comments text-secondary me-2"),
                                "Cross-Attention Explorer (Question Tokens Only)"
                            ], className="card-title"),
                            html.P("View cross-attention with image and question tokens only. Focuses on your input question!", 
                                   className="card-text"),
                            dbc.Button("Explore", id={"type": "analysis-btn", "mode": "cross_attention_explorer_text"}, 
                                     color="secondary", size="sm", className="w-100")
                        ])
                    ], className="h-100 analysis-card")
                ], width=3),
            ], className="mb-4"),
            
            # Analysis settings
            dbc.Row([
                dbc.Col([
                    html.Label("Heatmap Opacity:", className="fw-bold"),
                    dcc.Slider(id="attention-opacity-slider", min=0.1, max=1.0, step=0.1, 
                             value=0.7, marks={i/10: str(i/10) for i in range(1, 11)})
                ], width=3),
                dbc.Col([
                    html.Label("Blur Radius:", className="fw-bold"),
                    dcc.Slider(id="attention-blur-slider", min=0, max=5, step=0.5, 
                             value=1.0, marks={i/2: str(i/2) for i in range(0, 11)})
                ], width=3),
                dbc.Col([
                    html.Label("Color Scheme:", className="fw-bold"),
                    dcc.Dropdown(id="attention-colorscheme-dropdown", 
                               options=[
                                   {"label": "Hot", "value": "hot"},
                                   {"label": "Viridis", "value": "viridis"},
                                   {"label": "Plasma", "value": "plasma"},
                                   {"label": "Turbo", "value": "turbo"}
                               ], value="hot")
                ], width=3),
                dbc.Col([
                    html.I(id="attention-settings-icon", className="fas fa-info-circle text-info", 
                           style={"cursor": "help", "fontSize": "1.2rem"})
                ], width=3, className="d-flex align-items-end justify-content-center")
            ])
        ])
    
    def register_callbacks(self, app):
        """Register all callbacks for the attention analysis popup."""
        
        # Toggle modal
        @app.callback(
            Output(self.modal_id, "is_open"),
            [Input("open-attention-analysis", "n_clicks"),
             Input("close-attention-popup-footer", "n_clicks")],
            [State(self.modal_id, "is_open")]
        )
        def toggle_attention_modal(open_clicks, close_footer_clicks, is_open):
            """Toggle the attention analysis modal."""
            if open_clicks or close_footer_clicks:
                return not is_open
            return is_open
        
        # Analysis content update
        @app.callback(
            [Output("attention-analysis-content", "children"),
             Output("attention-summary-stats", "children"),
             Output("attention-detailed-analysis", "children"),
             Output("attention-export-status", "children")],
            [Input({"type": "analysis-btn", "mode": ALL}, "n_clicks")],
            [State("current-attention-data", "data"),
             State("current-image-data", "data"),
             State("question-input", "value"),
             State("attention-opacity-slider", "value"),
             State("attention-blur-slider", "value"),
             State("attention-colorscheme-dropdown", "value")]
        )
        def update_attention_analysis(n_clicks_list, attention_data, image_data, question,
                                    opacity, blur_radius, colorscheme):
            """Update attention analysis content based on selected mode."""
            
            # Determine which button was clicked
            ctx = callback_context
            if not ctx or not any(n_clicks_list):
                return "", "", "", ""
            
            # Get the analysis type from triggered button
            triggered_id = ctx.triggered[0]["prop_id"]
            if not triggered_id or triggered_id == ".":
                return "", "", "", ""
            
            # Parse the button mode
            try:
                button_data = json.loads(triggered_id.split('.')[0])
                analysis_type = button_data.get("mode", "focus_heatmap")
            except (json.JSONDecodeError, KeyError, IndexError):
                analysis_type = "focus_heatmap"
            
            # Check if we have attention data
            if not attention_data:
                return [
                    dbc.Alert("No attention data available. Please run a model prediction first.", 
                             color="warning"),
                    "", "", ""
                ]
            
            # Check if attention data has any useful information
            logger.info(f"🔍 Attention data type: {type(attention_data)}")
            if isinstance(attention_data, dict):
                logger.info(f"🔍 Attention data keys: {list(attention_data.keys())}")
                if 'cross_attention' in attention_data:
                    if attention_data['cross_attention'] is not None:
                        logger.info(f"✅ Cross-attention available in attention_data")
                    else:
                        logger.warning("❌ Cross-attention is None in attention_data")
            
            try:
                # Create analysis content based on type
                if analysis_type == "focus_heatmap":
                    content = self._create_focus_heatmap_content(
                        attention_data, image_data, question, opacity, blur_radius, colorscheme
                    )
                elif analysis_type == "intensity_map":
                    content = self._create_intensity_map_content(
                        attention_data, image_data, question
                    )
                elif analysis_type == "token_importance":
                    content = self._create_token_importance_content(
                        attention_data, image_data, question
                    )
                elif analysis_type == "quick_stats":
                    content = self._create_quick_stats_content(attention_data, question)
                elif analysis_type == "statistical":
                    content = self._create_advanced_statistical_content(
                        attention_data, image_data, question
                    )
                elif analysis_type == "attention_flow":
                    content = self._create_attention_flow_content(
                        attention_data, image_data, question
                    )
                elif analysis_type == "layer_comparison":
                    content = self._create_layer_comparison_content(
                        attention_data, image_data, question
                    )
                elif analysis_type == "multi_head":
                    content = self._create_multi_head_content(
                        attention_data, image_data, question
                    )
                elif analysis_type == "cross_attention_explorer":
                    content = self._create_cross_attention_explorer_content(
                        attention_data, image_data, question
                    )
                elif analysis_type == "cross_attention_explorer_text":
                    content = self._create_cross_attention_explorer_text_content(
                        attention_data, image_data, question
                    )
                else:
                    content = dbc.Alert("Analysis type not implemented yet.", color="info")
                
                # Generate summary stats and detailed analysis
                summary_stats = self._create_summary_statistics(attention_data)
                detailed_analysis = self._create_detailed_analysis(attention_data, analysis_type)
                
                return content, summary_stats, detailed_analysis, ""
                
            except Exception as e:
                logger.error(f"Error in attention analysis: {e}")
                return (
                    dbc.Alert(f"Error creating analysis: {str(e)}", color="danger"),
                    "", "", ""
                )
        
        # Export functionality
        @app.callback(
            Output("attention-export-status", "children", allow_duplicate=True),
            Input("export-attention-analysis", "n_clicks"),
            [State("current-attention-data", "data"),
             State("attention-analysis-content", "children")],
            prevent_initial_call=True
        )
        def export_attention_analysis(n_clicks, attention_data, analysis_content):
            """Export attention analysis results with comprehensive data."""
            if n_clicks and attention_data:
                try:
                    # Import export utilities
                    from utils.analysis_export import analysis_exporter
                    
                    # Prepare export data
                    export_data = {
                        'analysis_type': 'attention_analysis',
                        'title': 'Attention Analysis Results',
                        'description': 'Comprehensive attention analysis from ChartViz',
                        'metrics': attention_data,
                        'insights': ['Attention analysis completed successfully'],
                        'timestamp': attention_data.get('timestamp') if isinstance(attention_data, dict) else None
                    }
                    
                    # Export as JSON (default)
                    export_result = analysis_exporter.export_analysis_results(
                        export_data, 
                        export_format='json'
                    )
                    
                    if export_result.get('success'):
                        return dbc.Alert([
                            html.I(className="fas fa-check-circle me-2"),
                            f"Analysis exported successfully as {export_result['filename']}"
                        ], color="success", duration=5000)
                    else:
                        return dbc.Alert([
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            f"Export failed: {export_result.get('error', 'Unknown error')}"
                        ], color="danger", duration=5000)
                        
                except Exception as e:
                    logger.error(f"Export error: {e}")
                    return dbc.Alert([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        f"Export failed: {str(e)}"
                    ], color="danger", duration=5000)
            
            return ""
    
    def _create_focus_heatmap_content(self, attention_data, image_data, question, 
                                    opacity, blur_radius, colorscheme):
        """Create focus heatmap visualization content with real attention data."""
        try:
            if not attention_data:
                return self._create_no_data_message("focus heatmap")
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return self._create_no_data_message("focus heatmap (invalid data)")
            
            # Process image data if available
            if image_data and 'contents' in image_data:
                # Remove data:image/jpeg;base64, prefix if present
                image_content = image_data['contents']
                if ',' in image_content:
                    image_content = image_content.split(',')[1]
                
                # Decode base64 image
                image_bytes = base64.b64decode(image_content)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Create actual heatmap using attention analysis components
                fig = self.heatmap_overlay.create_focus_heatmap(
                    attention_output=attention_output,
                    image=image,
                    question=question,
                    opacity=opacity,
                    blur_radius=blur_radius
                )
                fig.update_layout(height=500)
                
                return html.Div([
                    html.H5("🔥 Model Focus Heatmap", className="mb-3"),
                    html.P("This visualization shows exactly where the model focused most while "
                          "processing your question. Brighter areas indicate higher attention.", 
                          className="text-muted mb-4"),
                    
                    dcc.Graph(
                        figure=fig,
                        config={"displayModeBar": True, "toImageButtonOptions": {"format": "png"}}
                    ),
                    
                    # Settings info
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Current Settings"),
                                    html.P([
                                        f"Opacity: {opacity:.1f} | ",
                                        f"Blur: {blur_radius:.1f} | ",
                                        f"Colors: {colorscheme}"
                                    ], className="mb-0 small")
                                ])
                            ])
                        ], width=12)
                    ], className="mt-3")
                ])
            else:
                return self._create_no_data_message("focus heatmap (no image data)")
                
        except Exception as e:
            logger.error(f"Error creating focus heatmap: {e}")
            # Check if it's a cross-attention error
            if "Cross-attention data required" in str(e):
                return dbc.Alert([
                    html.H5("⚠️ Cross-Attention Not Available", className="alert-heading"),
                    html.P("The model's attention extraction did not produce cross-attention data, "
                          "which is required for focus heatmaps."),
                    html.Hr(),
                    html.P([
                        "This can happen when:",
                        html.Ul([
                            html.Li("The model architecture doesn't support cross-attention extraction"),
                            html.Li("The attention patterns are too sparse or diffuse to extract reliably"),
                            html.Li("The model uses flash attention or other optimized attention mechanisms")
                        ])
                    ], className="mb-0"),
                    html.Hr(),
                    html.P("✅ Try other visualizations like 'Statistical Analysis' or 'Quick Stats' which use different attention data.",
                          className="mb-0 fw-bold")
                ], color="warning")
            else:
                return dbc.Alert(f"Error creating focus heatmap: {str(e)}", color="danger")
    
    def _create_statistics_content(self, attention_data, question):
        """Create statistics analysis content with real data."""
        try:
            if not attention_data:
                return self._create_no_data_message("statistics")
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return self._create_no_data_message("statistics (invalid data)")
            
            # Create actual statistics using attention statistics component
            fig = self.statistics.create_attention_distribution_analysis(
                attention_output=attention_output,
                question=question
            )
            
            return html.Div([
                html.H5("📊 Attention Statistics Analysis", className="mb-3"),
                html.P("Comprehensive statistical analysis of attention patterns including "
                      "distribution metrics, entropy, and concentration measures.", 
                      className="text-muted mb-4"),
                
                dcc.Graph(
                    figure=fig,
                    config={"displayModeBar": True, "toImageButtonOptions": {"format": "png"}}
                )
            ])
            
        except Exception as e:
            logger.error(f"Error creating statistics: {e}")
            return dbc.Alert(f"Error creating statistics analysis: {str(e)}", color="danger")
    
    def _create_intensity_map_content(self, attention_data, image_data, question):
        """Create intensity mapping content with real data."""
        try:
            if not attention_data or not image_data:
                return self._create_no_data_message("intensity mapping")
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return self._create_no_data_message("intensity mapping (invalid data)")
            
            # Process image data
            if 'contents' in image_data:
                image_content = image_data['contents']
                if ',' in image_content:
                    image_content = image_content.split(',')[1]
                
                image_bytes = base64.b64decode(image_content)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Create actual intensity map
                fig = self.heatmap_overlay.create_attention_intensity_map(
                    attention_output=attention_output,
                    image=image,
                    question=question,
                    show_top_regions=5
                )
                
                return html.Div([
                    html.H5("🎯 Attention Intensity Mapping", className="mb-3"),
                    html.P("Highlights the top attention regions with detailed annotations. "
                          "Each colored box represents a high-attention area with intensity scores.", 
                          className="text-muted mb-4"),
                    
                    dcc.Graph(
                        figure=fig,
                        config={"displayModeBar": True, "toImageButtonOptions": {"format": "png"}}
                    )
                ])
            else:
                return self._create_no_data_message("intensity mapping (no image)")
                
        except Exception as e:
            logger.error(f"Error creating intensity map: {e}")
            return dbc.Alert(f"Error creating intensity map: {str(e)}", color="danger")
    
    def _create_multi_head_content(self, attention_data, image_data, question):
        """Create multi-head comparison content using the new analysis framework."""
        try:
            if not attention_data:
                return self._create_no_data_message("multi-head comparison")
            
            # Import the advanced analysis framework
            from components.advanced_analysis import AdvancedAnalysisManager
            analysis_manager = AdvancedAnalysisManager()
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return self._create_no_data_message("multi-head comparison (invalid data)")
            
            # Process image if available
            image = None
            if image_data and 'contents' in image_data:
                image_content = image_data['contents']
                if ',' in image_content:
                    image_content = image_content.split(',')[1]
                
                image_bytes = base64.b64decode(image_content)
                image = Image.open(io.BytesIO(image_bytes))
            
            # Run multi-head analysis
            analysis_result = analysis_manager.run_analysis(
                analysis_type="multi_head",
                attention_output=attention_output,
                image=image,
                question=question
            )
            
            return html.Div([
                html.H5("🌊 Multi-Head Attention Analysis", className="mb-3"),
                html.P(analysis_result.description, className="text-muted mb-4"),
                
                dcc.Graph(
                    figure=analysis_result.visualization,
                    config={"displayModeBar": True, "toImageButtonOptions": {"format": "png"}}
                ),
                
                # Show key insights
                html.Hr(),
                html.H6("Head Specialization Insights", className="mb-2"),
                html.Ul([html.Li(insight) for insight in analysis_result.insights])
            ])
                
        except Exception as e:
            logger.error(f"Error creating multi-head comparison: {e}")
            return dbc.Alert(f"Error creating multi-head comparison: {str(e)}", color="danger")
    
    def _create_token_importance_content(self, attention_data, image_data, question):
        """Create token importance analysis content."""
        try:
            if not attention_data:
                return self._create_no_data_message("token importance")
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return self._create_no_data_message("token importance (invalid data)")
            
            # Calculate token importance
            if attention_output.cross_attention is None or not attention_output.text_tokens:
                return dbc.Alert("No token or attention data available for importance analysis", color="warning")
            
            cross_attn = attention_output.cross_attention
            if hasattr(cross_attn, 'detach'):
                cross_attn = cross_attn.detach().cpu().numpy()
            else:
                cross_attn = np.array(cross_attn)
            
            while len(cross_attn.shape) > 2:
                cross_attn = cross_attn.mean(axis=0)
            
            # Sum attention across image patches for each token
            token_importance = np.sum(cross_attn, axis=1)
            token_importance = token_importance / np.sum(token_importance)  # Normalize
            
            # Create token analysis
            tokens = attention_output.text_tokens[:len(token_importance)]
            importance_scores = token_importance[:len(tokens)].tolist()
            
            # Create bar chart
            fig = go.Figure(data=go.Bar(
                x=tokens,
                y=importance_scores,
                marker_color='lightblue'
            ))
            fig.update_layout(
                title="Token Importance Analysis",
                xaxis_title="Tokens",
                yaxis_title="Importance Score",
                height=400
            )
            
            return html.Div([
                html.H5("🔤 Token Importance Analysis", className="mb-3"),
                html.P("Shows how much attention each word in your question received from the model.", 
                       className="text-muted mb-4"),
                
                dcc.Graph(
                    figure=fig,
                    config={"displayModeBar": True, "toImageButtonOptions": {"format": "png"}}
                )
            ])
            
        except Exception as e:
            logger.error(f"Error creating token importance: {e}")
            return dbc.Alert(f"Error creating token importance analysis: {str(e)}", color="danger")
    
    def _create_quick_stats_content(self, attention_data, question):
        """Create quick statistics content."""
        try:
            if not attention_data:
                return self._create_no_data_message("quick statistics")
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return self._create_no_data_message("quick statistics (invalid data)")
            
            if attention_output.cross_attention is None:
                return dbc.Alert("No attention data available for statistics", color="warning")
            
            # Calculate quick statistics
            cross_attn = attention_output.cross_attention
            if hasattr(cross_attn, 'detach'):
                cross_attn = cross_attn.detach().cpu().numpy()
            else:
                cross_attn = np.array(cross_attn)
            
            while len(cross_attn.shape) > 2:
                cross_attn = cross_attn.mean(axis=0)
            
            flat_attn = cross_attn.flatten()
            
            # Calculate basic stats
            stats = {
                'Mean': np.mean(flat_attn),
                'Max': np.max(flat_attn),
                'Std Dev': np.std(flat_attn),
                'Total': np.sum(flat_attn)
            }
            
            # Create metric cards
            metric_cards = []
            colors = ['primary', 'success', 'info', 'warning']
            
            for i, (key, value) in enumerate(stats.items()):
                metric_cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{value:.4f}", className=f"text-{colors[i % len(colors)]}"),
                                html.P(key, className="mb-0 small")
                            ])
                        ])
                    ], width=3)
                )
            
            return html.Div([
                html.H5("📊 Quick Statistics", className="mb-3"),
                html.P("Basic statistical overview of attention patterns.", 
                       className="text-muted mb-4"),
                
                dbc.Row(metric_cards)
            ])
            
        except Exception as e:
            logger.error(f"Error creating quick stats: {e}")
            return dbc.Alert(f"Error creating quick statistics: {str(e)}", color="danger")
    
    def _create_advanced_statistical_content(self, attention_data, image_data, question):
        """Create advanced statistical analysis content using the new analysis framework."""
        try:
            if not attention_data:
                return self._create_no_data_message("advanced statistical analysis")
            
            # Import the advanced analysis framework
            from components.advanced_analysis import AdvancedAnalysisManager
            analysis_manager = AdvancedAnalysisManager()
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return self._create_no_data_message("advanced statistical analysis (invalid data)")
            
            # Process image if available
            image = None
            if image_data and 'contents' in image_data:
                image_content = image_data['contents']
                if ',' in image_content:
                    image_content = image_content.split(',')[1]
                
                image_bytes = base64.b64decode(image_content)
                image = Image.open(io.BytesIO(image_bytes))
            
            # Run statistical analysis
            analysis_result = analysis_manager.run_analysis(
                analysis_type="statistical",
                attention_output=attention_output,
                image=image,
                question=question
            )
            
            return html.Div([
                html.H5("📈 Advanced Statistical Analysis", className="mb-3"),
                html.P(analysis_result.description, className="text-muted mb-4"),
                
                dcc.Graph(
                    figure=analysis_result.visualization,
                    config={"displayModeBar": True, "toImageButtonOptions": {"format": "png"}}
                ),
                
                # Show key insights
                html.Hr(),
                html.H6("Key Insights", className="mb-2"),
                html.Ul([html.Li(insight) for insight in analysis_result.insights])
            ])
            
        except Exception as e:
            logger.error(f"Error creating advanced statistical analysis: {e}")
            return dbc.Alert(f"Error creating advanced statistical analysis: {str(e)}", color="danger")
    
    def _create_attention_flow_content(self, attention_data, image_data, question):
        """Create attention flow analysis content using the new analysis framework."""
        try:
            if not attention_data:
                return self._create_no_data_message("attention flow analysis")
            
            # Import the advanced analysis framework
            from components.advanced_analysis import AdvancedAnalysisManager
            analysis_manager = AdvancedAnalysisManager()
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return self._create_no_data_message("attention flow analysis (invalid data)")
            
            # Process image if available
            image = None
            if image_data and 'contents' in image_data:
                image_content = image_data['contents']
                if ',' in image_content:
                    image_content = image_content.split(',')[1]
                
                image_bytes = base64.b64decode(image_content)
                image = Image.open(io.BytesIO(image_bytes))
            
            # Run attention flow analysis
            analysis_result = analysis_manager.run_analysis(
                analysis_type="attention_flow",
                attention_output=attention_output,
                image=image,
                question=question
            )
            
            return html.Div([
                html.H5("🌊 Attention Flow Analysis", className="mb-3"),
                html.P(analysis_result.description, className="text-muted mb-4"),
                
                dcc.Graph(
                    figure=analysis_result.visualization,
                    config={"displayModeBar": True, "toImageButtonOptions": {"format": "png"}}
                ),
                
                # Show key insights
                html.Hr(),
                html.H6("Flow Insights", className="mb-2"),
                html.Ul([html.Li(insight) for insight in analysis_result.insights])
            ])
            
        except Exception as e:
            logger.error(f"Error creating attention flow analysis: {e}")
            return dbc.Alert(f"Error creating attention flow analysis: {str(e)}", color="danger")
    
    def _create_layer_comparison_content(self, attention_data, image_data, question):
        """Create layer comparison analysis content using the new analysis framework."""
        try:
            if not attention_data:
                return self._create_no_data_message("layer comparison analysis")
            
            # Import the advanced analysis framework
            from components.advanced_analysis import AdvancedAnalysisManager
            analysis_manager = AdvancedAnalysisManager()
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return self._create_no_data_message("layer comparison analysis (invalid data)")
            
            # Process image if available
            image = None
            if image_data and 'contents' in image_data:
                image_content = image_data['contents']
                if ',' in image_content:
                    image_content = image_content.split(',')[1]
                
                image_bytes = base64.b64decode(image_content)
                image = Image.open(io.BytesIO(image_bytes))
            
            # Run layer comparison analysis
            analysis_result = analysis_manager.run_analysis(
                analysis_type="layer_comparison",
                attention_output=attention_output,
                image=image,
                question=question
            )
            
            return html.Div([
                html.H5("🏗️ Layer Comparison Analysis", className="mb-3"),
                html.P(analysis_result.description, className="text-muted mb-4"),
                
                dcc.Graph(
                    figure=analysis_result.visualization,
                    config={"displayModeBar": True, "toImageButtonOptions": {"format": "png"}}
                ),
                
                # Show key insights
                html.Hr(),
                html.H6("Layer Insights", className="mb-2"),
                html.Ul([html.Li(insight) for insight in analysis_result.insights])
            ])
            
        except Exception as e:
            logger.error(f"Error creating layer comparison analysis: {e}")
            return dbc.Alert(f"Error creating layer comparison analysis: {str(e)}", color="danger")
    
    def _create_cross_attention_explorer_content(self, attention_data, image_data, question):
        """Create cross-attention explorer content with interactive token-to-image visualization."""
        try:
            if not attention_data:
                return self._create_no_data_message("cross-attention explorer")
            
            # Import the advanced analysis framework
            from components.advanced_analysis import AdvancedAnalysisManager
            analysis_manager = AdvancedAnalysisManager()
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return self._create_no_data_message("cross-attention explorer (invalid data)")
            
            # Check if cross-attention is available
            if attention_output.cross_attention is None:
                return dbc.Alert([
                    html.H5("⚠️ Cross-Attention Not Available", className="alert-heading"),
                    html.P("This analysis requires cross-modal attention data between text tokens and image patches."),
                    html.Hr(),
                    html.P("Possible reasons:", className="mb-2 fw-bold"),
                    html.Ul([
                        html.Li("GPU out of memory during attention extraction (most common for LLaVA-NeXT)"),
                        html.Li("The model architecture doesn't support attention extraction"),
                        html.Li("CUDA error during forward pass with attention"),
                        html.Li("The model requires more GPU memory than available (~23GB for LLaVA-NeXT-7B)")
                    ]),
                    html.Hr(),
                    html.P("💡 Solutions:", className="mb-2 fw-bold"),
                    html.Ul([
                        html.Li("Try a smaller model (e.g., LLaVA-v1.5 requires less memory)"),
                        html.Li("Use a machine with more GPU memory (24GB+ recommended)"),
                        html.Li("Check GPU memory availability with nvidia-smi"),
                        html.Li("Restart the application to clear GPU memory")
                    ]),
                    html.Hr(),
                    html.P("✅ Other analyses like 'Statistical Analysis' and 'Quick Stats' may still work.",
                          className="mb-0 text-success fw-bold")
                ], color="warning")
            
            # Process image if available
            image = None
            if image_data and 'contents' in image_data:
                image_content = image_data['contents']
                if ',' in image_content:
                    image_content = image_content.split(',')[1]
                
                image_bytes = base64.b64decode(image_content)
                image = Image.open(io.BytesIO(image_bytes))
            
            # Run cross-attention explorer analysis
            analysis_result = analysis_manager.run_analysis(
                analysis_type="cross_attention_explorer",
                attention_output=attention_output,
                image=image,
                question=question
            )
            
            return html.Div([
                html.H5([
                    html.I(className="fas fa-crosshairs me-2"),
                    "Cross-Attention Explorer"
                ], className="mb-3"),
                html.P(analysis_result.description, className="text-muted mb-4"),
                
                dcc.Graph(
                    figure=analysis_result.visualization,
                    config={
                        "displayModeBar": True, 
                        "toImageButtonOptions": {"format": "png", "height": 900, "width": 1400}
                    },
                    style={"height": "900px"}
                ),
                
                # Show key insights
                html.Hr(),
                html.H6("Token Attention Insights", className="mb-2"),
                html.Ul([html.Li(insight) for insight in analysis_result.insights]),
                
                # Show metrics if available
                html.Hr(),
                html.H6("Token Statistics", className="mb-2"),
                html.Div([
                    html.P([
                        html.Strong("Total Tokens: "),
                        str(len(analysis_result.metrics.get('token_statistics', [])))
                    ]),
                    html.P([
                        html.I(className="fas fa-info-circle me-2"),
                        html.Em("Use the dropdown menu above the visualization to select different tokens and see their attention patterns")
                    ], className="text-info")
                ])
            ])
                
        except Exception as e:
            logger.error(f"Error creating cross-attention explorer: {e}")
            import traceback
            traceback.print_exc()
            return dbc.Alert(f"Error creating cross-attention explorer: {str(e)}", color="danger")
    
    def _create_cross_attention_explorer_text_content(self, attention_data, image_data, question):
        """Create cross-attention explorer content with only question tokens (text-only mode)."""
        try:
            if not attention_data:
                return self._create_no_data_message("cross-attention explorer (question tokens only)")
            
            # Import the advanced analysis framework
            from components.advanced_analysis import AdvancedAnalysisManager
            analysis_manager = AdvancedAnalysisManager()
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return self._create_no_data_message("cross-attention explorer (invalid data)")
            
            # Check if cross-attention is available
            if attention_output.cross_attention is None:
                return dbc.Alert([
                    html.H5("⚠️ Cross-Attention Not Available", className="alert-heading"),
                    html.P("This analysis requires cross-modal attention data between text tokens and image patches."),
                    html.Hr(),
                    html.P("Possible reasons:", className="mb-2 fw-bold"),
                    html.Ul([
                        html.Li("GPU out of memory during attention extraction (most common for LLaVA-NeXT)"),
                        html.Li("The model architecture doesn't support attention extraction"),
                        html.Li("CUDA error during forward pass with attention"),
                        html.Li("The model requires more GPU memory than available (~23GB for LLaVA-NeXT-7B)")
                    ]),
                    html.Hr(),
                    html.P("💡 Solutions:", className="mb-2 fw-bold"),
                    html.Ul([
                        html.Li("Try a smaller model (e.g., LLaVA-v1.5 requires less memory)"),
                        html.Li("Use a machine with more GPU memory (24GB+ recommended)"),
                        html.Li("Check GPU memory availability with nvidia-smi"),
                        html.Li("Restart the application to clear GPU memory")
                    ]),
                    html.Hr(),
                    html.P("✅ Other analyses like 'Statistical Analysis' and 'Quick Stats' may still work.",
                          className="mb-0 text-success fw-bold")
                ], color="warning")
            
            # Process image if available
            image = None
            if image_data and 'contents' in image_data:
                image_content = image_data['contents']
                if ',' in image_content:
                    image_content = image_content.split(',')[1]
                
                image_bytes = base64.b64decode(image_content)
                image = Image.open(io.BytesIO(image_bytes))
            
            # Run cross-attention explorer analysis (text-only mode)
            analysis_result = analysis_manager.run_analysis(
                analysis_type="cross_attention_explorer_text",
                attention_output=attention_output,
                image=image,
                question=question
            )
            
            return html.Div([
                html.H5([
                    html.I(className="fas fa-comments me-2"),
                    "Cross-Attention Explorer (Question Tokens Only)"
                ], className="mb-3"),
                html.P(analysis_result.description, className="text-muted mb-4"),
                
                dcc.Graph(
                    figure=analysis_result.visualization,
                    config={
                        "displayModeBar": True, 
                        "toImageButtonOptions": {"format": "png", "height": 900, "width": 1400}
                    },
                    style={"height": "900px"}
                ),
                
                # Show key insights
                html.Hr(),
                html.H6("Question Token Attention Insights", className="mb-2"),
                html.Ul([html.Li(insight) for insight in analysis_result.insights]),
                
                # Show metrics if available
                html.Hr(),
                html.H6("Question Token Statistics", className="mb-2"),
                html.Div([
                    html.P([
                        html.Strong("Question Tokens: "),
                        str(len(analysis_result.metrics.get('token_statistics', [])))
                    ]),
                    html.P([
                        html.I(className="fas fa-info-circle me-2"),
                        html.Em("This view shows only the tokens from your input question, focusing on how they attend to the image regions.")
                    ], className="text-info")
                ])
            ])
                
        except Exception as e:
            logger.error(f"Error creating cross-attention explorer (text-only): {e}")
            import traceback
            traceback.print_exc()
            return dbc.Alert(f"Error creating cross-attention explorer (text-only): {str(e)}", color="danger")
    
    def _create_summary_statistics(self, attention_data):
        """Create summary statistics panel with real calculated metrics."""
        try:
            if not attention_data:
                return html.Div([
                    html.H6("Key Metrics", className="mb-3"),
                    dbc.Alert("No attention data available for statistics", color="info")
                ])
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return html.Div([
                    html.H6("Key Metrics", className="mb-3"),
                    dbc.Alert("Invalid attention data format", color="warning")
                ])
            
            # Get attention weights from the prepared object
            cross_attn = attention_output.cross_attention
                
            if cross_attn is None:
                return html.Div([
                    html.H6("Key Metrics", className="mb-3"),
                    dbc.Alert("No cross-attention data available", color="warning")
                ])
            
            # Process attention data - handle tensor or numpy array
            if hasattr(cross_attn, 'dim'):
                # PyTorch tensor
                while cross_attn.dim() > 2:
                    cross_attn = cross_attn.mean(dim=0)
                attn_weights = cross_attn.detach().cpu().numpy()
            else:
                # NumPy array or other
                attn_weights = np.array(cross_attn)
                while len(attn_weights.shape) > 2:
                    attn_weights = attn_weights.mean(axis=0)
            
            flat_weights = attn_weights.flatten()
            
            # Calculate entropy
            normalized_weights = flat_weights / np.sum(flat_weights)
            entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-12))
            
            # Calculate Gini coefficient (concentration)
            sorted_weights = np.sort(flat_weights)
            n = len(sorted_weights)
            gini = (2 * np.sum((np.arange(n) + 1) * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
            
            # Calculate top region coverage (top 10% of patches)
            top_10_percent = int(len(flat_weights) * 0.1)
            top_indices = np.argsort(flat_weights)[-top_10_percent:]
            top_coverage = np.sum(flat_weights[top_indices]) / np.sum(flat_weights) * 100
            
            # Get number of heads from the prepared object
            num_heads = attention_output.head_count if attention_output.head_count else 1
            
            return html.Div([
                html.H6("Key Metrics", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{entropy:.3f}", className="text-primary"),
                                html.P("Attention Entropy", className="mb-0 small")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{gini:.3f}", className="text-success"),
                                html.P("Focus Concentration", className="mb-0 small")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{top_coverage:.1f}%", className="text-warning"),
                                html.P("Top Region Coverage", className="mb-0 small")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{num_heads}", className="text-info"),
                                html.P("Attention Heads", className="mb-0 small")
                            ])
                        ])
                    ], width=3)
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return html.Div([
                html.H6("Key Metrics", className="mb-3"),
                dbc.Alert(f"Error calculating statistics: {str(e)}", color="danger")
            ])
    
    def _create_detailed_analysis(self, attention_data, analysis_type):
        """Create detailed analysis panel with real insights."""
        try:
            if not attention_data:
                return html.Div([
                    html.H6("Detailed Insights", className="mb-3"),
                    dbc.Alert("No attention data available", color="info")
                ])
            
            # Convert attention data to proper format
            attention_output = self._prepare_attention_data(attention_data)
            if not attention_output:
                return html.Div([
                    html.H6("Detailed Insights", className="mb-3"),
                    dbc.Alert("Invalid attention data format", color="warning")
                ])
            
            # Get attention weights from the prepared object
            cross_attn = attention_output.cross_attention
                
            if cross_attn is None:
                return html.Div([
                    html.H6("Detailed Insights", className="mb-3"),
                    dbc.Alert("No cross-attention data available", color="warning")
                ])
            
            # Process attention for analysis - handle tensor or numpy array
            if hasattr(cross_attn, 'dim'):
                # PyTorch tensor
                while cross_attn.dim() > 2:
                    cross_attn = cross_attn.mean(dim=0)
                attn_weights = cross_attn.detach().cpu().numpy()
            else:
                # NumPy array or other
                attn_weights = np.array(cross_attn)
                while len(attn_weights.shape) > 2:
                    attn_weights = attn_weights.mean(axis=0)
            
            # Calculate insights
            max_attention = np.max(attn_weights)
            mean_attention = np.mean(attn_weights)
            std_attention = np.std(attn_weights)
            
            # Find peak attention locations
            peak_idx = np.unravel_index(np.argmax(attn_weights), attn_weights.shape)
            
            insights = [
                f"Peak attention strength: {max_attention:.4f}",
                f"Average attention: {mean_attention:.4f}",
                f"Attention variance: {std_attention:.4f}",
                f"Peak location: ({peak_idx[0]}, {peak_idx[1]})"
            ]
            
            return html.Div([
                html.H6("Detailed Insights", className="mb-3"),
                html.Ul([
                    html.Li(insight) for insight in insights
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error creating detailed analysis: {e}")
            return html.Div([
                html.H6("Detailed Insights", className="mb-3"),
                dbc.Alert(f"Error generating insights: {str(e)}", color="danger")
            ])
    
    def _create_no_data_message(self, analysis_type):
        """Create a message for when no attention data is available."""
        return dbc.Alert([
            html.H4("📊 No Attention Data Available", className="alert-heading"),
            html.P(f"To view {analysis_type} analysis, please:"),
            html.Ol([
                html.Li("Select a model that supports attention extraction"),
                html.Li("Upload an image or select a dataset sample"),
                html.Li("Enter a question"),
                html.Li("Run a prediction to generate attention data")
            ]),
            html.Hr(),
            html.P("Once you have attention data, this analysis will show detailed insights into where the model focused while answering your question.", className="mb-0")
        ], color="info")
