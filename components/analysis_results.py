"""
Component for handling analysis results popup and visualization.
"""
import base64
import io
import logging

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback, no_update
import plotly.graph_objects as go
from PIL import Image

logger = logging.getLogger(__name__)


def create_analysis_modal():
    """Create the analysis results modal layout."""
    return dbc.Modal([
        dbc.ModalHeader([
            html.H4([
                html.I(className="fas fa-brain me-2"),
                "Analysis Results"
            ], className="mb-0")
        ], close_button=True),
        dbc.ModalBody([
            # Analysis type selector
            html.Div([
                html.H6("Analysis Type", className="mb-2"),
                dbc.ButtonGroup([
                    dbc.Button("Basic Analysis", id="basic-analysis-tab", color="primary", outline=True, size="sm"),
                    dbc.Button("Statistical", id="statistical-analysis-tab", color="info", outline=True, size="sm"),
                    dbc.Button("Attention Flow", id="flow-analysis-tab", color="success", outline=True, size="sm"),
                    dbc.Button("Layer Comparison", id="layer-analysis-tab", color="warning", outline=True, size="sm"),
                    dbc.Button("Multi-Head", id="multihead-analysis-tab", color="danger", outline=True, size="sm")
                ], className="mb-3")
            ]),
            
            dcc.Loading(
                id="analysis-loading",
                children=html.Div(id="analysis-modal-content"),
                type="default"
            )
        ], className="analysis-modal-body"),
        dbc.ModalFooter([
            dbc.ButtonGroup([
                dbc.Button([
                    html.I(className="fas fa-microscope me-1"),
                    "Advanced Analysis"
                ], id="open-advanced-analysis-btn", color="primary"),
                dbc.Button([
                    html.I(className="fas fa-download me-1"),
                    "Export Results"
                ], id="export-results-btn", color="secondary", outline=True),
                dbc.Button("Close", id="close-analysis-modal", color="secondary", className="ms-2")
            ])
        ])
    ], id="analysis-modal", size="xl", centered=True, backdrop="static")


def create_analysis_content(model_id, question, prediction_answer, confidence, processing_time, viz_type, visualization_figure=None):
    """Create the content for the analysis modal."""
    return html.Div([
        # Stats row
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span(prediction_answer, className="fs-5 fw-bold text-primary"),
                        html.Br(),
                        html.Span("Model Prediction", className="stat-label")
                    ], className="stats-panel text-start")
                ], width=8),
                dbc.Col([
                    html.Div([
                        html.Span(f"{confidence:.1%}", className="stat-value"),
                        html.Br(),
                        html.Span("Confidence", className="stat-label")
                    ], className="stats-panel")
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.Span(f"{processing_time:.1f}s", className="stat-value"),
                        html.Br(),
                        html.Span("Time", className="stat-label")
                    ], className="stats-panel")
                ], width=2)
            ], className="mb-4")
        ], className="analysis-stats-grid"),
        
        # Model and Question Information
        dbc.Card([
            dbc.CardBody([
                html.H6("Analysis Details", className="mb-3"),
                html.P([
                    html.Strong("Model: "), model_id
                ], className="mb-2"),
                html.P([
                    html.Strong("Question: "), question
                ], className="mb-2"),
                html.P([
                    html.Strong("Visualization Type: "), viz_type.replace('_', ' ').title()
                ], className="mb-0"),
            ])
        ], className="mb-4", color="light"),
        
        # Visualization
        html.Div([
            html.H5([
                html.I(className="fas fa-eye me-2"),
                f"{viz_type.replace('_', ' ').title() if viz_type else 'Attention'} Visualization"
            ], className="mb-3"),
            
            # Visualization graph
            dcc.Graph(
                figure=visualization_figure or create_no_data_placeholder(viz_type),
                style={"height": "500px"}
            )
        ], className="analysis-chart-container")
    ])


def create_sample_visualization(viz_type):
    """Create a sample visualization based on the visualization type."""
    if viz_type == "heatmap":
        # Sample attention heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[[1, 20, 30],
               [20, 1, 60],
               [30, 60, 1]],
            colorscale='Viridis',
            showscale=True
        ))
        fig.update_layout(
            title="Sample Attention Heatmap",
            xaxis_title="Tokens",
            yaxis_title="Attention Heads"
        )
    
    elif viz_type == "token_importance":
        # Sample token importance bars
        tokens = ["What", "is", "the", "highest", "bar", "?"]
        importance = [0.1, 0.2, 0.3, 0.9, 0.8, 0.1]
        
        fig = go.Figure(data=go.Bar(
            x=tokens,
            y=importance,
            marker_color='lightblue'
        ))
        fig.update_layout(
            title="Sample Token Importance",
            xaxis_title="Tokens",
            yaxis_title="Importance Score"
        )
    
    elif viz_type == "attention_flow":
        # Sample attention flow network
        fig = go.Figure(data=go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[2, 4, 3, 5, 1],
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(width=3)
        ))
        fig.update_layout(
            title="Sample Attention Flow",
            xaxis_title="Token Position",
            yaxis_title="Attention Weight"
        )
    
    elif viz_type == "multi_head":
        # Sample multi-head comparison
        fig = go.Figure()
        for i in range(1, 5):
            fig.add_trace(go.Scatter(
                x=[1, 2, 3, 4, 5],
                y=[i*0.2 + j*0.1 for j in range(5)],
                mode='lines+markers',
                name=f'Head {i}'
            ))
        fig.update_layout(
            title="Sample Multi-Head Attention",
            xaxis_title="Token Position",
            yaxis_title="Attention Weight"
        )
    
    elif viz_type == "layer_comparison":
        # Sample layer comparison
        fig = go.Figure()
        layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
        values = [0.2, 0.6, 0.8, 0.4]
        
        fig.add_trace(go.Bar(
            x=layers,
            y=values,
            marker_color='lightcoral'
        ))
        fig.update_layout(
            title="Sample Layer-wise Attention",
            xaxis_title="Model Layers",
            yaxis_title="Average Attention"
        )
    
    else:
        # Default scatter plot
        fig = go.Figure(data=go.Scatter(
            x=[1, 2, 3, 4],
            y=[10, 11, 12, 13],
            mode='markers',
            marker=dict(size=10)
        ))
        fig.update_layout(title="Sample Attention Visualization")
    
    return fig


def create_no_data_placeholder(viz_type):
    """Create a placeholder when no attention data is available."""
    fig = go.Figure()
    fig.update_layout(
        title=f"{viz_type.replace('_', ' ').title() if viz_type else 'Attention'} Analysis",
        annotations=[{
            'x': 0.5, 'y': 0.5, 
            'text': "Run model prediction to generate attention data<br>Then use the 'Advanced Analysis' button for detailed visualization",
            'showarrow': False, 'xref': 'paper', 'yref': 'paper',
            'font': {'size': 16, 'color': '#666'},
            'xanchor': 'center', 'yanchor': 'middle'
        }],
        xaxis={'visible': False},
        yaxis={'visible': False},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_advanced_analysis_content(analysis_type: str, analysis_result: dict, 
                                   model_id: str, question: str):
    """Create content for advanced analysis results."""
    
    # Analysis type info
    type_info_map = {
        'statistical': {
            'icon': 'fas fa-chart-bar',
            'color': 'info',
            'title': 'Statistical Analysis',
            'description': 'Comprehensive statistical metrics and distribution analysis'
        },
        'attention_flow': {
            'icon': 'fas fa-stream',
            'color': 'success',
            'title': 'Attention Flow Analysis',
            'description': 'Analysis of attention flow between text tokens and image regions'
        },
        'layer_comparison': {
            'icon': 'fas fa-layer-group',
            'color': 'warning',
            'title': 'Layer Comparison Analysis',
            'description': 'Comparison of attention patterns across model layers'
        },
        'multi_head': {
            'icon': 'fas fa-sitemap',
            'color': 'danger',
            'title': 'Multi-Head Analysis',
            'description': 'Analysis of attention patterns across different attention heads'
        }
    }
    
    type_info = type_info_map.get(analysis_type, {
        'icon': 'fas fa-brain',
        'color': 'primary',
        'title': 'Advanced Analysis',
        'description': 'Advanced model analysis results'
    })
    
    content = [
        # Analysis type header
        dbc.Alert([
            html.Div([
                html.I(className=f"{type_info['icon']} fa-2x me-3"),
                html.Div([
                    html.H5(type_info['title'], className="mb-1"),
                    html.P(type_info['description'], className="mb-0 small")
                ])
            ], className="d-flex align-items-center")
        ], color=type_info['color'], className="mb-4"),
        
        # Model and question info
        dbc.Card([
            dbc.CardBody([
                html.H6("Analysis Context", className="mb-3"),
                html.P([html.Strong("Model: "), model_id], className="mb-2"),
                html.P([html.Strong("Question: "), question], className="mb-0")
            ])
        ], className="mb-4", color="light"),
    ]
    
    # Add metrics if available
    if analysis_result.get('metrics'):
        metrics = analysis_result['metrics']
        metric_cards = []
        
        # Create metric cards for key metrics
        key_metrics = [
            ('attention_mean', 'Average Attention', 'fas fa-chart-line'),
            ('attention_entropy', 'Attention Entropy', 'fas fa-random'),
            ('attention_gini', 'Concentration', 'fas fa-bullseye'),
            ('attention_sparsity', 'Sparsity', 'fas fa-th-large')
        ]
        
        for metric_key, metric_label, icon in key_metrics:
            if metric_key in metrics:
                value = metrics[metric_key]
                display_value = f"{value:.3f}" if isinstance(value, float) else str(value)
                
                metric_cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className=f"{icon} fa-2x text-muted mb-2"),
                                    html.H4(display_value, className="text-primary"),
                                    html.P(metric_label, className="mb-0 small text-muted")
                                ], className="text-center")
                            ])
                        ])
                    ], width=3)
                )
        
        if metric_cards:
            content.append(
                html.Div([
                    html.H5("Key Metrics", className="mb-3"),
                    dbc.Row(metric_cards)
                ], className="mb-4")
            )
    
    # Add visualization if available
    if analysis_result.get('visualization_data'):
        try:
            # Convert visualization data back to plotly figure
            fig_dict = analysis_result['visualization_data']
            fig = go.Figure(fig_dict)
            
            content.append(
                html.Div([
                    html.H5("Analysis Visualization", className="mb-3"),
                    dcc.Graph(
                        figure=fig,
                        config={
                            "displayModeBar": True,
                            "toImageButtonOptions": {"format": "png", "filename": f"{analysis_type}_analysis"}
                        }
                    )
                ], className="mb-4")
            )
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            content.append(
                dbc.Alert(f"Error creating visualization: {str(e)}", color="warning", className="mb-4")
            )
    
    # Add insights if available
    if analysis_result.get('insights'):
        insights = analysis_result['insights']
        insight_items = []
        
        for insight in insights:
            insight_items.append(
                dbc.ListGroupItem([
                    html.Div([
                        html.I(className="fas fa-lightbulb text-warning me-2"),
                        html.Span(insight)
                    ])
                ])
            )
        
        content.append(
            html.Div([
                html.H5("Analysis Insights", className="mb-3"),
                dbc.ListGroup(insight_items, flush=True)
            ], className="mb-4")
        )
    
    return html.Div(content)


def register_callbacks(app, model_manager=None):
    """Register callbacks for the analysis results component."""
    
    @app.callback(
        [Output("analysis-modal", "is_open"),
         Output("analysis-modal-content", "children")],
        [Input("close-analysis-modal", "n_clicks"),
         Input("basic-analysis-tab", "n_clicks"),
         Input("statistical-analysis-tab", "n_clicks"),
         Input("flow-analysis-tab", "n_clicks"),
         Input("layer-analysis-tab", "n_clicks"),
         Input("multihead-analysis-tab", "n_clicks")],
        [State("model-dropdown", "value"),
         State("data-source-dropdown", "value"),
         State("current-image-data", "data"),
         State("question-input", "value"),
         State("dataset-dropdown", "value"),
         State("dataset-sample-input", "value"),
         State("current-attention-data", "data"),
         State("analysis-modal", "is_open")],
        prevent_initial_call=True
    )
    def handle_analysis_modal(close_clicks, basic_clicks, stat_clicks, 
                            flow_clicks, layer_clicks, head_clicks, model_id, data_source, 
                            image_data, question, dataset_name, sample_id, attention_data, is_open):
        """Handle opening and closing the analysis modal with proper analysis."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return False, ""
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Close modal
        if trigger_id == "close-analysis-modal":
            return False, ""
        
        # Determine analysis type
        analysis_type_map = {
            "basic-analysis-tab": "basic",
            "statistical-analysis-tab": "statistical",
            "flow-analysis-tab": "attention_flow",
            "layer-analysis-tab": "layer_comparison",
            "multihead-analysis-tab": "multi_head"
        }
        
        analysis_type = analysis_type_map.get(trigger_id, "basic")
        
        # Open modal and run analysis
        if trigger_id in analysis_type_map:
            # Validation
            if not model_id:
                error_content = dbc.Alert("Please select a model first.", color="warning")
                return True, error_content
            
            if not image_data or not question.strip():
                error_content = dbc.Alert("Please ensure you have an image loaded and a question entered.", color="warning")
                return True, error_content
            
            try:
                # Use the passed global model_manager or fallback 
                if model_manager is None:
                    from models import ModelManager
                    current_model_manager = ModelManager()
                else:
                    current_model_manager = model_manager
                
                # Load the selected model
                try:
                    model = current_model_manager.load_model(model_id)
                except Exception as e:
                    error_msg = dbc.Alert([
                        html.I(className="fas fa-exclamation-circle me-2"),
                        f"Failed to load model '{model_id}': {str(e)}"
                    ], color="danger")
                    return False, error_msg
                
                # Convert image data to PIL Image
                import base64
                import io
                from PIL import Image
                
                # Decode base64 image
                image_data_clean = image_data["contents"].split(",")[1]  # Remove data:image/...;base64, prefix
                image_bytes = base64.b64decode(image_data_clean)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Handle different analysis types
                if analysis_type == "basic":
                    # Basic analysis - make prediction and show basic results
                    prediction = model.predict(image, question)
                    
                    prediction_answer = prediction.answer
                    confidence = prediction.confidence
                    processing_time = prediction.processing_time
                    
                    # Create visualization based on type
                    viz_figure = create_no_data_placeholder("heatmap")
                    
                    # Create analysis content
                    modal_content = create_analysis_content(
                        model_id=model_id,
                        question=question,
                        prediction_answer=prediction_answer,
                        confidence=confidence,
                        processing_time=processing_time,
                        viz_type="heatmap",
                        visualization_figure=viz_figure
                    )
                
                else:
                    # Advanced analysis types
                    if not hasattr(model, 'run_advanced_analysis'):
                        error_content = dbc.Alert(f"Model '{model_id}' does not support advanced analysis.", color="warning")
                        return True, error_content
                    
                    if not attention_data:
                        error_content = dbc.Alert("No attention data available. Please run a basic prediction first.", color="warning")
                        return True, error_content
                    
                    try:
                        # Run advanced analysis
                        analysis_result = model.run_advanced_analysis(
                            analysis_type=analysis_type,
                            image=image,
                            question=question
                        )
                        
                        # Create advanced analysis content
                        modal_content = create_advanced_analysis_content(
                            analysis_type=analysis_type,
                            analysis_result=analysis_result,
                            model_id=model_id,
                            question=question
                        )
                        
                    except Exception as e:
                        logger.error(f"Advanced analysis failed: {e}")
                        error_content = dbc.Alert([
                            html.H4("Advanced Analysis Error", className="alert-heading"),
                            html.P(f"An error occurred during advanced analysis: {str(e)}")
                        ], color="danger")
                        return True, error_content
                
                # Store attention data for potential export
                from datetime import datetime
                attention_data = {
                    "model_id": model_id,
                    "question": question,
                    "prediction": prediction_answer,
                    "confidence": confidence,
                    "viz_type": "heatmap",
                    "timestamp": datetime.now().isoformat()
                }
                
                return True, modal_content
                
            except Exception as e:
                logger.error(f"Error during analysis: {e}")
                error_content = dbc.Alert([
                    html.H4("Analysis Error", className="alert-heading"),
                    html.P(f"An error occurred during analysis: {str(e)}")
                ], color="danger")
                
                return True, error_content
        
        return is_open, no_update
    
    @app.callback(
        Output("export-results-btn", "n_clicks"),
        Input("export-results-btn", "n_clicks"),
        State("current-attention-data", "data"),
        prevent_initial_call=True
    )
    def export_analysis_results(n_clicks, attention_data):
        """Handle exporting analysis results (placeholder for now)."""
        if n_clicks and attention_data:
            logger.info(f"Export requested for analysis: {attention_data}")
            # TODO: Implement actual export functionality
        return 0





