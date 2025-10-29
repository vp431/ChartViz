"""
Results Section Component - Handles model predictions, attention analysis, and results display.
Extracted from app.py for better code organization.
"""
import base64
import io
import logging
import time

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, no_update
from PIL import Image

logger = logging.getLogger(__name__)


def create_results_section():
    """Create the results visualization section."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-brain me-2"),
                            "Model Prediction"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody(id="prediction-results")
                ])
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-chart-line me-2"),
                            "Attention Analysis"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody(id="attention-stats")
                ])
            ], width=8)
        ], className="g-3 mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(id="viz-card-header"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="visualization-loading",
                            children=html.Div(id="visualization-content"),
                            type="default"
                        )
                    ])
                ])
            ], width=12)
        ])
    ], id="results-section", style={"display": "none"})


def _serialize_attention_data(attention_output):
    """Convert AttentionOutput to a serializable format for Dash storage."""
    if attention_output is None:
        logger.warning("❌ _serialize_attention_data: attention_output is None")
        return None
    
    try:
        # Convert tensors to numpy arrays and then to lists
        serialized = {}
        
        # Always include cross_attention field, even if None
        if hasattr(attention_output, 'cross_attention'):
            if attention_output.cross_attention is not None:
                # Convert tensor to numpy and then to list for JSON serialization
                cross_attn = attention_output.cross_attention
                if hasattr(cross_attn, 'detach'):
                    cross_attn = cross_attn.detach().cpu().numpy()
                serialized['cross_attention'] = cross_attn.tolist()
                logger.info(f"✅ Serialized cross_attention: shape {cross_attn.shape}")
            else:
                serialized['cross_attention'] = None
                logger.warning("❌ cross_attention is None in AttentionOutput")
        else:
            logger.warning("❌ AttentionOutput has no cross_attention attribute")
            serialized['cross_attention'] = None
        
        if hasattr(attention_output, 'text_self_attention') and attention_output.text_self_attention is not None:
            text_self = attention_output.text_self_attention
            if hasattr(text_self, 'detach'):
                text_self = text_self.detach().cpu().numpy()
            serialized['text_self_attention'] = text_self.tolist()
        else:
            serialized['text_self_attention'] = None
        
        if hasattr(attention_output, 'image_self_attention') and attention_output.image_self_attention is not None:
            img_self = attention_output.image_self_attention
            if hasattr(img_self, 'detach'):
                img_self = img_self.detach().cpu().numpy()
            serialized['image_self_attention'] = img_self.tolist()
        else:
            serialized['image_self_attention'] = None
        
        # Copy simple attributes
        for attr in ['text_tokens', 'image_patch_coords', 'predicted_answer', 
                     'confidence_score', 'layer_count', 'head_count', 'image_size', 'patch_size']:
            if hasattr(attention_output, attr):
                value = getattr(attention_output, attr)
                serialized[attr] = value
            else:
                serialized[attr] = None
        
        logger.info(f"✅ Serialization complete. Keys: {list(serialized.keys())}")
        return serialized
        
    except Exception as e:
        logger.error(f"❌ Error serializing attention data: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None


def register_results_section_callbacks(app, model_manager):
    """Register all callbacks related to the results section."""
    
    @app.callback(
        [Output("answer-section", "style"),
         Output("answer-alert", "style"),
         Output("model-answer-content", "children"),
         Output("answer-verification", "children"),
         Output("current-attention-data", "data")],
        Input("ask-question-btn", "n_clicks"),
        [State("model-dropdown", "value"),
         State("data-source-dropdown", "value"),
         State("current-image-data", "data"),
         State("question-input", "value"),
         State("dataset-dropdown", "value"),
         State("dataset-sample-input", "value"),
         State("dataset-samples-data", "data")],
        prevent_initial_call=True
    )
    def ask_question(n_clicks, model_id, data_source, image_data, question, dataset_name, sample_id, samples_data):
        """
        Handle asking a question and getting the model's answer.
        Extracted from app.py for better organization.
        """
        if not n_clicks or not question.strip():
            return no_update, no_update, no_update, no_update, no_update
        
        if not model_id:
            error_msg = dbc.Alert("Please select a model first.", color="warning")
            return {"display": "block"}, {"display": "none"}, error_msg, "", None
        
        if not image_data:
            error_msg = dbc.Alert("Please upload an image or select a dataset sample first.", color="warning")
            return {"display": "block"}, {"display": "none"}, error_msg, "", None
        
        try:
            # Use the passed global model_manager instance
            # Load the selected model
            try:
                model = model_manager.load_model(model_id)
            except Exception as e:
                error_msg = dbc.Alert([
                    html.I(className="fas fa-exclamation-circle me-2"),
                    f"Failed to load model '{model_id}': {str(e)}"
                ], color="danger")
                return {"display": "block"}, {"display": "none"}, error_msg, "", None
            
            # Convert image data to PIL Image
            # Decode base64 image
            image_data_clean = image_data["contents"].split(",")[1]  # Remove data:image/...;base64, prefix
            image_bytes = base64.b64decode(image_data_clean)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Make prediction using the selected model
            prediction = model.predict(image, question)
            
            # Extract attention data if the model supports it
            attention_data = None
            try:
                if model.config.supports_attention_extraction:
                    logger.info(f"🔍 Extracting attention data for model: {model_id}")
                    raw_attention = model_manager.extract_attention(model_id, image, question)
                    logger.info(f"🔍 Raw attention extracted: {type(raw_attention)}")
                    
                    if raw_attention and hasattr(raw_attention, 'cross_attention'):
                        if raw_attention.cross_attention is not None:
                            logger.info(f"✅ Cross-attention found: shape {raw_attention.cross_attention.shape}")
                        else:
                            logger.warning("❌ Cross-attention is None - attention extraction did not produce cross-attention")
                    else:
                        logger.warning(f"❌ Invalid raw_attention object: {raw_attention}")
                    
                    # Convert to serializable format for Dash storage
                    # Even if cross_attention is None, we still serialize the AttentionOutput
                    attention_data = _serialize_attention_data(raw_attention)
                    logger.info(f"🔍 Serialized attention data: {type(attention_data)}")
                    if attention_data:
                        logger.info(f"🔍 Serialized keys: {list(attention_data.keys())}")
                        if 'cross_attention' in attention_data:
                            if attention_data['cross_attention'] is not None:
                                logger.info(f"✅ Serialized cross-attention: {len(attention_data['cross_attention'])} elements")
                            else:
                                logger.warning("❌ Serialized cross-attention is None")
                    else:
                        logger.warning("❌ Serialized attention_data is None!")
                else:
                    logger.warning(f"❌ Model {model_id} does not support attention extraction")
            except Exception as e:
                logger.error(f"❌ Failed to extract attention data: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Set attention_data to None on error
                attention_data = None
            
            answer_content = html.Div([
                html.P([
                    html.Strong("Question: "), 
                    question
                ], className="mb-2"),
                html.P([
                    html.Strong("Model Answer: "),
                    html.Span(prediction.answer, className="text-success")
                ], className="mb-2"),
                html.P([
                    html.Small(f"Model: {model_id} | Confidence: {prediction.confidence:.1%} | Time: {prediction.processing_time:.2f}s", 
                              className="text-muted")
                ])
            ])
            
            # Verification removed: no dataset answer checking
            verification_content = ""
            
            return (
                {"display": "block"},  # Show answer section
                {"display": "block"},  # Show answer alert
                answer_content,
                verification_content,
                attention_data  # Store attention data
            )
            
        except Exception as e:
            logger.error(f"Error getting model answer: {e}")
            error_msg = dbc.Alert(f"Error getting answer: {str(e)}", color="danger")
            return {"display": "block"}, {"display": "none"}, error_msg, "", None


def create_model_info_section():
    """Create model information display."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-microchip me-2"),
                "Available Models"
            ], className="mb-0")
        ]),
        dbc.CardBody(id="model-info-content")
    ], className="mb-4")


def register_results_callbacks(app, model_manager):
    """
    Register callbacks for results section.
    This is a wrapper to maintain backward compatibility.
    """
    register_results_section_callbacks(app, model_manager)

