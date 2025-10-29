"""
Core attention visualization component for chart QA explainability.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import cv2
from PIL import Image
import torch

from models.base_model import AttentionOutput
from config import config


class AttentionVisualizer:
    """Core class for visualizing attention maps and patterns in chart QA models."""
    
    def __init__(self, colorscale: Optional[str] = None):
        """
        Initialize the attention visualizer.
        
        Args:
            colorscale: Plotly colorscale for attention heatmaps
        """
        self.colorscale = colorscale or "Viridis"
        self.viz_config = config.visualization
    
    def create_attention_heatmap(self, 
                               attention_output: AttentionOutput,
                               image: Image.Image,
                               question: str,
                               layer_idx: Optional[int] = None,
                               head_idx: Optional[int] = None) -> go.Figure:
        """
        Create an interactive attention heatmap overlay on the chart image.
        
        Args:
            attention_output: Attention analysis results
            image: Original chart image
            question: Question text
            layer_idx: Specific layer to visualize (None for average)
            head_idx: Specific attention head (None for average)
            
        Returns:
            Plotly figure with attention heatmap
        """
        if attention_output.cross_attention is None:
            raise ValueError("No cross-attention data available")
        
        # Get cross-attention weights [text_tokens, image_patches]
        cross_attn = attention_output.cross_attention
        
        # Handle multi-layer, multi-head attention
        if cross_attn.dim() > 2:
            if layer_idx is not None and cross_attn.dim() >= 3:
                cross_attn = cross_attn[layer_idx]
            if head_idx is not None and cross_attn.dim() >= 3:
                cross_attn = cross_attn[:, head_idx] if cross_attn.dim() == 3 else cross_attn[head_idx]
            
            # Average over remaining dimensions if needed
            while cross_attn.dim() > 2:
                cross_attn = cross_attn.mean(dim=0)
        
        # Convert to numpy
        attn_weights = cross_attn.detach().cpu().numpy()
        
        # Sum attention across text tokens to get image attention
        image_attention = np.sum(attn_weights, axis=0)  # [image_patches]
        
        # Create attention heatmap
        heatmap = self._create_patch_heatmap(
            image_attention, 
            attention_output.image_size,
            attention_output.patch_size
        )
        
        # Create figure with image and overlay
        fig = self._create_overlay_figure(image, heatmap, question, attention_output)
        
        return fig
    
    def create_token_attention_plot(self, 
                                  attention_output: AttentionOutput,
                                  question: str) -> go.Figure:
        """
        Create a bar plot showing attention weights for each text token.
        
        Args:
            attention_output: Attention analysis results
            question: Question text
            
        Returns:
            Plotly figure with token attention bars
        """
        if attention_output.cross_attention is None or attention_output.text_tokens is None:
            raise ValueError("Cross-attention data and text tokens required")
        
        # Get cross-attention and average over image patches
        cross_attn = attention_output.cross_attention
        if cross_attn.dim() > 2:
            while cross_attn.dim() > 2:
                cross_attn = cross_attn.mean(dim=0)
        
        attn_weights = cross_attn.detach().cpu().numpy()
        token_importance = np.sum(attn_weights, axis=1)  # Sum over image patches
        
        # Normalize
        token_importance = token_importance / np.sum(token_importance)
        
        # Create bar plot
        fig = go.Figure(data=[
            go.Bar(
                x=attention_output.text_tokens,
                y=token_importance,
                marker=dict(
                    color=token_importance,
                    colorscale=self.colorscale,
                    showscale=True,
                    colorbar=dict(title="Attention Weight")
                ),
                hovertemplate="<b>%{x}</b><br>Attention: %{y:.3f}<extra></extra>"
            )
        ])
        
        fig.update_layout(
            title=f"Token Attention Weights: {question}",
            xaxis_title="Tokens",
            yaxis_title="Attention Weight",
            height=400,
            showlegend=False
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_attention_flow_diagram(self, 
                                    attention_output: AttentionOutput,
                                    top_k_tokens: int = 5,
                                    top_k_patches: int = 10) -> go.Figure:
        """
        Create a flow diagram showing attention between top tokens and image patches.
        
        Args:
            attention_output: Attention analysis results
            top_k_tokens: Number of top tokens to show
            top_k_patches: Number of top patches to show
            
        Returns:
            Plotly figure with attention flow diagram
        """
        if attention_output.cross_attention is None:
            raise ValueError("Cross-attention data required")
        
        # Get attention weights
        cross_attn = attention_output.cross_attention
        if cross_attn.dim() > 2:
            while cross_attn.dim() > 2:
                cross_attn = cross_attn.mean(dim=0)
        
        attn_weights = cross_attn.detach().cpu().numpy()
        
        # Get top tokens and patches
        token_importance = np.sum(attn_weights, axis=1)
        top_token_indices = np.argsort(token_importance)[-top_k_tokens:]
        
        patch_importance = np.sum(attn_weights, axis=0)
        top_patch_indices = np.argsort(patch_importance)[-top_k_patches:]
        
        # Create Sankey diagram
        fig = self._create_sankey_diagram(
            attention_output, 
            attn_weights,
            top_token_indices, 
            top_patch_indices
        )
        
        return fig
    
    def create_multi_head_comparison(self, 
                                   attention_output: AttentionOutput,
                                   image: Image.Image,
                                   max_heads: int = 8) -> go.Figure:
        """
        Create a comparison of attention patterns across multiple heads.
        
        Args:
            attention_output: Attention analysis results
            image: Original chart image
            max_heads: Maximum number of heads to display
            
        Returns:
            Plotly figure with multi-head comparison
        """
        if attention_output.cross_attention is None:
            raise ValueError("Cross-attention data required")
        
        cross_attn = attention_output.cross_attention
        
        # Ensure we have head dimension
        if cross_attn.dim() < 3:
            raise ValueError("Multi-head attention data required")
        
        # Get number of heads
        if cross_attn.dim() == 4:  # [layers, heads, text, image]
            num_heads = cross_attn.shape[1]
            # Use last layer
            cross_attn = cross_attn[-1]
        else:  # [heads, text, image]
            num_heads = cross_attn.shape[0]
        
        num_heads = min(num_heads, max_heads)
        
        # Create subplots
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Head {i+1}" for i in range(num_heads)],
            specs=[[{"type": "xy"} for _ in range(cols)] for _ in range(rows)]
        )
        
        for head_idx in range(num_heads):
            row = head_idx // cols + 1
            col = head_idx % cols + 1
            
            # Get attention for this head
            head_attn = cross_attn[head_idx]
            attn_weights = head_attn.detach().cpu().numpy()
            
            # Sum over text tokens to get image attention
            image_attention = np.sum(attn_weights, axis=0)
            
            # Create heatmap for this head
            heatmap = self._create_patch_heatmap(
                image_attention,
                attention_output.image_size,
                attention_output.patch_size
            )
            
            # Add heatmap to subplot
            fig.add_trace(
                go.Heatmap(
                    z=heatmap,
                    colorscale=self.colorscale,
                    showscale=(head_idx == 0),  # Only show colorbar for first head
                    hovertemplate=f"Head {head_idx+1}<br>Attention: %{{z:.3f}}<extra></extra>"
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Multi-Head Attention Comparison",
            height=300 * rows,
            showlegend=False
        )
        
        return fig
    
    def _create_patch_heatmap(self, 
                            attention_weights: np.ndarray,
                            image_size: Tuple[int, int],
                            patch_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert 1D patch attention weights to 2D heatmap.
        
        Args:
            attention_weights: 1D array of attention weights for patches
            image_size: Original image size (width, height)
            patch_size: Patch size (width, height)
            
        Returns:
            2D heatmap array
        """
        patch_w, patch_h = patch_size
        img_w, img_h = image_size
        
        # Calculate grid dimensions
        grid_w = img_w // patch_w
        grid_h = img_h // patch_h
        
        # Reshape to grid
        if len(attention_weights) != grid_w * grid_h:
            # Truncate or pad as needed
            expected_patches = grid_w * grid_h
            if len(attention_weights) > expected_patches:
                attention_weights = attention_weights[:expected_patches]
            else:
                padded = np.zeros(expected_patches)
                padded[:len(attention_weights)] = attention_weights
                attention_weights = padded
        
        heatmap = attention_weights.reshape(grid_h, grid_w)
        
        # Normalize to 0-1 range
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap
    
    def _create_overlay_figure(self, 
                             image: Image.Image,
                             heatmap: np.ndarray,
                             question: str,
                             attention_output: AttentionOutput) -> go.Figure:
        """
        Create a figure with image and attention heatmap overlay.
        
        Args:
            image: Original chart image
            heatmap: Attention heatmap
            question: Question text
            attention_output: Attention analysis results
            
        Returns:
            Plotly figure with overlay
        """
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Create figure
        fig = go.Figure()
        
        # Add original image
        fig.add_trace(go.Image(z=img_array, name="Chart"))
        
        # Add attention heatmap overlay
        fig.add_trace(
            go.Heatmap(
                z=heatmap,
                colorscale=self.colorscale,
                opacity=self.viz_config.overlay_alpha,
                showscale=True,
                colorbar=dict(
                    title="Attention",
                    x=1.02,
                    len=0.7
                ),
                hovertemplate="Attention: %{z:.3f}<extra></extra>",
                name="Attention"
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Attention Map: {question}",
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            height=600,
            margin=dict(l=0, r=80, t=50, b=0),
            showlegend=False
        )
        
        # Add prediction info as annotation
        if attention_output.predicted_answer:
            fig.add_annotation(
                text=f"Predicted: {attention_output.predicted_answer}<br>"
                     f"Confidence: {attention_output.confidence_score:.2f}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=12)
            )
        
        return fig
    
    def _create_sankey_diagram(self, 
                             attention_output: AttentionOutput,
                             attn_weights: np.ndarray,
                             top_token_indices: np.ndarray,
                             top_patch_indices: np.ndarray) -> go.Figure:
        """Create a Sankey diagram showing attention flow."""
        
        # Prepare nodes
        token_labels = [attention_output.text_tokens[i] for i in top_token_indices]
        patch_labels = [f"Patch {i}" for i in top_patch_indices]
        
        all_labels = token_labels + patch_labels
        
        # Prepare links
        source = []
        target = []
        value = []
        
        for i, token_idx in enumerate(top_token_indices):
            for j, patch_idx in enumerate(top_patch_indices):
                attention_weight = attn_weights[token_idx, patch_idx]
                if attention_weight > 0.01:  # Only show significant connections
                    source.append(i)  # Token index
                    target.append(len(token_labels) + j)  # Patch index (offset)
                    value.append(attention_weight)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color="lightblue"
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color="rgba(0,0,255,0.3)"
            )
        )])
        
        fig.update_layout(
            title="Attention Flow: Tokens → Image Patches",
            height=500,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def save_attention_analysis(self, 
                              attention_output: AttentionOutput,
                              save_path: str,
                              format: str = "html") -> None:
        """
        Save attention analysis to file.
        
        Args:
            attention_output: Attention analysis results
            save_path: Path to save file
            format: Output format ('html', 'png', 'pdf')
        """
        # This could be extended to save various analysis outputs
        # For now, we'll focus on the core visualization functionality
        pass

