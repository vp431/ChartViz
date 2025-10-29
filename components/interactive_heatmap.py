"""
Interactive heatmap component for detailed attention analysis.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union, Callable
import pandas as pd

from models.base_model import AttentionOutput


class InteractiveHeatmap:
    """Interactive heatmap component for attention visualization with rich interactions."""
    
    def __init__(self, colorscale: str = "Viridis"):
        """
        Initialize interactive heatmap component.
        
        Args:
            colorscale: Plotly colorscale for heatmaps
        """
        self.colorscale = colorscale
    
    def create_layered_attention_heatmap(self, 
                                       attention_output: AttentionOutput,
                                       layer_names: Optional[List[str]] = None) -> go.Figure:
        """
        Create an interactive heatmap showing attention across multiple layers.
        
        Args:
            attention_output: Attention analysis results
            layer_names: Names for each layer (auto-generated if None)
            
        Returns:
            Plotly figure with layered attention heatmap
        """
        if attention_output.cross_attention is None:
            raise ValueError("Cross-attention data required")
        
        cross_attn = attention_output.cross_attention
        
        # Ensure we have layer dimension
        if cross_attn.dim() < 3:
            raise ValueError("Multi-layer attention data required")
        
        # Handle different tensor shapes
        if cross_attn.dim() == 4:  # [layers, heads, text, image]
            # Average over heads
            layer_attention = cross_attn.mean(dim=1)  # [layers, text, image]
        else:  # [layers, text, image]
            layer_attention = cross_attn
        
        num_layers = layer_attention.shape[0]
        
        if layer_names is None:
            layer_names = [f"Layer {i+1}" for i in range(num_layers)]
        
        # Convert to numpy
        layer_attention_np = layer_attention.detach().cpu().numpy()
        
        # Create dropdown for layer selection
        buttons = []
        for i, layer_name in enumerate(layer_names):
            # Sum over text tokens to get image attention
            image_attention = np.sum(layer_attention_np[i], axis=0)
            
            # Reshape to grid
            heatmap = self._create_heatmap_grid(
                image_attention, 
                attention_output.image_size,
                attention_output.patch_size
            )
            
            buttons.append({
                'label': layer_name,
                'method': 'restyle',
                'args': [{'z': [heatmap]}]
            })
        
        # Create initial heatmap (first layer)
        initial_attention = np.sum(layer_attention_np[0], axis=0)
        initial_heatmap = self._create_heatmap_grid(
            initial_attention,
            attention_output.image_size, 
            attention_output.patch_size
        )
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=initial_heatmap,
            colorscale=self.colorscale,
            showscale=True,
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Attention: %{z:.3f}<extra></extra>"
        ))
        
        # Add dropdown menu
        fig.update_layout(
            title="Layer-wise Attention Analysis",
            updatemenus=[{
                'buttons': buttons,
                'direction': 'down',
                'showactive': True,
                'x': 1.02,
                'xanchor': 'left',
                'y': 1,
                'yanchor': 'top'
            }],
            height=600,
            margin=dict(r=150)
        )
        
        return fig
    
    def create_token_to_patch_heatmap(self, 
                                    attention_output: AttentionOutput,
                                    max_tokens: int = 20) -> go.Figure:
        """
        Create detailed token-to-patch attention heatmap.
        
        Args:
            attention_output: Attention analysis results
            max_tokens: Maximum number of tokens to display
            
        Returns:
            Plotly figure with token-to-patch heatmap
        """
        if attention_output.cross_attention is None or attention_output.text_tokens is None:
            raise ValueError("Cross-attention data and text tokens required")
        
        cross_attn = attention_output.cross_attention
        
        # Handle multi-dimensional attention
        if cross_attn.dim() > 2:
            # Average over extra dimensions
            while cross_attn.dim() > 2:
                cross_attn = cross_attn.mean(dim=0)
        
        attn_weights = cross_attn.detach().cpu().numpy()
        
        # Limit number of tokens for readability
        num_tokens = min(len(attention_output.text_tokens), max_tokens)
        tokens = attention_output.text_tokens[:num_tokens]
        weights = attn_weights[:num_tokens]
        
        # Create patch labels
        num_patches = weights.shape[1]
        patch_labels = [f"Patch {i}" for i in range(num_patches)]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=weights,
            x=patch_labels,
            y=tokens,
            colorscale=self.colorscale,
            showscale=True,
            hovertemplate="Token: %{y}<br>Patch: %{x}<br>Attention: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Token-to-Patch Attention Matrix",
            xaxis_title="Image Patches",
            yaxis_title="Text Tokens",
            height=max(400, num_tokens * 20),
            margin=dict(l=100, r=50, t=50, b=50)
        )
        
        return fig
    
    def create_head_comparison_grid(self, 
                                  attention_output: AttentionOutput,
                                  max_heads: int = 12) -> go.Figure:
        """
        Create a grid comparing attention patterns across heads.
        
        Args:
            attention_output: Attention analysis results
            max_heads: Maximum number of heads to display
            
        Returns:
            Plotly figure with head comparison grid
        """
        if attention_output.cross_attention is None:
            raise ValueError("Cross-attention data required")
        
        cross_attn = attention_output.cross_attention
        
        # Ensure we have head dimension
        if cross_attn.dim() < 3:
            raise ValueError("Multi-head attention data required")
        
        if cross_attn.dim() == 4:  # [layers, heads, text, image]
            # Use last layer
            cross_attn = cross_attn[-1]
        
        num_heads = min(cross_attn.shape[0], max_heads)
        
        # Calculate grid layout
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        # Create subplots
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Head {i+1}" for i in range(num_heads)],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for head_idx in range(num_heads):
            row = head_idx // cols + 1
            col = head_idx % cols + 1
            
            # Get attention for this head
            head_attn = cross_attn[head_idx]
            attn_weights = head_attn.detach().cpu().numpy()
            
            # Sum over text tokens to get image attention
            image_attention = np.sum(attn_weights, axis=0)
            
            # Create heatmap
            heatmap = self._create_heatmap_grid(
                image_attention,
                attention_output.image_size,
                attention_output.patch_size
            )
            
            # Add to subplot
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
            height=200 * rows + 100,
            showlegend=False
        )
        
        return fig
    
    def create_attention_statistics_plot(self, 
                                       attention_outputs: List[AttentionOutput],
                                       model_names: List[str]) -> go.Figure:
        """
        Create comparative statistics plot for multiple models.
        
        Args:
            attention_outputs: List of attention analysis results
            model_names: Names of the models
            
        Returns:
            Plotly figure with attention statistics
        """
        if len(attention_outputs) != len(model_names):
            raise ValueError("Number of attention outputs must match model names")
        
        # Calculate statistics for each model
        stats_data = []
        
        for output, model_name in zip(attention_outputs, model_names):
            if output.cross_attention is None:
                continue
            
            cross_attn = output.cross_attention
            if cross_attn.dim() > 2:
                while cross_attn.dim() > 2:
                    cross_attn = cross_attn.mean(dim=0)
            
            attn_weights = cross_attn.detach().cpu().numpy()
            
            # Calculate various statistics
            # Token importance (sum over image patches)
            token_importance = np.sum(attn_weights, axis=1)
            
            # Image patch importance (sum over text tokens)
            patch_importance = np.sum(attn_weights, axis=0)
            
            # Attention entropy
            normalized_weights = attn_weights / np.sum(attn_weights)
            entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-12))
            
            # Max attention value
            max_attention = np.max(attn_weights)
            
            # Concentration (Gini coefficient)
            flat_weights = attn_weights.flatten()
            sorted_weights = np.sort(flat_weights)
            n = len(sorted_weights)
            gini = (2 * np.sum((np.arange(n) + 1) * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
            
            stats_data.append({
                'Model': model_name,
                'Entropy': entropy,
                'Max Attention': max_attention,
                'Gini Coefficient': gini,
                'Token Variance': np.var(token_importance),
                'Patch Variance': np.var(patch_importance)
            })
        
        if not stats_data:
            raise ValueError("No valid attention data found")
        
        # Create DataFrame
        df = pd.DataFrame(stats_data)
        
        # Create subplot figure
        metrics = ['Entropy', 'Max Attention', 'Gini Coefficient', 'Token Variance', 'Patch Variance']
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metrics,
            specs=[[{"type": "bar"} for _ in range(3)] for _ in range(2)]
        )
        
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            fig.add_trace(
                go.Bar(
                    x=df['Model'],
                    y=df[metric],
                    name=metric,
                    showlegend=False,
                    marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Attention Pattern Statistics Comparison",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_interactive_attention_explorer(self, 
                                            attention_output: AttentionOutput) -> go.Figure:
        """
        Create an interactive explorer for detailed attention analysis.
        
        Args:
            attention_output: Attention analysis results
            
        Returns:
            Plotly figure with interactive controls
        """
        if attention_output.cross_attention is None:
            raise ValueError("Cross-attention data required")
        
        cross_attn = attention_output.cross_attention
        
        # Handle different tensor shapes
        if cross_attn.dim() == 4:  # [layers, heads, text, image]
            num_layers = cross_attn.shape[0]
            num_heads = cross_attn.shape[1]
            
            # Create initial view (last layer, first head)
            initial_attn = cross_attn[-1, 0].detach().cpu().numpy()
        elif cross_attn.dim() == 3:  # [heads, text, image] or [layers, text, image]
            if attention_output.head_count and attention_output.head_count > 1:
                # Assume heads dimension
                num_heads = cross_attn.shape[0]
                num_layers = 1
                initial_attn = cross_attn[0].detach().cpu().numpy()
            else:
                # Assume layers dimension
                num_layers = cross_attn.shape[0]
                num_heads = 1
                initial_attn = cross_attn[-1].detach().cpu().numpy()
        else:  # [text, image]
            num_layers = 1
            num_heads = 1
            initial_attn = cross_attn.detach().cpu().numpy()
        
        # Sum over text tokens for initial view
        initial_image_attn = np.sum(initial_attn, axis=0)
        initial_heatmap = self._create_heatmap_grid(
            initial_image_attn,
            attention_output.image_size,
            attention_output.patch_size
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add main heatmap
        fig.add_trace(go.Heatmap(
            z=initial_heatmap,
            colorscale=self.colorscale,
            showscale=True,
            name="Attention Heatmap",
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Attention: %{z:.3f}<extra></extra>"
        ))
        
        # Add controls
        controls = []
        
        # Layer selector (if multiple layers)
        if num_layers > 1:
            layer_buttons = []
            for i in range(num_layers):
                layer_buttons.append({
                    'label': f'Layer {i+1}',
                    'method': 'restyle',
                    'args': [{'visible': [True]}],  # This would need more complex logic for full implementation
                })
            
            controls.append({
                'buttons': layer_buttons,
                'direction': 'down',
                'showactive': True,
                'x': 1.02,
                'xanchor': 'left',
                'y': 1,
                'yanchor': 'top',
                'type': 'dropdown'
            })
        
        # Head selector (if multiple heads)
        if num_heads > 1:
            head_buttons = []
            for i in range(num_heads):
                head_buttons.append({
                    'label': f'Head {i+1}',
                    'method': 'restyle',
                    'args': [{'visible': [True]}],  # This would need more complex logic for full implementation
                })
            
            controls.append({
                'buttons': head_buttons,
                'direction': 'down',
                'showactive': True,
                'x': 1.02,
                'xanchor': 'left',
                'y': 0.8,
                'yanchor': 'top',
                'type': 'dropdown'
            })
        
        fig.update_layout(
            title="Interactive Attention Explorer",
            updatemenus=controls,
            height=600,
            margin=dict(r=150)
        )
        
        return fig
    
    def _create_heatmap_grid(self, 
                           attention_weights: np.ndarray,
                           image_size: Tuple[int, int],
                           patch_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert 1D patch attention weights to 2D heatmap grid.
        
        Args:
            attention_weights: 1D array of attention weights
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
        
        # Handle size mismatch
        expected_patches = grid_w * grid_h
        if len(attention_weights) != expected_patches:
            if len(attention_weights) > expected_patches:
                attention_weights = attention_weights[:expected_patches]
            else:
                padded = np.zeros(expected_patches)
                padded[:len(attention_weights)] = attention_weights
                attention_weights = padded
        
        # Reshape to grid
        heatmap = attention_weights.reshape(grid_h, grid_w)
        
        # Normalize to 0-1 range
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap




