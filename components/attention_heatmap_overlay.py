"""
Advanced attention heatmap overlay component for visual focus analysis.
Shows exactly where the model focused most on the image with high-resolution overlays.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
import base64
import io

from models.base_model import AttentionOutput
from config import config


class AttentionHeatmapOverlay:
    """Advanced component for creating high-quality attention heatmap overlays."""
    
    def __init__(self, colormap: str = "viridis"):
        """
        Initialize the attention heatmap overlay component.
        
        Args:
            colormap: Color map for attention visualization
        """
        self.colormap = colormap
        self.viz_config = config.visualization
    
    def create_focus_heatmap(self, 
                           attention_output: AttentionOutput,
                           image: Image.Image,
                           question: str,
                           opacity: float = 0.6,
                           blur_radius: float = 2.0) -> go.Figure:
        """
        Create a high-resolution attention heatmap showing where the model focused most.
        
        Args:
            attention_output: Attention analysis results
            image: Original chart image
            question: Question text
            opacity: Opacity of the attention overlay (0-1)
            blur_radius: Gaussian blur radius for smoother heatmaps
            
        Returns:
            Plotly figure with focus heatmap overlay
        """
        if attention_output.cross_attention is None:
            raise ValueError("Cross-attention data required for focus heatmap")
        
        # Process attention weights
        cross_attn = attention_output.cross_attention
        
        # Handle multi-dimensional attention (average over layers/heads)
        while cross_attn.dim() > 2:
            cross_attn = cross_attn.mean(dim=0)
        
        attn_weights = cross_attn.detach().cpu().numpy()
        
        # Sum attention across text tokens to get image focus
        image_attention = np.sum(attn_weights, axis=0)
        
        # Create high-resolution heatmap
        heatmap_overlay = self._create_high_res_heatmap(
            image_attention,
            image.size,
            attention_output.patch_size,
            blur_radius=blur_radius
        )
        
        # Create the visualization
        fig = self._create_overlay_visualization(
            image, heatmap_overlay, question, attention_output, opacity
        )
        
        return fig
    
    def create_attention_intensity_map(self,
                                     attention_output: AttentionOutput,
                                     image: Image.Image,
                                     question: str,
                                     show_top_regions: int = 5) -> go.Figure:
        """
        Create an attention intensity map highlighting the top focused regions.
        
        Args:
            attention_output: Attention analysis results
            image: Original chart image
            question: Question text
            show_top_regions: Number of top attention regions to highlight
            
        Returns:
            Plotly figure with intensity map
        """
        if attention_output.cross_attention is None:
            raise ValueError("Cross-attention data required")
        
        cross_attn = attention_output.cross_attention
        while cross_attn.dim() > 2:
            cross_attn = cross_attn.mean(dim=0)
        
        attn_weights = cross_attn.detach().cpu().numpy()
        image_attention = np.sum(attn_weights, axis=0)
        
        # Find top attention regions
        top_indices = np.argsort(image_attention)[-show_top_regions:]
        top_values = image_attention[top_indices]
        
        # Create annotated image with top regions
        annotated_image = self._create_annotated_image(
            image, 
            top_indices, 
            top_values, 
            attention_output.patch_size
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add annotated image
        img_array = np.array(annotated_image)
        fig.add_trace(go.Image(z=img_array))
        
        # Add attention statistics
        fig.add_annotation(
            text=self._create_attention_summary(
                image_attention, top_indices, top_values, question
            ),
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11),
            align="left"
        )
        
        fig.update_layout(
            title=f"Top {show_top_regions} Attention Regions: {question}",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            height=600,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False
        )
        
        return fig
    
    def create_multi_scale_attention(self,
                                   attention_output: AttentionOutput,
                                   image: Image.Image,
                                   question: str,
                                   scales: List[float] = [0.5, 1.0, 2.0]) -> go.Figure:
        """
        Create multi-scale attention visualization showing different resolution views.
        
        Args:
            attention_output: Attention analysis results
            image: Original chart image
            question: Question text
            scales: Different scales for visualization
            
        Returns:
            Plotly figure with multi-scale view
        """
        if attention_output.cross_attention is None:
            raise ValueError("Cross-attention data required")
        
        cross_attn = attention_output.cross_attention
        while cross_attn.dim() > 2:
            cross_attn = cross_attn.mean(dim=0)
        
        attn_weights = cross_attn.detach().cpu().numpy()
        image_attention = np.sum(attn_weights, axis=0)
        
        # Create subplots for different scales
        cols = len(scales)
        fig = make_subplots(
            rows=1, cols=cols,
            subplot_titles=[f"Scale {scale}x" for scale in scales],
            specs=[[{"type": "xy"} for _ in range(cols)]]
        )
        
        for i, scale in enumerate(scales):
            col = i + 1
            
            # Create heatmap at this scale
            heatmap = self._create_scaled_heatmap(
                image_attention,
                image.size,
                attention_output.patch_size,
                scale
            )
            
            # Add heatmap to subplot
            fig.add_trace(
                go.Heatmap(
                    z=heatmap,
                    colorscale=self.colormap,
                    showscale=(i == cols - 1),  # Only show colorbar for last subplot
                    hovertemplate=f"Scale {scale}x<br>Attention: %{{z:.3f}}<extra></extra>"
                ),
                row=1, col=col
            )
        
        fig.update_layout(
            title=f"Multi-Scale Attention Analysis: {question}",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_attention_evolution(self,
                                 attention_outputs: List[AttentionOutput],
                                 image: Image.Image,
                                 questions: List[str],
                                 animation_frame_duration: int = 1000) -> go.Figure:
        """
        Create an animated visualization showing attention evolution across different questions.
        
        Args:
            attention_outputs: List of attention analysis results
            image: Original chart image
            questions: List of questions
            animation_frame_duration: Duration of each frame in milliseconds
            
        Returns:
            Plotly figure with animated attention evolution
        """
        if len(attention_outputs) != len(questions):
            raise ValueError("Number of attention outputs must match number of questions")
        
        # Process all attention outputs
        frames = []
        for i, (output, question) in enumerate(zip(attention_outputs, questions)):
            if output.cross_attention is None:
                continue
            
            cross_attn = output.cross_attention
            while cross_attn.dim() > 2:
                cross_attn = cross_attn.mean(dim=0)
            
            attn_weights = cross_attn.detach().cpu().numpy()
            image_attention = np.sum(attn_weights, axis=0)
            
            # Create heatmap
            heatmap = self._create_high_res_heatmap(
                image_attention,
                image.size,
                output.patch_size,
                blur_radius=1.5
            )
            
            # Create frame
            frame = go.Frame(
                data=[go.Heatmap(
                    z=heatmap,
                    colorscale=self.colormap,
                    showscale=True,
                    hovertemplate=f"Question {i+1}<br>Attention: %{{z:.3f}}<extra></extra>"
                )],
                name=f"Q{i+1}",
                layout=dict(
                    title=f"Question {i+1}: {question[:50]}{'...' if len(question) > 50 else ''}"
                )
            )
            frames.append(frame)
        
        if not frames:
            raise ValueError("No valid attention data found")
        
        # Create initial figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title="Attention Evolution Across Questions",
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": animation_frame_duration, "redraw": True},
                                      "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Question:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f"Q{i+1}"], {"frame": {"duration": 300, "redraw": True},
                                             "mode": "immediate", "transition": {"duration": 300}}],
                        "label": f"Q{i+1}",
                        "method": "animate"
                    } for i in range(len(frames))
                ]
            }],
            height=600,
            margin=dict(l=10, r=10, t=80, b=60)
        )
        
        return fig
    
    def _create_high_res_heatmap(self,
                               attention_weights: np.ndarray,
                               image_size: Tuple[int, int],
                               patch_size: Tuple[int, int],
                               blur_radius: float = 2.0) -> np.ndarray:
        """Create a high-resolution attention heatmap with gaussian blur."""
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
        
        # Upscale to image resolution using bicubic interpolation
        heatmap_upscaled = cv2.resize(
            heatmap.astype(np.float32),
            (img_w, img_h),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Apply gaussian blur for smoother appearance
        if blur_radius > 0:
            # Convert blur radius to kernel size (must be odd)
            kernel_size = int(blur_radius * 6) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            heatmap_upscaled = cv2.GaussianBlur(
                heatmap_upscaled,
                (kernel_size, kernel_size),
                blur_radius
            )
        
        return heatmap_upscaled
    
    def _create_scaled_heatmap(self,
                             attention_weights: np.ndarray,
                             image_size: Tuple[int, int],
                             patch_size: Tuple[int, int],
                             scale: float) -> np.ndarray:
        """Create a heatmap at a specific scale."""
        # Create base heatmap
        heatmap = self._create_high_res_heatmap(
            attention_weights, image_size, patch_size, blur_radius=1.0
        )
        
        # Scale the heatmap
        if scale != 1.0:
            new_size = (int(image_size[0] * scale), int(image_size[1] * scale))
            heatmap = cv2.resize(heatmap, new_size, interpolation=cv2.INTER_CUBIC)
        
        return heatmap
    
    def _create_annotated_image(self,
                              image: Image.Image,
                              top_indices: np.ndarray,
                              top_values: np.ndarray,
                              patch_size: Tuple[int, int]) -> Image.Image:
        """Create an annotated image with top attention regions marked."""
        # Create a copy of the image
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        patch_w, patch_h = patch_size
        img_w, img_h = image.size
        grid_w = img_w // patch_w
        
        # Colors for different attention levels
        colors = ["#FF0000", "#FF4500", "#FFA500", "#FFD700", "#FFFF00"]
        
        for i, (patch_idx, attention_val) in enumerate(zip(top_indices, top_values)):
            # Convert patch index to grid coordinates
            grid_y = patch_idx // grid_w
            grid_x = patch_idx % grid_w
            
            # Convert to pixel coordinates
            x1 = grid_x * patch_w
            y1 = grid_y * patch_h
            x2 = min(x1 + patch_w, img_w)
            y2 = min(y1 + patch_h, img_h)
            
            # Choose color based on attention rank
            color = colors[min(i, len(colors) - 1)]
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Add attention value label
            label = f"{attention_val:.3f}"
            bbox = draw.textbbox((x1, y1), label, font=font)
            label_bg = [x1, y1 - (bbox[3] - bbox[1]) - 2, 
                       x1 + (bbox[2] - bbox[0]) + 4, y1]
            draw.rectangle(label_bg, fill=color)
            draw.text((x1 + 2, y1 - (bbox[3] - bbox[1]) - 1), label, 
                     fill="white", font=font)
        
        return annotated
    
    def _create_attention_summary(self,
                                image_attention: np.ndarray,
                                top_indices: np.ndarray,
                                top_values: np.ndarray,
                                question: str) -> str:
        """Create a text summary of attention statistics."""
        total_attention = np.sum(image_attention)
        max_attention = np.max(image_attention)
        mean_attention = np.mean(image_attention)
        std_attention = np.std(image_attention)
        
        # Calculate attention concentration (what % of total attention is in top regions)
        top_concentration = np.sum(top_values) / total_attention * 100
        
        summary = f"""<b>Attention Analysis</b><br>
<b>Question:</b> {question[:60]}{'...' if len(question) > 60 else ''}<br><br>
<b>Statistics:</b><br>
• Max Attention: {max_attention:.3f}<br>
• Mean Attention: {mean_attention:.3f}<br>
• Std Deviation: {std_attention:.3f}<br>
• Top {len(top_values)} regions: {top_concentration:.1f}% of total<br><br>
<b>Top Regions (marked with colored boxes):</b><br>
"""
        
        for i, (idx, val) in enumerate(zip(top_indices, top_values)):
            summary += f"• Region {idx}: {val:.3f}<br>"
        
        return summary
    
    def _create_overlay_visualization(self,
                                    image: Image.Image,
                                    heatmap: np.ndarray,
                                    question: str,
                                    attention_output: AttentionOutput,
                                    opacity: float) -> go.Figure:
        """Create the main overlay visualization."""
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Create figure with two traces: image and heatmap
        fig = go.Figure()
        
        # Add original image as background
        fig.add_trace(go.Image(
            z=img_array,
            name="Original Image",
            hoverinfo="skip"
        ))
        
        # Add attention heatmap overlay
        fig.add_trace(go.Heatmap(
            z=heatmap,
            colorscale=self.colormap,
            opacity=opacity,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Attention<br>Intensity",
                    side="right"
                ),
                x=1.02,
                len=0.7,
                thickness=15
            ),
            hovertemplate="<b>Attention Focus</b><br>" +
                         "X: %{x}<br>Y: %{y}<br>" +
                         "Intensity: %{z:.3f}<extra></extra>",
            name="Attention Overlay"
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Model Focus Analysis: {question}",
                x=0.5,
                xanchor="center",
                font=dict(size=16)
            ),
            xaxis=dict(
                showticklabels=False, 
                showgrid=False, 
                zeroline=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                showticklabels=False, 
                showgrid=False, 
                zeroline=False,
                autorange="reversed"  # Flip y-axis to match image coordinates
            ),
            height=600,
            margin=dict(l=10, r=100, t=60, b=10),
            showlegend=False,
            plot_bgcolor="white"
        )
        
        # Add prediction info if available
        if attention_output.predicted_answer:
            fig.add_annotation(
                text=f"<b>Model Prediction:</b> {attention_output.predicted_answer}<br>" +
                     f"<b>Confidence:</b> {attention_output.confidence_score:.2%}",
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                xanchor="left", yanchor="bottom",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=12)
            )
        
        return fig




