"""
Advanced Analysis Component - Comprehensive attention and model analysis.
Provides sophisticated analysis capabilities for chart QA models.
"""
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import base64
import io

from models.base_model import AttentionOutput, ModelPrediction

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    analysis_type: str
    title: str
    description: str
    visualization: go.Figure
    metrics: Dict[str, Any]
    insights: List[str]
    raw_data: Optional[Dict[str, Any]] = None


class BaseAnalyzer(ABC):
    """Base class for analysis components."""
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    def analyze(self, attention_output: AttentionOutput, image: Optional[Image.Image] = None, 
                question: Optional[str] = None, **kwargs) -> AnalysisResult:
        """Perform analysis and return results."""
        pass
    
    def _validate_attention_data(self, attention_output: AttentionOutput) -> bool:
        """Validate that attention data is available and valid."""
        if not attention_output:
            return False
        
        # Check for at least one type of attention
        has_cross = attention_output.cross_attention is not None
        has_text_self = attention_output.text_self_attention is not None
        has_image_self = attention_output.image_self_attention is not None
        
        return has_cross or has_text_self or has_image_self
    
    def _prepare_attention_tensor(self, attention_tensor: torch.Tensor) -> np.ndarray:
        """Convert attention tensor to numpy array for analysis."""
        if attention_tensor is None:
            return None
        
        # Handle different tensor dimensions
        if hasattr(attention_tensor, 'detach'):
            # PyTorch tensor
            tensor = attention_tensor.detach().cpu()
        else:
            # Already numpy or list
            tensor = torch.tensor(attention_tensor) if not isinstance(attention_tensor, torch.Tensor) else attention_tensor
        
        # Convert to numpy
        if hasattr(tensor, 'numpy'):
            return tensor.numpy()
        else:
            return np.array(tensor)


class StatisticalAnalyzer(BaseAnalyzer):
    """Statistical analysis of attention patterns."""
    
    def analyze(self, attention_output: AttentionOutput, image: Optional[Image.Image] = None, 
                question: Optional[str] = None, **kwargs) -> AnalysisResult:
        """Perform comprehensive statistical analysis of attention patterns."""
        
        if not self._validate_attention_data(attention_output):
            return self._create_no_data_result()
        
        metrics = {}
        insights = []
        
        # Analyze cross-attention if available
        if attention_output.cross_attention is not None:
            cross_attn = self._prepare_attention_tensor(attention_output.cross_attention)
            
            # Flatten for statistical analysis
            while len(cross_attn.shape) > 2:
                cross_attn = cross_attn.mean(axis=0)
            
            flat_attn = cross_attn.flatten()
            
            # Calculate statistical metrics
            metrics.update({
                'cross_attention_mean': float(np.mean(flat_attn)),
                'cross_attention_std': float(np.std(flat_attn)),
                'cross_attention_max': float(np.max(flat_attn)),
                'cross_attention_min': float(np.min(flat_attn)),
                'cross_attention_entropy': self._calculate_entropy(flat_attn),
                'cross_attention_gini': self._calculate_gini_coefficient(flat_attn),
                'cross_attention_sparsity': self._calculate_sparsity(flat_attn)
            })
            
            # Generate insights
            if metrics['cross_attention_entropy'] > 5.0:
                insights.append("High attention entropy indicates distributed focus across many regions")
            elif metrics['cross_attention_entropy'] < 2.0:
                insights.append("Low attention entropy indicates concentrated focus on specific regions")
            
            if metrics['cross_attention_gini'] > 0.7:
                insights.append("High Gini coefficient suggests very uneven attention distribution")
            elif metrics['cross_attention_gini'] < 0.3:
                insights.append("Low Gini coefficient indicates relatively uniform attention distribution")
        
        # Create visualization
        fig = self._create_statistical_visualization(metrics, attention_output)
        
        return AnalysisResult(
            analysis_type="statistical",
            title="Statistical Analysis of Attention Patterns",
            description="Comprehensive statistical metrics including entropy, distribution, and concentration measures",
            visualization=fig,
            metrics=metrics,
            insights=insights,
            raw_data={'attention_output': attention_output}
        )
    
    def _calculate_entropy(self, weights: np.ndarray) -> float:
        """Calculate Shannon entropy of attention weights."""
        # Normalize weights to probabilities
        weights = weights / np.sum(weights)
        # Add small epsilon to avoid log(0)
        weights = weights + 1e-12
        entropy = -np.sum(weights * np.log(weights))
        return float(entropy)
    
    def _calculate_gini_coefficient(self, weights: np.ndarray) -> float:
        """Calculate Gini coefficient to measure attention concentration."""
        sorted_weights = np.sort(weights)
        n = len(sorted_weights)
        cumsum = np.cumsum(sorted_weights)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_weights)) / (n * cumsum[-1]) - (n + 1) / n
        return float(gini)
    
    def _calculate_sparsity(self, weights: np.ndarray, threshold: float = 0.01) -> float:
        """Calculate sparsity ratio (percentage of weights below threshold)."""
        max_weight = np.max(weights)
        threshold_value = max_weight * threshold
        sparse_count = np.sum(weights < threshold_value)
        sparsity = sparse_count / len(weights)
        return float(sparsity)
    
    def _create_statistical_visualization(self, metrics: Dict[str, Any], 
                                        attention_output: AttentionOutput) -> go.Figure:
        """Create comprehensive statistical visualization."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Attention Distribution', 'Statistical Metrics', 
                          'Concentration Analysis', 'Entropy Analysis'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        if attention_output.cross_attention is not None:
            cross_attn = self._prepare_attention_tensor(attention_output.cross_attention)
            while len(cross_attn.shape) > 2:
                cross_attn = cross_attn.mean(axis=0)
            flat_attn = cross_attn.flatten()
            
            # 1. Attention distribution histogram
            fig.add_trace(
                go.Histogram(x=flat_attn, nbinsx=50, name="Attention Distribution"),
                row=1, col=1
            )
            
            # 2. Statistical metrics bar chart
            metric_names = ['Mean', 'Std', 'Max', 'Entropy', 'Gini', 'Sparsity']
            metric_values = [
                metrics.get('cross_attention_mean', 0),
                metrics.get('cross_attention_std', 0),
                metrics.get('cross_attention_max', 0),
                metrics.get('cross_attention_entropy', 0) / 10,  # Scale for visualization
                metrics.get('cross_attention_gini', 0),
                metrics.get('cross_attention_sparsity', 0)
            ]
            
            fig.add_trace(
                go.Bar(x=metric_names, y=metric_values, name="Statistical Metrics"),
                row=1, col=2
            )
            
            # 3. Concentration analysis (sorted attention weights)
            sorted_weights = np.sort(flat_attn)[::-1]  # Descending order
            cumulative_sum = np.cumsum(sorted_weights) / np.sum(sorted_weights)
            
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(cumulative_sum)), 
                    y=cumulative_sum,
                    mode='lines',
                    name="Cumulative Attention"
                ),
                row=2, col=1
            )
            
            # 4. Entropy analysis across different regions
            # Divide attention into regions and calculate entropy for each
            n_regions = min(10, len(flat_attn) // 10)
            if n_regions > 1:
                region_size = len(flat_attn) // n_regions
                region_entropies = []
                
                for i in range(n_regions):
                    start_idx = i * region_size
                    end_idx = start_idx + region_size
                    region_weights = flat_attn[start_idx:end_idx]
                    if len(region_weights) > 0:
                        region_entropy = self._calculate_entropy(region_weights)
                        region_entropies.append(region_entropy)
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(region_entropies))),
                        y=region_entropies,
                        mode='lines+markers',
                        name="Regional Entropy"
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text="Statistical Analysis of Attention Patterns",
            showlegend=True
        )
        
        return fig
    
    def _create_no_data_result(self) -> AnalysisResult:
        """Create result when no attention data is available."""
        fig = go.Figure()
        fig.update_layout(
            title="No Attention Data Available",
            annotations=[{
                'x': 0.5, 'y': 0.5,
                'text': "No attention data available for statistical analysis",
                'showarrow': False, 'xref': 'paper', 'yref': 'paper',
                'font': {'size': 16, 'color': '#666'},
                'xanchor': 'center', 'yanchor': 'middle'
            }],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return AnalysisResult(
            analysis_type="statistical",
            title="Statistical Analysis",
            description="No attention data available",
            visualization=fig,
            metrics={},
            insights=["No attention data available for analysis"]
        )


class AttentionFlowAnalyzer(BaseAnalyzer):
    """Analyze attention flow patterns across tokens and regions."""
    
    def analyze(self, attention_output: AttentionOutput, image: Optional[Image.Image] = None, 
                question: Optional[str] = None, **kwargs) -> AnalysisResult:
        """Analyze attention flow patterns."""
        
        if not self._validate_attention_data(attention_output):
            return self._create_no_data_result()
        
        metrics = {}
        insights = []
        
        # Analyze attention flow
        if attention_output.cross_attention is not None:
            cross_attn = self._prepare_attention_tensor(attention_output.cross_attention)
            
            # Calculate flow metrics
            flow_metrics = self._calculate_flow_metrics(cross_attn, attention_output.text_tokens)
            metrics.update(flow_metrics)
            
            # Generate insights based on flow patterns
            insights = self._generate_flow_insights(flow_metrics, attention_output.text_tokens)
        
        # Create visualization
        fig = self._create_flow_visualization(attention_output, metrics)
        
        return AnalysisResult(
            analysis_type="attention_flow",
            title="Attention Flow Analysis",
            description="Analysis of how attention flows between text tokens and image regions",
            visualization=fig,
            metrics=metrics,
            insights=insights,
            raw_data={'attention_output': attention_output}
        )
    
    def _calculate_flow_metrics(self, cross_attn: np.ndarray, text_tokens: List[str]) -> Dict[str, Any]:
        """Calculate attention flow metrics."""
        metrics = {}
        
        # Ensure 2D attention matrix
        while len(cross_attn.shape) > 2:
            cross_attn = cross_attn.mean(axis=0)
        
        # Token-level flow analysis
        if text_tokens:
            token_flows = []
            for i, token in enumerate(text_tokens):
                if i < cross_attn.shape[0]:
                    token_attention = cross_attn[i, :]
                    flow_strength = np.sum(token_attention)
                    flow_concentration = np.max(token_attention) / (np.mean(token_attention) + 1e-8)
                    token_flows.append({
                        'token': token,
                        'flow_strength': float(flow_strength),
                        'flow_concentration': float(flow_concentration)
                    })
            
            metrics['token_flows'] = token_flows
            
            # Overall flow metrics
            total_flow = np.sum(cross_attn)
            max_flow = np.max(cross_attn)
            avg_flow = np.mean(cross_attn)
            
            metrics.update({
                'total_attention_flow': float(total_flow),
                'max_attention_flow': float(max_flow),
                'average_attention_flow': float(avg_flow),
                'flow_variance': float(np.var(cross_attn)),
                'dominant_token_index': int(np.argmax(np.sum(cross_attn, axis=1))),
                'dominant_region_index': int(np.argmax(np.sum(cross_attn, axis=0)))
            })
        
        return metrics
    
    def _generate_flow_insights(self, metrics: Dict[str, Any], text_tokens: List[str]) -> List[str]:
        """Generate insights based on flow analysis."""
        insights = []
        
        if 'token_flows' in metrics and text_tokens:
            # Find tokens with highest flow
            token_flows = metrics['token_flows']
            if token_flows:
                max_flow_token = max(token_flows, key=lambda x: x['flow_strength'])
                insights.append(f"Token '{max_flow_token['token']}' has the strongest attention flow")
                
                # Find highly concentrated tokens
                concentrated_tokens = [tf for tf in token_flows if tf['flow_concentration'] > 2.0]
                if concentrated_tokens:
                    insights.append(f"{len(concentrated_tokens)} tokens show highly concentrated attention patterns")
        
        # Flow distribution insights
        if 'flow_variance' in metrics:
            if metrics['flow_variance'] > 0.01:
                insights.append("High attention flow variance indicates diverse focusing patterns")
            else:
                insights.append("Low attention flow variance suggests consistent focusing behavior")
        
        return insights
    
    def _create_flow_visualization(self, attention_output: AttentionOutput, 
                                 metrics: Dict[str, Any]) -> go.Figure:
        """Create attention flow visualization."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Token Flow Strength', 'Flow Concentration', 
                          'Attention Flow Matrix', 'Flow Direction'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        if 'token_flows' in metrics and attention_output.cross_attention is not None:
            token_flows = metrics['token_flows']
            tokens = [tf['token'] for tf in token_flows]
            flow_strengths = [tf['flow_strength'] for tf in token_flows]
            flow_concentrations = [tf['flow_concentration'] for tf in token_flows]
            
            # 1. Token flow strength
            fig.add_trace(
                go.Bar(x=tokens, y=flow_strengths, name="Flow Strength"),
                row=1, col=1
            )
            
            # 2. Flow concentration
            fig.add_trace(
                go.Bar(x=tokens, y=flow_concentrations, name="Flow Concentration"),
                row=1, col=2
            )
            
            # 3. Attention flow matrix
            cross_attn = self._prepare_attention_tensor(attention_output.cross_attention)
            while len(cross_attn.shape) > 2:
                cross_attn = cross_attn.mean(axis=0)
            
            fig.add_trace(
                go.Heatmap(
                    z=cross_attn,
                    x=[f"Region {i}" for i in range(cross_attn.shape[1])],
                    y=tokens,
                    colorscale='Viridis',
                    name="Attention Matrix"
                ),
                row=2, col=1
            )
            
            # 4. Flow direction analysis
            # Calculate primary flow direction for each token
            flow_directions = []
            for i in range(min(len(tokens), cross_attn.shape[0])):
                token_attn = cross_attn[i, :]
                peak_region = np.argmax(token_attn)
                flow_directions.append(peak_region)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(flow_directions))),
                    y=flow_directions,
                    mode='lines+markers',
                    name="Primary Focus Region",
                    text=tokens
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=700,
            title_text="Attention Flow Analysis",
            showlegend=True
        )
        
        return fig
    
    def _create_no_data_result(self) -> AnalysisResult:
        """Create result when no attention data is available."""
        fig = go.Figure()
        fig.update_layout(
            title="No Attention Data Available",
            annotations=[{
                'x': 0.5, 'y': 0.5,
                'text': "No attention data available for flow analysis",
                'showarrow': False, 'xref': 'paper', 'yref': 'paper',
                'font': {'size': 16, 'color': '#666'},
                'xanchor': 'center', 'yanchor': 'middle'
            }],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return AnalysisResult(
            analysis_type="attention_flow",
            title="Attention Flow Analysis",
            description="No attention data available",
            visualization=fig,
            metrics={},
            insights=["No attention data available for analysis"]
        )


class LayerComparisonAnalyzer(BaseAnalyzer):
    """Compare attention patterns across different model layers."""
    
    def analyze(self, attention_output: AttentionOutput, image: Optional[Image.Image] = None, 
                question: Optional[str] = None, **kwargs) -> AnalysisResult:
        """Analyze attention patterns across layers using REAL multi-layer data."""
        
        if not self._validate_attention_data(attention_output):
            return self._create_no_data_result()
        
        metrics = {}
        insights = []
        
        # Extract REAL layer-wise metrics from attention data
        metrics = self._extract_real_layer_metrics(attention_output)
        
        if not metrics or metrics.get('total_layers', 0) == 0:
            # If no multi-layer data available, return informative message
            return self._create_no_layer_data_result()
        
        insights = self._generate_layer_insights(metrics)
        
        # Create visualization
        fig = self._create_layer_comparison_visualization(metrics)
        
        return AnalysisResult(
            analysis_type="layer_comparison",
            title="Layer-wise Attention Comparison (Real Data)",
            description="Comparison of attention patterns across different model layers using actual extracted attention weights",
            visualization=fig,
            metrics=metrics,
            insights=insights,
            raw_data={'attention_output': attention_output}
        )
    
    def _extract_real_layer_metrics(self, attention_output: AttentionOutput) -> Dict[str, Any]:
        """Extract REAL layer-wise metrics from multi-layer attention data."""
        metrics = {}
        
        if attention_output.cross_attention is None:
            return metrics
        
        cross_attn = self._prepare_attention_tensor(attention_output.cross_attention)
        
        # Check if we have multi-layer attention data
        # Expected shapes: [layers, heads, text_tokens, image_patches] or [layers, text_tokens, image_patches]
        if len(cross_attn.shape) < 3:
            logger.warning("No multi-layer attention data available (tensor is 2D or less)")
            return metrics
        
        # Determine if first dimension is layers
        num_layers = cross_attn.shape[0]
        
        # If we have very few "layers" (e.g., < 3), it might not be layer dimension
        if num_layers < 3:
            logger.warning(f"Insufficient layers detected ({num_layers}), may not be layer dimension")
            return metrics
        
        logger.info(f"Extracting layer-wise metrics from {num_layers} layers")
        
        layer_metrics = []
        
        for layer_idx in range(num_layers):
            # Extract attention for this layer
            layer_attn = cross_attn[layer_idx]
            
            # If we have heads dimension, average over heads
            while len(layer_attn.shape) > 2:
                layer_attn = layer_attn.mean(axis=0)
            
            # Calculate statistics for this layer
            flat_attn = layer_attn.flatten()
            
            # Calculate entropy
            normalized = flat_attn / (np.sum(flat_attn) + 1e-12)
            entropy = -np.sum(normalized * np.log(normalized + 1e-12))
            
            # Calculate variance
            variance = float(np.var(flat_attn))
            
            # Calculate peak attention
            peak = float(np.max(flat_attn))
            
            # Calculate average attention
            avg = float(np.mean(flat_attn))
            
            # Calculate sparsity (percentage of attention below 1% of max)
            threshold = peak * 0.01
            sparsity = float(np.sum(flat_attn < threshold) / len(flat_attn))
            
            layer_metrics.append({
                'layer': layer_idx + 1,
                'average_attention': avg,
                'attention_variance': variance,
                'peak_attention': peak,
                'attention_entropy': float(entropy),
                'sparsity': sparsity
            })
        
        metrics['layer_metrics'] = layer_metrics
        metrics['total_layers'] = num_layers
        
        # Calculate layer progression metrics
        entropies = [lm['attention_entropy'] for lm in layer_metrics]
        metrics['entropy_progression'] = 'increasing' if entropies[-1] > entropies[0] else 'decreasing'
        metrics['entropy_change'] = float(entropies[-1] - entropies[0])
        
        # Calculate average variance change across layers
        variances = [lm['attention_variance'] for lm in layer_metrics]
        metrics['variance_trend'] = 'increasing' if variances[-1] > variances[0] else 'decreasing'
        
        # Identify most and least focused layers
        avg_attentions = [lm['average_attention'] for lm in layer_metrics]
        metrics['most_active_layer'] = int(np.argmax(avg_attentions) + 1)
        metrics['least_active_layer'] = int(np.argmin(avg_attentions) + 1)
        
        return metrics
    
    def _generate_layer_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate insights based on REAL layer analysis."""
        insights = []
        
        if 'layer_metrics' in metrics:
            layer_metrics = metrics['layer_metrics']
            num_layers = len(layer_metrics)
            
            # Find layer with highest attention
            max_attention_layer = max(layer_metrics, key=lambda x: x['average_attention'])
            insights.append(f"Layer {max_attention_layer['layer']} (of {num_layers}) shows the highest average attention")
            
            # Analyze entropy progression
            if metrics.get('entropy_progression') == 'increasing':
                change = metrics.get('entropy_change', 0)
                insights.append(f"Attention becomes more distributed in deeper layers (entropy change: +{change:.3f})")
            else:
                change = abs(metrics.get('entropy_change', 0))
                insights.append(f"Attention becomes more focused in deeper layers (entropy change: -{change:.3f})")
            
            # Variance analysis
            variances = [lm['attention_variance'] for lm in layer_metrics]
            if max(variances) > 2 * min(variances):
                insights.append("Significant variation in attention patterns across layers indicates diverse processing")
            else:
                insights.append("Consistent attention patterns across layers suggests stable feature processing")
            
            # Analyze sparsity trends
            sparsities = [lm.get('sparsity', 0) for lm in layer_metrics]
            avg_sparsity = np.mean(sparsities)
            if avg_sparsity > 0.7:
                insights.append(f"High sparsity ({avg_sparsity:.1%}) - model focuses on specific regions at each layer")
            elif avg_sparsity < 0.3:
                insights.append(f"Low sparsity ({avg_sparsity:.1%}) - model considers broad image regions at each layer")
            
            # Early vs late layer comparison
            early_layers = layer_metrics[:num_layers//3]
            late_layers = layer_metrics[-num_layers//3:]
            early_avg_entropy = np.mean([lm['attention_entropy'] for lm in early_layers])
            late_avg_entropy = np.mean([lm['attention_entropy'] for lm in late_layers])
            
            if late_avg_entropy > early_avg_entropy * 1.2:
                insights.append("Late layers show much broader attention patterns than early layers")
            elif early_avg_entropy > late_avg_entropy * 1.2:
                insights.append("Early layers show broader attention, late layers are more focused")
        
        return insights
    
    def _create_no_layer_data_result(self) -> AnalysisResult:
        """Create result when no multi-layer attention data is available."""
        fig = go.Figure()
        fig.update_layout(
            title="Multi-Layer Attention Data Not Available",
            annotations=[{
                'x': 0.5, 'y': 0.5,
                'text': "This model did not provide multi-layer attention data.<br>" +
                        "The attention tensor is 2D (single layer/averaged).<br><br>" +
                        "Try other analysis types like 'Statistical Analysis' or 'Attention Flow'.",
                'showarrow': False, 'xref': 'paper', 'yref': 'paper',
                'font': {'size': 14, 'color': '#666'},
                'xanchor': 'center', 'yanchor': 'middle'
            }],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return AnalysisResult(
            analysis_type="layer_comparison",
            title="Layer Comparison Analysis",
            description="Multi-layer attention data not available from this model",
            visualization=fig,
            metrics={},
            insights=["Model does not provide per-layer attention data",
                     "Attention tensor is already averaged across layers",
                     "This is normal for some model architectures"]
        )
    
    def _create_layer_comparison_visualization(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create layer comparison visualization."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Attention by Layer', 'Attention Variance by Layer',
                          'Peak Attention Progression', 'Entropy Progression'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        if 'layer_metrics' in metrics:
            layer_metrics = metrics['layer_metrics']
            layers = [lm['layer'] for lm in layer_metrics]
            avg_attentions = [lm['average_attention'] for lm in layer_metrics]
            variances = [lm['attention_variance'] for lm in layer_metrics]
            peak_attentions = [lm['peak_attention'] for lm in layer_metrics]
            entropies = [lm['attention_entropy'] for lm in layer_metrics]
            
            # 1. Average attention by layer
            fig.add_trace(
                go.Bar(x=layers, y=avg_attentions, name="Average Attention"),
                row=1, col=1
            )
            
            # 2. Attention variance by layer
            fig.add_trace(
                go.Bar(x=layers, y=variances, name="Attention Variance"),
                row=1, col=2
            )
            
            # 3. Peak attention progression
            fig.add_trace(
                go.Scatter(
                    x=layers, y=peak_attentions,
                    mode='lines+markers',
                    name="Peak Attention"
                ),
                row=2, col=1
            )
            
            # 4. Entropy progression
            fig.add_trace(
                go.Scatter(
                    x=layers, y=entropies,
                    mode='lines+markers',
                    name="Attention Entropy"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=600,
            title_text="Layer-wise Attention Analysis",
            showlegend=True
        )
        
        return fig
    
    def _create_no_data_result(self) -> AnalysisResult:
        """Create result when no attention data is available."""
        fig = go.Figure()
        fig.update_layout(
            title="No Attention Data Available",
            annotations=[{
                'x': 0.5, 'y': 0.5,
                'text': "No attention data available for layer comparison",
                'showarrow': False, 'xref': 'paper', 'yref': 'paper',
                'font': {'size': 16, 'color': '#666'},
                'xanchor': 'center', 'yanchor': 'middle'
            }],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return AnalysisResult(
            analysis_type="layer_comparison",
            title="Layer Comparison Analysis",
            description="No attention data available",
            visualization=fig,
            metrics={},
            insights=["No attention data available for analysis"]
        )


class MultiHeadAnalyzer(BaseAnalyzer):
    """Analyze attention patterns across different attention heads."""
    
    def analyze(self, attention_output: AttentionOutput, image: Optional[Image.Image] = None, 
                question: Optional[str] = None, **kwargs) -> AnalysisResult:
        """Analyze multi-head attention patterns using REAL data."""
        
        if not self._validate_attention_data(attention_output):
            return self._create_no_data_result()
        
        metrics = {}
        insights = []
        
        # Analyze multi-head patterns using REAL data
        if attention_output.cross_attention is not None:
            cross_attn = self._prepare_attention_tensor(attention_output.cross_attention)
            head_metrics = self._analyze_attention_heads(cross_attn, attention_output.head_count or 8)
            
            if not head_metrics or head_metrics.get('num_heads', 0) == 0:
                # If no multi-head data available, return informative message
                return self._create_no_head_data_result()
            
            metrics.update(head_metrics)
            insights = self._generate_head_insights(head_metrics)
        
        # Create visualization
        fig = self._create_multi_head_visualization(metrics)
        
        return AnalysisResult(
            analysis_type="multi_head",
            title="Multi-Head Attention Analysis (Real Data)",
            description="Analysis of attention patterns across different attention heads using actual extracted weights",
            visualization=fig,
            metrics=metrics,
            insights=insights,
            raw_data={'attention_output': attention_output}
        )
    
    def _analyze_attention_heads(self, cross_attn: np.ndarray, num_heads: int) -> Dict[str, Any]:
        """Analyze REAL attention patterns across heads."""
        metrics = {}
        
        # Check if we have multi-head attention data
        original_shape = cross_attn.shape
        
        # If we don't have a heads dimension, we can't do per-head analysis
        if len(original_shape) < 3:
            logger.warning("No multi-head attention data available (tensor is 2D or less)")
            return metrics
        
        # Determine which dimension is heads
        # Common patterns: [layers, heads, text, image] or [heads, text, image]
        if len(original_shape) == 4:
            # [layers, heads, text, image] - average over layers first
            cross_attn = cross_attn.mean(axis=0)  # Now [heads, text, image]
        
        if len(original_shape) < 3 or cross_attn.shape[0] > 32:
            # If first dim is > 32, it's probably not heads (more likely text tokens)
            logger.warning(f"Cannot identify heads dimension (shape: {original_shape})")
            return metrics
        
        actual_num_heads = cross_attn.shape[0]
        logger.info(f"Analyzing {actual_num_heads} attention heads")
        
        head_analyses = []
        
        for head_idx in range(actual_num_heads):
            # Extract attention for this head
            head_attn = cross_attn[head_idx]  # Shape: [text, image]
            
            # Flatten for analysis
            flat_attn = head_attn.flatten()
            
            # Calculate real statistics for this head
            avg_attention = float(np.mean(flat_attn))
            
            # Calculate entropy
            normalized = flat_attn / (np.sum(flat_attn) + 1e-12)
            entropy = -np.sum(normalized * np.log(normalized + 1e-12))
            
            # Calculate concentration (ratio of max to mean)
            max_attn = float(np.max(flat_attn))
            concentration = max_attn / (avg_attention + 1e-8)
            
            # Calculate sparsity
            threshold = max_attn * 0.1
            sparsity = float(np.sum(flat_attn < threshold) / len(flat_attn))
            
            # Infer specialization based on attention patterns
            specialization = self._infer_head_specialization(head_attn, concentration, entropy)
            
            head_analyses.append({
                'head': head_idx + 1,
                'average_attention': avg_attention,
                'attention_entropy': float(entropy),
                'concentration_score': concentration,
                'sparsity': sparsity,
                'specialization_type': specialization
            })
        
        metrics['head_analyses'] = head_analyses
        metrics['num_heads'] = actual_num_heads
        
        # Calculate head diversity metrics
        entropies = [ha['attention_entropy'] for ha in head_analyses]
        concentrations = [ha['concentration_score'] for ha in head_analyses]
        sparsities = [ha['sparsity'] for ha in head_analyses]
        
        metrics['entropy_diversity'] = float(np.std(entropies))
        metrics['concentration_diversity'] = float(np.std(concentrations))
        metrics['avg_sparsity'] = float(np.mean(sparsities))
        
        # Identify specialized vs generalist heads
        high_concentration_heads = [ha for ha in head_analyses if ha['concentration_score'] > np.mean(concentrations) * 1.5]
        metrics['num_specialized_heads'] = len(high_concentration_heads)
        
        return metrics
    
    def _infer_head_specialization(self, head_attn: np.ndarray, concentration: float, entropy: float) -> str:
        """Infer head specialization based on REAL attention patterns."""
        # Use statistical properties to infer specialization
        if concentration > 10:
            return "Highly Focused"
        elif concentration < 2:
            return "Broad Context"
        elif entropy > 5:
            return "Distributed"
        elif entropy < 2:
            return "Concentrated"
        else:
            return "Balanced"
    
    def _classify_head_specialization(self, head_idx: int, num_heads: int) -> str:
        """Classify the specialization type of an attention head."""
        specializations = [
            'Global Context', 'Local Details', 'Text Focus', 'Image Focus',
            'Structural', 'Semantic', 'Positional', 'Relational'
        ]
        return specializations[head_idx % len(specializations)]
    
    def _generate_head_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate insights based on REAL head analysis."""
        insights = []
        
        if 'head_analyses' in metrics:
            head_analyses = metrics['head_analyses']
            num_heads = len(head_analyses)
            
            # Find most and least concentrated heads
            most_concentrated = max(head_analyses, key=lambda x: x['concentration_score'])
            least_concentrated = min(head_analyses, key=lambda x: x['concentration_score'])
            
            insights.append(f"Head {most_concentrated['head']} (of {num_heads}) shows highest concentration ({most_concentrated['specialization_type']}, ratio: {most_concentrated['concentration_score']:.2f})")
            insights.append(f"Head {least_concentrated['head']} shows most distributed attention ({least_concentrated['specialization_type']}, ratio: {least_concentrated['concentration_score']:.2f})")
            
            # Analyze diversity
            entropy_diversity = metrics.get('entropy_diversity', 0)
            if entropy_diversity > 0.5:
                insights.append(f"High diversity in attention patterns across heads (std: {entropy_diversity:.3f})")
            else:
                insights.append(f"Similar attention patterns across most heads (std: {entropy_diversity:.3f})")
            
            # Specialization insights
            specializations = [ha['specialization_type'] for ha in head_analyses]
            unique_specializations = len(set(specializations))
            insights.append(f"Model uses {unique_specializations} different attention pattern types")
            
            # Specialized heads
            num_specialized = metrics.get('num_specialized_heads', 0)
            if num_specialized > 0:
                insights.append(f"{num_specialized} heads show specialized attention (>1.5x mean concentration)")
            
            # Sparsity insights
            avg_sparsity = metrics.get('avg_sparsity', 0)
            if avg_sparsity > 0.7:
                insights.append(f"High average sparsity ({avg_sparsity:.1%}) - heads focus on specific regions")
            elif avg_sparsity < 0.3:
                insights.append(f"Low average sparsity ({avg_sparsity:.1%}) - heads consider broad regions")
        
        return insights
    
    def _create_multi_head_visualization(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create multi-head attention visualization."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Head Attention Strength', 'Head Concentration Scores',
                          'Head Entropy Distribution', 'Head Specialization'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        if 'head_analyses' in metrics:
            head_analyses = metrics['head_analyses']
            heads = [ha['head'] for ha in head_analyses]
            attentions = [ha['average_attention'] for ha in head_analyses]
            concentrations = [ha['concentration_score'] for ha in head_analyses]
            entropies = [ha['attention_entropy'] for ha in head_analyses]
            specializations = [ha['specialization_type'] for ha in head_analyses]
            
            # 1. Head attention strength
            fig.add_trace(
                go.Bar(x=heads, y=attentions, name="Attention Strength"),
                row=1, col=1
            )
            
            # 2. Head concentration scores
            fig.add_trace(
                go.Bar(x=heads, y=concentrations, name="Concentration Score"),
                row=1, col=2
            )
            
            # 3. Head entropy distribution
            fig.add_trace(
                go.Histogram(x=entropies, nbinsx=10, name="Entropy Distribution"),
                row=2, col=1
            )
            
            # 4. Head specialization scatter
            fig.add_trace(
                go.Scatter(
                    x=concentrations,
                    y=entropies,
                    mode='markers+text',
                    text=[f"H{h}" for h in heads],
                    textposition="top center",
                    name="Head Specialization",
                    hovertext=specializations
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=600,
            title_text="Multi-Head Attention Analysis",
            showlegend=True
        )
        
        return fig
    
    def _create_no_data_result(self) -> AnalysisResult:
        """Create result when no attention data is available."""
        fig = go.Figure()
        fig.update_layout(
            title="No Attention Data Available",
            annotations=[{
                'x': 0.5, 'y': 0.5,
                'text': "No attention data available for multi-head analysis",
                'showarrow': False, 'xref': 'paper', 'yref': 'paper',
                'font': {'size': 16, 'color': '#666'},
                'xanchor': 'center', 'yanchor': 'middle'
            }],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return AnalysisResult(
            analysis_type="multi_head",
            title="Multi-Head Analysis",
            description="No attention data available",
            visualization=fig,
            metrics={},
            insights=["No attention data available for analysis"]
        )
    
    def _create_no_head_data_result(self) -> AnalysisResult:
        """Create result when no multi-head attention data is available."""
        fig = go.Figure()
        fig.update_layout(
            title="Multi-Head Attention Data Not Available",
            annotations=[{
                'x': 0.5, 'y': 0.5,
                'text': "This model did not provide per-head attention data.<br>" +
                        "The attention tensor doesn't have a heads dimension.<br><br>" +
                        "Try other analysis types like 'Statistical Analysis' or 'Attention Flow'.",
                'showarrow': False, 'xref': 'paper', 'yref': 'paper',
                'font': {'size': 14, 'color': '#666'},
                'xanchor': 'center', 'yanchor': 'middle'
            }],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return AnalysisResult(
            analysis_type="multi_head",
            title="Multi-Head Analysis",
            description="Multi-head attention data not available from this model",
            visualization=fig,
            metrics={},
            insights=["Model does not provide per-head attention data",
                     "Attention tensor is already averaged across heads",
                     "This is normal for some model architectures"]
        )


class CrossAttentionExplorer(BaseAnalyzer):
    """Interactive cross-attention explorer for token-to-image region visualization."""
    
    def analyze(self, attention_output: AttentionOutput, image: Optional[Image.Image] = None, 
                question: Optional[str] = None, **kwargs) -> AnalysisResult:
        """Create interactive cross-attention explorer visualization."""
        
        if not self._validate_attention_data(attention_output):
            return self._create_no_data_result()
        
        if attention_output.cross_attention is None:
            return self._create_no_data_result()
        
        metrics = {}
        insights = []
        
        # Prepare cross-attention data
        cross_attn = self._prepare_attention_tensor(attention_output.cross_attention)
        
        # Ensure 2D: [text_tokens, image_patches]
        while len(cross_attn.shape) > 2:
            cross_attn = cross_attn.mean(axis=0)
        
        # Validate text tokens
        if not attention_output.text_tokens or len(attention_output.text_tokens) == 0:
            return self._create_no_data_result()
        
        # Calculate token-level attention statistics
        token_stats = self._calculate_token_statistics(cross_attn, attention_output.text_tokens)
        metrics['token_statistics'] = token_stats
        
        # Generate insights
        insights = self._generate_explorer_insights(token_stats, attention_output.text_tokens)
        
        # Create interactive visualization
        fig = self._create_interactive_explorer(
            cross_attn, 
            attention_output.text_tokens,
            attention_output.image_patch_coords,
            image,
            attention_output.image_size
        )
        
        return AnalysisResult(
            analysis_type="cross_attention_explorer",
            title="Cross-Attention Explorer – Interactive Token-to-Image Attention",
            description="Select any token from your question to see which image regions it attends to. The left panel shows the actual chart image with attention heatmap overlay, while the right bar chart shows patch-level attention strength for each image region.",
            visualization=fig,
            metrics=metrics,
            insights=insights,
            raw_data={
                'attention_output': attention_output,
                'cross_attention_shape': cross_attn.shape if hasattr(cross_attn, 'shape') else None,
                'num_tokens': len(attention_output.text_tokens) if attention_output.text_tokens else 0
            }
        )
    
    def _calculate_token_statistics(self, cross_attn: np.ndarray, text_tokens: List[str]) -> List[Dict[str, Any]]:
        """Calculate statistics for each token's attention distribution."""
        token_stats = []
        
        for i, token in enumerate(text_tokens):
            if i < cross_attn.shape[0]:
                token_attn = cross_attn[i, :]
                
                # Calculate statistics
                stats = {
                    'token': token,
                    'token_index': i,
                    'total_attention': float(np.sum(token_attn)),
                    'max_attention': float(np.max(token_attn)),
                    'mean_attention': float(np.mean(token_attn)),
                    'std_attention': float(np.std(token_attn)),
                    'entropy': self._calculate_entropy(token_attn),
                    'top_region_index': int(np.argmax(token_attn)),
                    'concentration_ratio': float(np.max(token_attn) / (np.mean(token_attn) + 1e-8)),
                    'attention_distribution': token_attn.tolist()
                }
                
                token_stats.append(stats)
        
        return token_stats
    
    def _calculate_entropy(self, weights: np.ndarray) -> float:
        """Calculate Shannon entropy of attention weights."""
        weights = weights / (np.sum(weights) + 1e-12)
        weights = weights + 1e-12
        entropy = -np.sum(weights * np.log(weights))
        return float(entropy)
    
    def _generate_explorer_insights(self, token_stats: List[Dict[str, Any]], text_tokens: List[str]) -> List[str]:
        """Generate insights from token attention patterns."""
        insights = []
        
        if not token_stats:
            return ["No token statistics available"]
        
        # Find most focused token
        most_focused = max(token_stats, key=lambda x: x['concentration_ratio'])
        insights.append(f"Token '{most_focused['token']}' shows highest attention concentration (ratio: {most_focused['concentration_ratio']:.2f})")
        
        # Find most distributed token
        most_distributed = max(token_stats, key=lambda x: x['entropy'])
        insights.append(f"Token '{most_distributed['token']}' has most distributed attention (entropy: {most_distributed['entropy']:.2f})")
        
        # Find token with strongest overall attention
        strongest_token = max(token_stats, key=lambda x: x['total_attention'])
        insights.append(f"Token '{strongest_token['token']}' has strongest overall attention to image regions")
        
        # Analyze attention patterns
        high_concentration_tokens = [ts for ts in token_stats if ts['concentration_ratio'] > 3.0]
        if high_concentration_tokens:
            insights.append(f"{len(high_concentration_tokens)} tokens show highly concentrated attention (focusing on specific regions)")
        
        # Analyze entropy distribution
        avg_entropy = np.mean([ts['entropy'] for ts in token_stats])
        if avg_entropy > 4.0:
            insights.append("Overall high entropy indicates model considers multiple image regions for most tokens")
        elif avg_entropy < 2.0:
            insights.append("Overall low entropy indicates model focuses on specific regions for most tokens")
        
        return insights
    
    def _create_interactive_explorer(self, cross_attn: np.ndarray, text_tokens: List[str],
                                    image_patch_coords: List[Tuple[int, int]], 
                                    image: Optional[Image.Image],
                                    image_size: Optional[Tuple[int, int]]) -> go.Figure:
        """Create interactive cross-attention explorer with per-token attention maps overlaid on the actual image."""
        
        # Validate inputs
        if text_tokens is None or len(text_tokens) == 0:
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text="No text tokens available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig
        
        # Calculate grid dimensions from patch coordinates
        if image_patch_coords and len(image_patch_coords) > 0:
            # Infer grid size from patch coordinates
            unique_x = len(set(coord[0] for coord in image_patch_coords))
            unique_y = len(set(coord[1] for coord in image_patch_coords))
            grid_h = unique_y if unique_y > 0 else int(np.sqrt(len(image_patch_coords)))
            grid_w = unique_x if unique_x > 0 else int(np.sqrt(len(image_patch_coords)))
        else:
            # Fallback: assume square grid
            grid_size = int(np.sqrt(cross_attn.shape[1]))
            grid_h = grid_w = grid_size
        
        # Prepare image background if available
        img_width, img_height = image_size if image_size else (800, 600)
        
        # Create the figure with one large attention heatmap per token
        # We'll create multiple heatmap traces, one for each token, and use visibility to switch between them
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Question Token → Image Regions Attention',
                'Patch-Level Attention Strength'
            ),
            specs=[[{"type": "xy"}, {"type": "bar"}]],
            column_widths=[0.65, 0.35],
            horizontal_spacing=0.12
        )
        
        # Convert image to base64 for background (if available)
        img_str = None
        if image is not None:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            img_str = base64.b64encode(img_bytes).decode()
        
        # Create a heatmap trace for EACH token
        # Only the first one will be visible initially
        for token_idx, token in enumerate(text_tokens):
            if token_idx < cross_attn.shape[0]:
                token_attn = cross_attn[token_idx, :]
                
                # Reshape to 2D grid for heatmap
                if len(token_attn) == grid_h * grid_w:
                    attention_grid = token_attn.reshape(grid_h, grid_w)
                else:
                    # Pad or truncate to fit grid
                    target_size = grid_h * grid_w
                    if len(token_attn) < target_size:
                        padded = np.pad(token_attn, (0, target_size - len(token_attn)), mode='constant')
                        attention_grid = padded.reshape(grid_h, grid_w)
                    else:
                        attention_grid = token_attn[:target_size].reshape(grid_h, grid_w)
                
                # Add heatmap for this token with image dimensions
                fig.add_trace(
                    go.Heatmap(
                        z=attention_grid,
                        colorscale='Hot',
                        name=f"Attention: {token}",
                        visible=(token_idx == 0),  # Only first token visible initially
                        hovertemplate=f'<b>Token: {token}</b><br><b>Grid Position:</b> (%{{x}}, %{{y}})<br><b>Attention:</b> %{{z:.4f}}<extra></extra>',
                        colorbar=dict(
                            title="Attention<br>Strength",
                            x=0.63,
                            len=0.8
                        ),
                        opacity=0.6,  # Make heatmap semi-transparent to see image below
                        x0=0,
                        dx=img_width/grid_w,
                        y0=0,
                        dy=img_height/grid_h
                    ),
                    row=1, col=1
                )
        
        # 2. Create bar chart showing patch-level attention for the currently selected token
        # We'll create multiple bar traces, one for each token
        for token_idx, token in enumerate(text_tokens):
            if token_idx < cross_attn.shape[0]:
                token_attn = cross_attn[token_idx, :]
                
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(token_attn))),
                        y=token_attn,
                        name=f"Patches: {token}",
                        visible=(token_idx == 0),  # Only first token visible initially
                        marker=dict(
                            color=token_attn,
                            colorscale='Viridis',
                            showscale=False
                        ),
                        hovertemplate=f'<b>Token: {token}</b><br><b>Patch Index:</b> %{{x}}<br><b>Attention:</b> %{{y:.4f}}<extra></extra>'
                    ),
                    row=1, col=2
                )
        
        # Create dropdown buttons for token selection
        # Each button will show/hide the corresponding heatmap and bar chart
        buttons = []
        first_token = text_tokens[0] if text_tokens and len(text_tokens) > 0 else "None"
        
        for token_idx, token in enumerate(text_tokens):
            # Create visibility array: [heatmap traces, bar traces]
            # We have len(text_tokens) heatmap traces + len(text_tokens) bar traces
            visibility = [False] * (len(text_tokens) * 2)
            
            # Set the current token's heatmap and bar to visible
            visibility[token_idx] = True  # Heatmap for this token
            visibility[len(text_tokens) + token_idx] = True  # Bar chart for this token
            
            buttons.append(
                dict(
                    label=f"{token_idx+1}. {token}",
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title.text": f"Cross-Attention Explorer – Token: '{token}'"}
                    ]
                )
            )
        
        # Update layout with dropdown menu and image background
        layout_update = {
            'height': 700,
            'title': {
                'text': f"Cross-Attention Explorer – Token: '{first_token}'",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            'showlegend': False,
            'hovermode': 'closest',
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'updatemenus': [
                dict(
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.5,
                    xanchor="center",
                    y=1.15,
                    yanchor="top",
                    bgcolor="#e8f4f8",
                    bordercolor="#2c3e50",
                    borderwidth=2,
                    font=dict(size=11)
                )
            ],
            'annotations': [
                dict(
                    text="<b>Select Token:</b>",
                    xref="paper", yref="paper",
                    x=0.5, y=1.19,
                    xanchor='center',
                    showarrow=False,
                    font=dict(size=13, color="#2c3e50")
                ),
                dict(
                    text="💡 Select a token from the dropdown to see which image regions it attends to",
                    xref="paper", yref="paper",
                    x=0.5, y=1.08,
                    showarrow=False,
                    font=dict(size=11, color="#7f8c8d"),
                    xanchor='center'
                )
            ]
        }
        
        # Add image as background if available
        if img_str is not None:
            layout_update['images'] = [
                dict(
                    source=f'data:image/png;base64,{img_str}',
                    xref="x",
                    yref="y",
                    x=0,
                    y=img_height,
                    sizex=img_width,
                    sizey=img_height,
                    sizing="stretch",
                    opacity=1.0,
                    layer="below",
                    xanchor="left",
                    yanchor="top"
                )
            ]
        
        fig.update_layout(**layout_update)
        
        # Update axes - first subplot shows the image with proper scaling
        fig.update_xaxes(
            title_text="Image Width (pixels)", 
            row=1, col=1, 
            showgrid=False,
            range=[0, img_width],
            constrain='domain'
        )
        fig.update_yaxes(
            title_text="Image Height (pixels)", 
            row=1, col=1, 
            showgrid=False,
            range=[0, img_height],
            scaleanchor="x",
            scaleratio=1,
            constrain='domain'
        )
        
        # Second subplot shows patch-level attention bars
        fig.update_xaxes(title_text="Patch Index", row=1, col=2)
        fig.update_yaxes(title_text="Attention Strength", row=1, col=2)
        
        return fig
    
    def _create_no_data_result(self) -> AnalysisResult:
        """Create result when no attention data is available."""
        fig = go.Figure()
        fig.update_layout(
            title="No Cross-Attention Data Available",
            annotations=[{
                'x': 0.5, 'y': 0.5,
                'text': "Cross-attention data is required for the explorer.<br>Please ensure the model supports attention extraction.",
                'showarrow': False, 'xref': 'paper', 'yref': 'paper',
                'font': {'size': 16, 'color': '#666'},
                'xanchor': 'center', 'yanchor': 'middle'
            }],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return AnalysisResult(
            analysis_type="cross_attention_explorer",
            title="Cross-Attention Explorer",
            description="No cross-attention data available",
            visualization=fig,
            metrics={},
            insights=["No cross-attention data available for exploration"]
        )


class CrossAttentionExplorerText(BaseAnalyzer):
    """Interactive cross-attention explorer for text-only tokens (excludes <image> token)."""
    
    def analyze(self, attention_output: AttentionOutput, image: Optional[Image.Image] = None, 
                question: Optional[str] = None, **kwargs) -> AnalysisResult:
        """Create interactive cross-attention explorer visualization for text tokens only."""
        
        if not self._validate_attention_data(attention_output):
            return self._create_no_data_result()
        
        if attention_output.cross_attention is None:
            return self._create_no_data_result()
        
        metrics = {}
        insights = []
        
        # Prepare cross-attention data
        cross_attn = self._prepare_attention_tensor(attention_output.cross_attention)
        
        # Ensure 2D: [text_tokens, image_patches]
        while len(cross_attn.shape) > 2:
            cross_attn = cross_attn.mean(axis=0)
        
        # Keep all tokens and attention data, but identify which ones to show in dropdown
        if not attention_output.text_tokens or len(attention_output.text_tokens) == 0:
            return self._create_no_data_result()
        
        # Identify text tokens (excluding <image> tokens) for dropdown display
        text_only_indices = []
        
        for i, token in enumerate(attention_output.text_tokens):
            # Skip <image> tokens and variations
            if not (token.startswith('<image') or token == '<image>' or 'image' in token.lower() and token.startswith('<')):
                text_only_indices.append(i)
        
        if len(text_only_indices) == 0:
            return self._create_no_text_tokens_result()
        
        # Use full attention matrix and all tokens (don't filter the data)
        # We'll pass the text_only_indices to the visualization to control dropdown
        full_cross_attn = cross_attn
        all_tokens = attention_output.text_tokens
        
        # Calculate token-level attention statistics for ALL tokens
        token_stats = self._calculate_token_statistics(full_cross_attn, all_tokens)
        metrics['token_statistics'] = token_stats
        metrics['original_token_count'] = len(all_tokens)
        metrics['text_token_count'] = len(text_only_indices)
        metrics['image_tokens_hidden'] = len(all_tokens) - len(text_only_indices)
        
        # Generate insights (only for text tokens in dropdown)
        text_token_stats = [token_stats[i] for i in text_only_indices if i < len(token_stats)]
        insights = self._generate_explorer_insights(text_token_stats, [all_tokens[i] for i in text_only_indices])
        insights.insert(0, f"Showing {len(text_only_indices)} question tokens in dropdown (hiding {metrics['image_tokens_hidden']} <image> tokens)")
        
        # Create interactive visualization with all data but filtered dropdown
        fig = self._create_interactive_explorer(
            full_cross_attn, 
            all_tokens,
            attention_output.image_patch_coords,
            image,
            attention_output.image_size,
            text_only_indices=text_only_indices  # Pass indices to control dropdown
        )
        
        return AnalysisResult(
            analysis_type="cross_attention_explorer_text",
            title="Cross-Attention Explorer (Text) – Text Token to Image Attention",
            description="Interactive exploration of how text tokens (excluding <image> tokens) attend to different image regions. This shows the cross-attention between your question text and the chart image, helping you understand which parts of the chart the model focuses on for each word in your question.",
            visualization=fig,
            metrics=metrics,
            insights=insights,
            raw_data={
                'attention_output': attention_output,
                'full_attention_shape': full_cross_attn.shape if hasattr(full_cross_attn, 'shape') else None,
                'all_tokens': all_tokens,
                'text_only_indices': text_only_indices
            }
        )
    
    def _calculate_token_statistics(self, cross_attn: np.ndarray, text_tokens: List[str]) -> List[Dict[str, Any]]:
        """Calculate statistics for each token's attention distribution."""
        token_stats = []
        
        for i, token in enumerate(text_tokens):
            if i < cross_attn.shape[0]:
                token_attn = cross_attn[i, :]
                
                # Calculate statistics
                stats = {
                    'token': token,
                    'token_index': i,
                    'total_attention': float(np.sum(token_attn)),
                    'max_attention': float(np.max(token_attn)),
                    'mean_attention': float(np.mean(token_attn)),
                    'std_attention': float(np.std(token_attn)),
                    'entropy': self._calculate_entropy(token_attn),
                    'top_region_index': int(np.argmax(token_attn)),
                    'concentration_ratio': float(np.max(token_attn) / (np.mean(token_attn) + 1e-8)),
                    'attention_distribution': token_attn.tolist()
                }
                
                token_stats.append(stats)
        
        return token_stats
    
    def _calculate_entropy(self, weights: np.ndarray) -> float:
        """Calculate Shannon entropy of attention weights."""
        weights = weights / (np.sum(weights) + 1e-12)
        weights = weights + 1e-12
        entropy = -np.sum(weights * np.log(weights))
        return float(entropy)
    
    def _generate_explorer_insights(self, token_stats: List[Dict[str, Any]], text_tokens: List[str]) -> List[str]:
        """Generate insights from token attention patterns."""
        insights = []
        
        if not token_stats:
            return ["No text token statistics available"]
        
        # Find most focused token
        most_focused = max(token_stats, key=lambda x: x['concentration_ratio'])
        insights.append(f"Text token '{most_focused['token']}' shows highest attention concentration (ratio: {most_focused['concentration_ratio']:.2f})")
        
        # Find most distributed token
        most_distributed = max(token_stats, key=lambda x: x['entropy'])
        insights.append(f"Text token '{most_distributed['token']}' has most distributed attention (entropy: {most_distributed['entropy']:.2f})")
        
        # Find token with strongest overall attention
        strongest_token = max(token_stats, key=lambda x: x['total_attention'])
        insights.append(f"Text token '{strongest_token['token']}' has strongest overall attention to image regions")
        
        # Analyze attention patterns
        high_concentration_tokens = [ts for ts in token_stats if ts['concentration_ratio'] > 3.0]
        if high_concentration_tokens:
            insights.append(f"{len(high_concentration_tokens)} text tokens show highly concentrated attention (focusing on specific regions)")
        
        # Analyze entropy distribution
        avg_entropy = np.mean([ts['entropy'] for ts in token_stats])
        if avg_entropy > 4.0:
            insights.append("Overall high entropy indicates text tokens consider multiple image regions")
        elif avg_entropy < 2.0:
            insights.append("Overall low entropy indicates text tokens focus on specific regions")
        
        return insights
    
    def _create_interactive_explorer(self, cross_attn: np.ndarray, text_tokens: List[str],
                                    image_patch_coords: List[Tuple[int, int]], 
                                    image: Optional[Image.Image],
                                    image_size: Optional[Tuple[int, int]],
                                    text_only_indices: Optional[List[int]] = None) -> go.Figure:
        """Create interactive cross-attention explorer with per-token attention maps overlaid on the actual image.
        
        Args:
            text_only_indices: Optional list of indices to include in dropdown. If provided, only these
                              token indices will appear in the dropdown menu (used to hide <image> tokens).
        """
        
        # Validate inputs
        if text_tokens is None or len(text_tokens) == 0:
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text="No text tokens available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig
        
        # Calculate grid dimensions from patch coordinates
        if image_patch_coords and len(image_patch_coords) > 0:
            # Infer grid size from patch coordinates
            unique_x = len(set(coord[0] for coord in image_patch_coords))
            unique_y = len(set(coord[1] for coord in image_patch_coords))
            grid_h = unique_y if unique_y > 0 else int(np.sqrt(len(image_patch_coords)))
            grid_w = unique_x if unique_x > 0 else int(np.sqrt(len(image_patch_coords)))
        else:
            # Fallback: assume square grid
            grid_size = int(np.sqrt(cross_attn.shape[1]))
            grid_h = grid_w = grid_size
        
        # Prepare image background if available
        img_width, img_height = image_size if image_size else (800, 600)
        
        # Create the figure with one large attention heatmap per token
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Text Token → Image Regions Attention',
                'Patch-Level Attention Strength'
            ),
            specs=[[{"type": "xy"}, {"type": "bar"}]],
            column_widths=[0.65, 0.35],
            horizontal_spacing=0.12
        )
        
        # Convert image to base64 for background (if available)
        img_str = None
        if image is not None:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            img_str = base64.b64encode(img_bytes).decode()
        
        # Create a heatmap trace for EACH text token
        # Determine which token should be visible initially
        first_visible_idx = text_only_indices[0] if text_only_indices and len(text_only_indices) > 0 else 0
        
        for token_idx, token in enumerate(text_tokens):
            if token_idx < cross_attn.shape[0]:
                token_attn = cross_attn[token_idx, :]
                
                # Normalize attention values for better visualization
                # This ensures each token's attention pattern is visible regardless of absolute scale
                token_attn_normalized = token_attn.copy()
                attn_min = token_attn_normalized.min()
                attn_max = token_attn_normalized.max()
                
                # Only normalize if there's variation in the attention values
                if attn_max > attn_min and attn_max > 1e-8:
                    # Min-max normalization to [0, 1] range for better visibility
                    token_attn_normalized = (token_attn_normalized - attn_min) / (attn_max - attn_min)
                elif attn_max <= 1e-8:
                    # All values are essentially zero - keep as is but user will see no pattern
                    token_attn_normalized = token_attn_normalized
                
                # Reshape to 2D grid for heatmap
                if len(token_attn_normalized) == grid_h * grid_w:
                    attention_grid = token_attn_normalized.reshape(grid_h, grid_w)
                else:
                    # Pad or truncate to fit grid
                    target_size = grid_h * grid_w
                    if len(token_attn_normalized) < target_size:
                        padded = np.pad(token_attn_normalized, (0, target_size - len(token_attn_normalized)), mode='constant')
                        attention_grid = padded.reshape(grid_h, grid_w)
                    else:
                        attention_grid = token_attn_normalized[:target_size].reshape(grid_h, grid_w)
                
                # Add heatmap for this token with image dimensions
                # Show both normalized (for visualization) and original (for accuracy) values
                fig.add_trace(
                    go.Heatmap(
                        z=attention_grid,
                        colorscale='Hot',
                        name=f"Text: {token}",
                        visible=(token_idx == first_visible_idx),  # Show first text-only token initially
                        hovertemplate=(
                            "Normalized: %{z:.4f}<br>"
                            "Original: %{customdata:.6f}<br>"
                            "Row: %{y}<br>Col: %{x}<extra></extra>"
                        ),
                        customdata=token_attn[:grid_h * grid_w].reshape(grid_h, grid_w) if len(token_attn) >= grid_h * grid_w 
                                   else np.pad(token_attn, (0, grid_h * grid_w - len(token_attn)), mode='constant').reshape(grid_h, grid_w),
                        colorbar=dict(
                            title=f"Normalized<br>Attention<br>(Range: {attn_min:.2e} to {attn_max:.2e})",
                            x=0.62, 
                            len=0.8
                        ),
                        showscale=(token_idx == first_visible_idx),  # Only show colorbar for first trace
                        zmin=0,
                        zmax=1
                    ),
                    row=1, col=1
                )
                
                # Add corresponding bar chart (initially only first text-only token visible)
                # Use ORIGINAL attention values for the bar chart (not normalized)
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(token_attn))),
                        y=token_attn,
                        name=f"Bar: {token}",
                        visible=(token_idx == first_visible_idx),
                        hovertemplate="Patch: %{x}<br>Original Attention: %{y:.6f}<extra></extra>",
                        marker_color='red',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
        
        # Add image background if available
        if img_str is not None:
            fig.update_layout(
                images=[dict(
                    source=f"data:image/png;base64,{img_str}",
                    xref="x", yref="y",
                    x=0, y=grid_h,
                    sizex=grid_w, sizey=grid_h,
                    sizing="stretch",
                    opacity=0.7,
                    layer="below"
                )]
            )
        
        # Create dropdown menu for token selection
        # If text_only_indices is provided, only show those tokens in dropdown
        dropdown_buttons = []
        indices_to_show = text_only_indices if text_only_indices is not None else list(range(len(text_tokens)))
        
        for token_idx in indices_to_show:
            if token_idx < len(text_tokens) and token_idx < cross_attn.shape[0]:
                token = text_tokens[token_idx]
                
                # Get the original attention range for this token for the colorbar
                token_attn = cross_attn[token_idx, :]
                attn_min = token_attn.min()
                attn_max = token_attn.max()
                
                # Create visibility array for this token
                visibility = [False] * (len(text_tokens) * 2)  # 2 traces per token
                visibility[token_idx * 2] = True      # Heatmap
                visibility[token_idx * 2 + 1] = True  # Bar chart
                
                # Update showscale for colorbars - only show for the active token
                showscale_array = [False] * (len(text_tokens) * 2)
                showscale_array[token_idx * 2] = True  # Only show colorbar for active heatmap
                
                dropdown_buttons.append(dict(
                    label=f"Text Token: {token}",
                    method="update",
                    args=[
                        {
                            "visible": visibility,
                            "showscale": showscale_array
                        },
                        {"title": f"Cross-Attention Explorer (Text) - Token: '{token}' (Original range: {attn_min:.2e} to {attn_max:.2e})"}
                    ]
                ))
        
        # Update layout with dropdown and styling
        fig.update_layout(
            title="Cross-Attention Explorer (Text) - Select a text token to see its attention pattern",
            height=600,
            updatemenus=[
                dict(
                    active=0,
                    buttons=dropdown_buttons,
                    direction="down",
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                )
            ],
            annotations=[
                dict(
                    text="Select Text Token:",
                    showarrow=False,
                    x=0.02, y=1.18,
                    xref="paper", yref="paper",
                    align="left",
                    font=dict(size=12)
                )
            ]
        )
        
        # Configure subplot layouts
        fig.update_xaxes(title_text="Image Patch Columns", row=1, col=1)
        fig.update_yaxes(title_text="Image Patch Rows", row=1, col=1, autorange="reversed")
        fig.update_xaxes(title_text="Image Patch Index", row=1, col=2)
        fig.update_yaxes(title_text="Attention Weight", row=1, col=2)
        
        return fig
    
    def _create_no_data_result(self) -> AnalysisResult:
        """Create result when no attention data is available."""
        fig = go.Figure()
        fig.update_layout(
            title="No Attention Data Available",
            annotations=[{
                'x': 0.5, 'y': 0.5,
                'text': "No attention data available for text token analysis",
                'showarrow': False, 'xref': 'paper', 'yref': 'paper',
                'font': {'size': 16, 'color': '#666'},
                'xanchor': 'center', 'yanchor': 'middle'
            }],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return AnalysisResult(
            analysis_type="cross_attention_explorer_text",
            title="Cross-Attention Explorer (Text)",
            description="No attention data available",
            visualization=fig,
            metrics={},
            insights=["No attention data available for analysis"]
        )
    
    def _create_no_text_tokens_result(self) -> AnalysisResult:
        """Create result when no text tokens are available after filtering."""
        fig = go.Figure()
        fig.update_layout(
            title="No Text Tokens Available",
            annotations=[{
                'x': 0.5, 'y': 0.5,
                'text': "No text tokens available after filtering out <image> tokens",
                'showarrow': False, 'xref': 'paper', 'yref': 'paper',
                'font': {'size': 16, 'color': '#666'},
                'xanchor': 'center', 'yanchor': 'middle'
            }],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return AnalysisResult(
            analysis_type="cross_attention_explorer_text",
            title="Cross-Attention Explorer (Text)",
            description="No text tokens available after filtering",
            visualization=fig,
            metrics={},
            insights=["Only <image> tokens found - no text tokens to analyze"]
        )


class AdvancedAnalysisManager:
    """Manager for all advanced analysis components."""
    
    def __init__(self):
        self.analyzers = {
            'statistical': StatisticalAnalyzer(),
            'attention_flow': AttentionFlowAnalyzer(),
            'layer_comparison': LayerComparisonAnalyzer(),
            'multi_head': MultiHeadAnalyzer(),
            'cross_attention_explorer': CrossAttentionExplorer(),
            'cross_attention_explorer_text': CrossAttentionExplorerText()
        }
    
    def get_available_analyses(self) -> List[str]:
        """Get list of available analysis types."""
        return list(self.analyzers.keys())
    
    def run_analysis(self, analysis_type: str, attention_output: AttentionOutput,
                    image: Optional[Image.Image] = None, question: Optional[str] = None,
                    **kwargs) -> AnalysisResult:
        """Run a specific analysis type."""
        if analysis_type not in self.analyzers:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        analyzer = self.analyzers[analysis_type]
        return analyzer.analyze(attention_output, image, question, **kwargs)
    
    def run_all_analyses(self, attention_output: AttentionOutput,
                        image: Optional[Image.Image] = None, question: Optional[str] = None,
                        **kwargs) -> Dict[str, AnalysisResult]:
        """Run all available analyses."""
        results = {}
        for analysis_type, analyzer in self.analyzers.items():
            try:
                results[analysis_type] = analyzer.analyze(attention_output, image, question, **kwargs)
            except Exception as e:
                logger.error(f"Error running {analysis_type} analysis: {e}")
                # Create error result
                fig = go.Figure()
                fig.update_layout(title=f"Error in {analysis_type} analysis")
                results[analysis_type] = AnalysisResult(
                    analysis_type=analysis_type,
                    title=f"{analysis_type.title()} Analysis Error",
                    description=f"Error occurred: {str(e)}",
                    visualization=fig,
                    metrics={},
                    insights=[f"Analysis failed: {str(e)}"]
                )
        
        return results





