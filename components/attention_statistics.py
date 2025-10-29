"""
Advanced attention statistics and analysis component.
Provides detailed statistical analysis of attention patterns.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import torch

from models.base_model import AttentionOutput


class AttentionStatistics:
    """Component for advanced statistical analysis of attention patterns."""
    
    def __init__(self):
        """Initialize the attention statistics component."""
        self.color_palette = px.colors.qualitative.Set1
    
    def create_attention_distribution_analysis(self,
                                             attention_output: AttentionOutput,
                                             question: str) -> go.Figure:
        """
        Create comprehensive attention distribution analysis.
        
        Args:
            attention_output: Attention analysis results
            question: Question text
            
        Returns:
            Plotly figure with distribution analysis
        """
        if attention_output.cross_attention is None:
            raise ValueError("Cross-attention data required")
        
        cross_attn = attention_output.cross_attention
        while cross_attn.dim() > 2:
            cross_attn = cross_attn.mean(dim=0)
        
        attn_weights = cross_attn.detach().cpu().numpy()
        
        # Calculate various distributions
        token_attention = np.sum(attn_weights, axis=1)  # Attention per token
        patch_attention = np.sum(attn_weights, axis=0)  # Attention per patch
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Token Attention Distribution",
                "Patch Attention Distribution", 
                "Attention Heatmap",
                "Statistical Summary"
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "table"}]
            ]
        )
        
        # Token attention distribution
        token_labels = attention_output.text_tokens if attention_output.text_tokens else \
                      [f"Token_{i}" for i in range(len(token_attention))]
        
        fig.add_trace(
            go.Bar(
                x=token_labels,
                y=token_attention,
                name="Token Attention",
                marker_color=self.color_palette[0],
                hovertemplate="<b>%{x}</b><br>Attention: %{y:.4f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Patch attention distribution
        patch_labels = [f"Patch_{i}" for i in range(len(patch_attention))]
        fig.add_trace(
            go.Bar(
                x=patch_labels,
                y=patch_attention,
                name="Patch Attention",
                marker_color=self.color_palette[1],
                hovertemplate="<b>%{x}</b><br>Attention: %{y:.4f}<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Attention heatmap
        fig.add_trace(
            go.Heatmap(
                z=attn_weights,
                x=patch_labels,
                y=token_labels,
                colorscale="Viridis",
                name="Attention Matrix",
                hovertemplate="Token: %{y}<br>Patch: %{x}<br>Attention: %{z:.4f}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Statistical summary
        stats_data = self._calculate_attention_statistics(
            attn_weights, token_attention, patch_attention
        )
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color="lightblue",
                    font=dict(size=12, color="black"),
                    align="left"
                ),
                cells=dict(
                    values=[list(stats_data.keys()), list(stats_data.values())],
                    fill_color="white",
                    font=dict(size=11),
                    align="left"
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Attention Distribution Analysis: {question}",
            height=800,
            showlegend=False
        )
        
        # Update x-axis for token plot
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        
        return fig
    
    def create_attention_clustering_analysis(self,
                                           attention_output: AttentionOutput,
                                           question: str,
                                           n_clusters: int = 3) -> go.Figure:
        """
        Create attention pattern clustering analysis.
        
        Args:
            attention_output: Attention analysis results
            question: Question text
            n_clusters: Number of clusters for analysis
            
        Returns:
            Plotly figure with clustering analysis
        """
        if attention_output.cross_attention is None:
            raise ValueError("Cross-attention data required")
        
        cross_attn = attention_output.cross_attention
        while cross_attn.dim() > 2:
            cross_attn = cross_attn.mean(dim=0)
        
        attn_weights = cross_attn.detach().cpu().numpy()
        
        # Perform clustering on tokens based on their attention patterns
        if len(attn_weights) < n_clusters:
            n_clusters = len(attn_weights)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        token_clusters = kmeans.fit_predict(attn_weights)
        
        # Calculate silhouette score
        if len(np.unique(token_clusters)) > 1:
            silhouette_avg = silhouette_score(attn_weights, token_clusters)
        else:
            silhouette_avg = 0.0
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Token Clusters (PCA Projection)",
                "Cluster Attention Patterns",
                "Cluster Statistics",
                "Attention Similarity Matrix"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "heatmap"}]
            ]
        )
        
        # PCA projection for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        
        if attn_weights.shape[1] >= 2:
            tokens_2d = pca.fit_transform(attn_weights)
        else:
            # Fallback for low-dimensional data
            tokens_2d = np.column_stack([
                np.arange(len(attn_weights)),
                np.sum(attn_weights, axis=1)
            ])
        
        # Token cluster scatter plot
        for cluster_id in range(n_clusters):
            cluster_mask = token_clusters == cluster_id
            cluster_tokens = tokens_2d[cluster_mask]
            
            if len(cluster_tokens) > 0:
                token_names = [attention_output.text_tokens[i] if attention_output.text_tokens else f"Token_{i}" 
                             for i in range(len(token_clusters)) if token_clusters[i] == cluster_id]
                
                fig.add_trace(
                    go.Scatter(
                        x=cluster_tokens[:, 0],
                        y=cluster_tokens[:, 1],
                        mode="markers+text",
                        name=f"Cluster {cluster_id}",
                        text=token_names,
                        textposition="top center",
                        marker=dict(
                            size=10,
                            color=self.color_palette[cluster_id % len(self.color_palette)]
                        ),
                        hovertemplate="<b>%{text}</b><br>Cluster: " + str(cluster_id) + "<extra></extra>"
                    ),
                    row=1, col=1
                )
        
        # Cluster attention patterns
        cluster_means = []
        for cluster_id in range(n_clusters):
            cluster_mask = token_clusters == cluster_id
            if np.any(cluster_mask):
                cluster_mean = np.mean(attn_weights[cluster_mask], axis=0)
                cluster_means.append(cluster_mean)
            else:
                cluster_means.append(np.zeros(attn_weights.shape[1]))
        
        cluster_means = np.array(cluster_means)
        
        for cluster_id in range(n_clusters):
            fig.add_trace(
                go.Bar(
                    x=[f"Patch_{i}" for i in range(len(cluster_means[cluster_id]))],
                    y=cluster_means[cluster_id],
                    name=f"Cluster {cluster_id} Avg",
                    marker_color=self.color_palette[cluster_id % len(self.color_palette)],
                    opacity=0.7,
                    hovertemplate=f"Cluster {cluster_id}<br>Patch: %{{x}}<br>Avg Attention: %{{y:.4f}}<extra></extra>"
                ),
                row=1, col=2
            )
        
        # Cluster statistics
        cluster_stats = self._calculate_cluster_statistics(
            attn_weights, token_clusters, silhouette_avg
        )
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color="lightgreen",
                    font=dict(size=12),
                    align="left"
                ),
                cells=dict(
                    values=[list(cluster_stats.keys()), list(cluster_stats.values())],
                    fill_color="white",
                    font=dict(size=11),
                    align="left"
                )
            ),
            row=2, col=1
        )
        
        # Attention similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(attn_weights)
        token_labels = attention_output.text_tokens if attention_output.text_tokens else \
                      [f"Token_{i}" for i in range(len(attn_weights))]
        
        fig.add_trace(
            go.Heatmap(
                z=similarity_matrix,
                x=token_labels,
                y=token_labels,
                colorscale="RdBu",
                zmid=0,
                hovertemplate="Token 1: %{y}<br>Token 2: %{x}<br>Similarity: %{z:.3f}<extra></extra>"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Attention Clustering Analysis: {question}",
            height=900,
            showlegend=True
        )
        
        return fig
    
    def create_comparative_attention_analysis(self,
                                            attention_outputs: List[AttentionOutput],
                                            model_names: List[str],
                                            question: str) -> go.Figure:
        """
        Create comparative analysis across multiple models.
        
        Args:
            attention_outputs: List of attention analysis results
            model_names: Names of the models
            question: Question text
            
        Returns:
            Plotly figure with comparative analysis
        """
        if len(attention_outputs) != len(model_names):
            raise ValueError("Number of attention outputs must match model names")
        
        # Calculate statistics for each model
        model_stats = []
        attention_matrices = []
        
        for output, model_name in zip(attention_outputs, model_names):
            if output.cross_attention is None:
                continue
            
            cross_attn = output.cross_attention
            while cross_attn.dim() > 2:
                cross_attn = cross_attn.mean(dim=0)
            
            attn_weights = cross_attn.detach().cpu().numpy()
            attention_matrices.append(attn_weights)
            
            # Calculate comprehensive statistics
            stats = self._calculate_comprehensive_statistics(attn_weights, model_name)
            model_stats.append(stats)
        
        if not model_stats:
            raise ValueError("No valid attention data found")
        
        # Create comparative visualization
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Attention Entropy Comparison",
                "Attention Concentration Comparison",
                "Focus Diversity Comparison", 
                "Token Importance Variance",
                "Model Attention Patterns",
                "Statistical Summary"
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "table"}]
            ]
        )
        
        # Extract metrics for comparison
        entropies = [stats['entropy'] for stats in model_stats]
        concentrations = [stats['gini_coefficient'] for stats in model_stats]
        diversities = [stats['attention_diversity'] for stats in model_stats]
        token_variances = [stats['token_variance'] for stats in model_stats]
        
        # Entropy comparison
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=entropies,
                name="Entropy",
                marker_color=self.color_palette[0],
                hovertemplate="<b>%{x}</b><br>Entropy: %{y:.3f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Concentration comparison
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=concentrations,
                name="Gini Coefficient",
                marker_color=self.color_palette[1],
                hovertemplate="<b>%{x}</b><br>Gini: %{y:.3f}<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Diversity comparison
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=diversities,
                name="Attention Diversity",
                marker_color=self.color_palette[2],
                hovertemplate="<b>%{x}</b><br>Diversity: %{y:.3f}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Token variance comparison
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=token_variances,
                name="Token Variance",
                marker_color=self.color_palette[3],
                hovertemplate="<b>%{x}</b><br>Variance: %{y:.3f}<extra></extra>"
            ),
            row=2, col=2
        )
        
        # Model attention patterns heatmap
        if attention_matrices:
            # Create average attention pattern for each model
            max_tokens = max(matrix.shape[0] for matrix in attention_matrices)
            max_patches = max(matrix.shape[1] for matrix in attention_matrices)
            
            # Normalize all matrices to same size and create combined heatmap
            combined_matrix = np.zeros((len(model_names), max_patches))
            for i, matrix in enumerate(attention_matrices):
                # Sum over tokens to get patch attention
                patch_attention = np.sum(matrix, axis=0)
                # Pad or truncate to max_patches
                if len(patch_attention) <= max_patches:
                    combined_matrix[i, :len(patch_attention)] = patch_attention
                else:
                    combined_matrix[i, :] = patch_attention[:max_patches]
            
            fig.add_trace(
                go.Heatmap(
                    z=combined_matrix,
                    x=[f"Patch_{i}" for i in range(max_patches)],
                    y=model_names,
                    colorscale="Viridis",
                    hovertemplate="Model: %{y}<br>Patch: %{x}<br>Attention: %{z:.4f}<extra></extra>"
                ),
                row=3, col=1
            )
        
        # Summary table
        summary_data = self._create_comparative_summary(model_stats)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Model"] + list(summary_data[0].keys())[1:],
                    fill_color="lightcoral",
                    font=dict(size=11),
                    align="left"
                ),
                cells=dict(
                    values=[[row[key] for row in summary_data] for key in summary_data[0].keys()],
                    fill_color="white",
                    font=dict(size=10),
                    align="left"
                )
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title=f"Comparative Attention Analysis: {question}",
            height=1200,
            showlegend=False
        )
        
        return fig
    
    def _calculate_attention_statistics(self,
                                      attn_weights: np.ndarray,
                                      token_attention: np.ndarray,
                                      patch_attention: np.ndarray) -> Dict[str, str]:
        """Calculate comprehensive attention statistics."""
        # Flatten for overall statistics
        flat_weights = attn_weights.flatten()
        
        # Basic statistics
        from scipy import stats as scipy_stats
        
        attention_stats = {
            "Total Attention": f"{np.sum(flat_weights):.4f}",
            "Mean Attention": f"{np.mean(flat_weights):.4f}",
            "Std Deviation": f"{np.std(flat_weights):.4f}",
            "Max Attention": f"{np.max(flat_weights):.4f}",
            "Min Attention": f"{np.min(flat_weights):.4f}",
            "Median Attention": f"{np.median(flat_weights):.4f}",
            "Skewness": f"{scipy_stats.skew(flat_weights):.4f}",
            "Kurtosis": f"{scipy_stats.kurtosis(flat_weights):.4f}",
        }
        
        # Entropy calculation
        normalized_weights = flat_weights / np.sum(flat_weights)
        entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-12))
        attention_stats["Entropy"] = f"{entropy:.4f}"
        
        # Gini coefficient (concentration measure)
        sorted_weights = np.sort(flat_weights)
        n = len(sorted_weights)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
        attention_stats["Gini Coefficient"] = f"{gini:.4f}"
        
        # Token-level statistics
        attention_stats["Token Attention Var"] = f"{np.var(token_attention):.4f}"
        attention_stats["Max Token Attention"] = f"{np.max(token_attention):.4f}"
        
        # Patch-level statistics  
        attention_stats["Patch Attention Var"] = f"{np.var(patch_attention):.4f}"
        attention_stats["Max Patch Attention"] = f"{np.max(patch_attention):.4f}"
        
        return attention_stats
    
    def _calculate_cluster_statistics(self,
                                    attn_weights: np.ndarray,
                                    token_clusters: np.ndarray,
                                    silhouette_avg: float) -> Dict[str, str]:
        """Calculate clustering-specific statistics."""
        n_clusters = len(np.unique(token_clusters))
        
        stats = {
            "Number of Clusters": str(n_clusters),
            "Silhouette Score": f"{silhouette_avg:.4f}",
            "Largest Cluster Size": str(np.max(np.bincount(token_clusters))),
            "Smallest Cluster Size": str(np.min(np.bincount(token_clusters))),
        }
        
        # Calculate within-cluster variance
        total_variance = 0
        for cluster_id in range(n_clusters):
            cluster_mask = token_clusters == cluster_id
            if np.any(cluster_mask):
                cluster_data = attn_weights[cluster_mask]
                cluster_variance = np.var(cluster_data)
                total_variance += cluster_variance
        
        stats["Avg Within-Cluster Var"] = f"{total_variance / n_clusters:.4f}"
        
        return stats
    
    def _calculate_similarity_matrix(self, attn_weights: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix between tokens."""
        n_tokens = attn_weights.shape[0]
        similarity_matrix = np.zeros((n_tokens, n_tokens))
        
        for i in range(n_tokens):
            for j in range(n_tokens):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Calculate cosine similarity
                    sim = 1 - cosine(attn_weights[i], attn_weights[j])
                    similarity_matrix[i, j] = sim
        
        return similarity_matrix
    
    def _calculate_comprehensive_statistics(self,
                                          attn_weights: np.ndarray,
                                          model_name: str) -> Dict[str, Union[str, float]]:
        """Calculate comprehensive statistics for a single model."""
        flat_weights = attn_weights.flatten()
        token_attention = np.sum(attn_weights, axis=1)
        patch_attention = np.sum(attn_weights, axis=0)
        
        # Entropy
        normalized_weights = flat_weights / np.sum(flat_weights)
        entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-12))
        
        # Gini coefficient
        sorted_weights = np.sort(flat_weights)
        n = len(sorted_weights)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
        
        # Attention diversity (effective number of attention points)
        attention_diversity = np.exp(entropy)
        
        return {
            "model_name": model_name,
            "entropy": entropy,
            "gini_coefficient": gini,
            "attention_diversity": attention_diversity,
            "max_attention": np.max(flat_weights),
            "token_variance": np.var(token_attention),
            "patch_variance": np.var(patch_attention),
            "mean_attention": np.mean(flat_weights)
        }
    
    def _create_comparative_summary(self, model_stats: List[Dict]) -> List[Dict[str, str]]:
        """Create a summary table for comparative analysis."""
        summary = []
        for stats in model_stats:
            summary.append({
                "Model": stats["model_name"],
                "Entropy": f"{stats['entropy']:.3f}",
                "Gini Coeff": f"{stats['gini_coefficient']:.3f}",
                "Diversity": f"{stats['attention_diversity']:.3f}",
                "Max Attn": f"{stats['max_attention']:.3f}",
                "Token Var": f"{stats['token_variance']:.3f}",
                "Patch Var": f"{stats['patch_variance']:.3f}"
            })
        
        return summary




