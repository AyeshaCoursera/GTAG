"""Interactive visualization utilities using Plotly."""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Optional, Dict


class InteractiveVisualizer:
    """Create interactive Plotly visualizations."""
    
    def __init__(self, output_dir: str = "outputs"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
    def pca_plot(self, pca_result: np.ndarray, labels: np.ndarray,
                 title: str = "Interactive PCA Plot") -> go.Figure:
        """Create interactive PCA plot."""
        fig = px.scatter(
            x=pca_result[:, 0], y=pca_result[:, 1],
            color=labels, title=title,
            labels={'x': 'PC1', 'y': 'PC2', 'color': 'Tissue'}
        )
        fig.update_layout(width=900, height=700, template='plotly_white')
        
        filepath = f"{self.output_dir}/pca_plot.html"
        fig.write_html(filepath)
        print(f"Saved: {filepath}")
        return fig
    
    def heatmap(self, data_matrix: np.ndarray, row_labels: List[str],
                col_labels: List[str], title: str = "Expression Heatmap") -> go.Figure:
        """Create interactive heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix, x=col_labels, y=row_labels,
            colorscale='YlOrRd', hoverongaps=False,
            hovertemplate='Tissue: %{y}<br>Gene: %{x}<br>Expression: %{z:.3f}<extra></extra>'
        ))
        fig.update_layout(title=title, width=1000, height=600,
                         xaxis=dict(tickangle=-45), template='plotly_white')
        
        filepath = f"{self.output_dir}/heatmap.html"
        fig.write_html(filepath)
        print(f"Saved: {filepath}")
        return fig
    
    def shap_summary(self, shap_values: np.ndarray, feature_names: List[str],
                     feature_values: np.ndarray, title: str = "SHAP Summary") -> go.Figure:
        """Create interactive SHAP summary plot."""
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1][:20]
        
        plot_data = []
        for rank, idx in enumerate(sorted_idx):
            for sample in range(min(shap_values.shape[0], 100)):
                plot_data.append({
                    'gene': feature_names[idx],
                    'shap_value': shap_values[sample, idx],
                    'feature_value': feature_values[sample, idx]
                })
        
        df = pd.DataFrame(plot_data)
        fig = px.scatter(df, x='shap_value', y='gene', color='feature_value',
                        color_continuous_scale='RdYlBu', title=title,
                        labels={'shap_value': 'SHAP value', 'gene': ''})
        fig.update_layout(width=1000, height=600, template='plotly_white')
        
        filepath = f"{self.output_dir}/shap_plot.html"
        fig.write_html(filepath)
        print(f"Saved: {filepath}")
        return fig
    
    def tsne_plot(self, tsne_result: np.ndarray, labels: np.ndarray,
                  title: str = "t-SNE Visualization") -> go.Figure:
        """Create interactive t-SNE plot."""
        fig = px.scatter(x=tsne_result[:, 0], y=tsne_result[:, 1],
                        color=labels, title=title,
                        labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Tissue'})
        fig.update_layout(width=900, height=700, template='plotly_white')
        
        filepath = f"{self.output_dir}/tsne_plot.html"
        fig.write_html(filepath)
        print(f"Saved: {filepath}")
        return fig
