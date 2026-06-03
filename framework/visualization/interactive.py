"""Interactive visualizations using Plotly for The Visual Computer."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class InteractiveVisualizer:
    """
    Interactive visualization module for transcriptomic data.
    Creates Plotly-based interactive plots suitable for web-based exploration.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def interactive_pca_plot(self,
                             pca_result: np.ndarray,
                             tissue_labels: np.ndarray,
                             gene_names: Optional[List[str]] = None,
                             title: str = "Interactive PCA Plot") -> go.Figure:
        """
        Create an interactive PCA plot with hover tooltips.
        
        Parameters
        ----------
        pca_result : np.ndarray
            PCA coordinates (n_samples × 2)
        tissue_labels : np.ndarray
            Tissue labels for each point
        gene_names : list, optional
            Gene names for hover information
        title : str
            Plot title
            
        Returns
        -------
        plotly.graph_objects.Figure
        """
        # Prepare hover text
        hover_text = [f"Tissue: {t}" for t in tissue_labels]
        if gene_names is not None and len(gene_names) == len(tissue_labels):
            hover_text = [f"{h}<br>Top gene: {g}" for h, g in zip(hover_text, gene_names)]
        
        fig = px.scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            color=tissue_labels,
            hover_name=hover_text,
            title=title,
            labels={'x': 'Principal Component 1', 'y': 'Principal Component 2', 'color': 'Tissue'}
        )
        
        fig.update_layout(
            width=900,
            height=700,
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig
    
    def interactive_heatmap(self,
                           data_matrix: np.ndarray,
                           row_labels: List[str],
                           col_labels: List[str],
                           title: str = "Interactive Expression Heatmap",
                           colorscale: str = "YlOrRd") -> go.Figure:
        """
        Create an interactive heatmap with clickable rows/columns.
        
        Parameters
        ----------
        data_matrix : np.ndarray
            Data matrix (rows × columns)
        row_labels : list
            Labels for rows (e.g., tissues)
        col_labels : list
            Labels for columns (e.g., genes)
        title : str
            Plot title
        colorscale : str
            Plotly colorscale name
            
        Returns
        -------
        plotly.graph_objects.Figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=col_labels,
            y=row_labels,
            colorscale=colorscale,
            hoverongaps=False,
            hovertemplate='Tissue: %{y}<br>Gene: %{x}<br>Expression: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            width=1000,
            height=600,
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
            template='plotly_white'
        )
        
        return fig
    
    def interactive_coexpression_network(self,
                                        correlation_matrix: np.ndarray,
                                        gene_names: List[str],
                                        threshold: float = 0.6,
                                        title: str = "Co-expression Network") -> go.Figure:
        """
        Create an interactive network visualization of gene co-expression.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix (genes × genes)
        gene_names : list
            Gene names
        threshold : float
            Minimum correlation to show an edge
        title : str
            Plot title
            
        Returns
        -------
        plotly.graph_objects.Figure
        """
        # Build edge list
        edges = []
        n_genes = len(gene_names)
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                if abs(correlation_matrix[i, j]) > threshold:
                    edges.append({
                        'source': gene_names[i],
                        'target': gene_names[j],
                        'weight': correlation_matrix[i, j]
                    })
        
        # Create network graph
        fig = go.Figure()
        
        # Add edges
        for edge in edges:
            fig.add_trace(go.Scatter(
                x=[edge['source'], edge['target']],
                y=[0, 0],  # Placeholder - would need layout algorithm
                mode='lines',
                line=dict(width=abs(edge['weight']) * 3, color=('red' if edge['weight'] > 0 else 'blue')),
                hoverinfo='text',
                text=f"{edge['source']} ↔ {edge['target']}: r={edge['weight']:.3f}",
                showlegend=False
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=list(range(n_genes)),
            y=[0] * n_genes,
            mode='markers+text',
            marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
            text=gene_names,
            textposition='middle center',
            hoverinfo='text',
            hovertext=gene_names,
            showlegend=False
        ))
        
        fig.update_layout(
            title=title,
            width=1200,
            height=800,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            template='plotly_white'
        )
        
        return fig
    
    def interactive_shap_summary(self,
                                shap_values: np.ndarray,
                                feature_names: List[str],
                                feature_values: np.ndarray,
                                title: str = "SHAP Feature Importance Summary") -> go.Figure:
        """
        Create an interactive SHAP summary plot.
        
        Parameters
        ----------
        shap_values : np.ndarray
            SHAP values (samples × features)
        feature_names : list
            Feature (gene) names
        feature_values : np.ndarray
            Original feature values for coloring
        title : str
            Plot title
            
        Returns
        -------
        plotly.graph_objects.Figure
        """
        # Calculate mean absolute SHAP for ordering
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        # Prepare data for plotting
        top_n = min(20, len(sorted_idx))
        plot_data = []
        for rank, idx in enumerate(sorted_idx[:top_n]):
            for sample in range(shap_values.shape[0]):
                plot_data.append({
                    'gene': feature_names[idx],
                    'shap_value': shap_values[sample, idx],
                    'feature_value': feature_values[sample, idx],
                    'rank': rank
                })
        
        df = pd.DataFrame(plot_data)
        
        fig = px.scatter(
            df,
            x='shap_value',
            y='gene',
            color='feature_value',
            color_continuous_scale='RdYlBu',
            title=title,
            labels={'shap_value': 'SHAP value (impact on model output)', 'gene': '', 'feature_value': 'Gene expression'},
            hover_data={'rank': False}
        )
        
        fig.update_layout(
            width=1000,
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def interactive_tsne_plot(self,
                             tsne_result: np.ndarray,
                             labels: np.ndarray,
                             is_real: np.ndarray,
                             title: str = "Interactive t-SNE: Real vs Synthetic") -> go.Figure:
        """
        Create an interactive t-SNE plot comparing real and synthetic data.
        
        Parameters
        ----------
        tsne_result : np.ndarray
            t-SNE coordinates (n_samples × 2)
        labels : np.ndarray
            Tissue or class labels
        is_real : np.ndarray
            Boolean array indicating real (True) vs synthetic (False)
        title : str
            Plot title
            
        Returns
        -------
        plotly.graph_objects.Figure
        """
        # Create combined label for coloring
        combined_labels = [f"{'Real' if r else 'Synthetic'}: {l}" for r, l in zip(is_real, labels)]
        
        fig = px.scatter(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            color=combined_labels,
            title=title,
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Data Type'},
            hover_data={'label': labels}
        )
        
        fig.update_layout(
            width=900,
            height=700,
            template='plotly_white'
        )
        
        return fig
    
    def save_interactive_plot(self, fig: go.Figure, filename: str):
        """Save interactive plot as HTML file."""
        filepath = f"{self.output_dir}/{filename}.html"
        fig.write_html(filepath)
        print(f"Saved interactive plot: {filepath}")
        return filepath
    
    def create_interactive_dashboard(self,
                                    pca_fig: go.Figure,
                                    heatmap_fig: go.Figure,
                                    shap_fig: go.Figure,
                                    output_file: str = "transcriptomic_dashboard") -> None:
        """
        Combine multiple interactive plots into a single HTML dashboard.
        
        Parameters
        ----------
        pca_fig : go.Figure
            PCA plot figure
        heatmap_fig : go.Figure
            Heatmap figure
        shap_fig : go.Figure
            SHAP summary figure
        output_file : str
            Output filename (without extension)
        """
        from plotly.subplots import make_subplots
        
        # Create subplot dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PCA: Tissue Separation', 'Tissue-Specific Expression Heatmap',
                           'SHAP Feature Importance', 'Model Performance'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Add traces from each figure
        for trace in pca_fig.data:
            fig.add_trace(trace, row=1, col=1)
        for trace in heatmap_fig.data:
            fig.add_trace(trace, row=1, col=2)
        for trace in shap_fig.data:
            fig.add_trace(trace, row=2, col=1)
        
        fig.update_layout(
            title="Transcriptomic Visual Analytics Dashboard",
            width=1400,
            height=1000,
            template='plotly_white',
            showlegend=True
        )
        
        filepath = f"{self.output_dir}/{output_file}.html"
        fig.write_html(filepath)
        print(f"Saved interactive dashboard: {filepath}")
