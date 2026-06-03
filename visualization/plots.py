"""Static visualization"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from sklearn.decomposition import PCA


class StaticVisualizer:
    """Create static plots for transcriptomic analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        sns.set_style(style)
        
    def plot_tissue_heatmap(self, tissue_results: Dict, output_path: str = None):
        """Plot tissue-specific gene expression heatmap."""
        tissues = list(tissue_results.keys())[:4]
        genes_to_plot = []
        for tissue in tissues:
            genes_to_plot.extend(tissue_results[tissue]['top_genes'][:5])
        genes_to_plot = list(set(genes_to_plot))[:15]
        
        heatmap_data = []
        for tissue in tissues:
            tissue_genes = tissue_results[tissue]['top_genes']
            tissue_expr = tissue_results[tissue]['expression_values']
            row = []
            for gene in genes_to_plot:
                if gene in tissue_genes:
                    idx = tissue_genes.index(gene)
                    row.append(tissue_expr[idx])
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, xticklabels=genes_to_plot,
                   yticklabels=[t[:20] for t in tissues],
                   cmap='YlOrRd', annot=False,
                   cbar_kws={'label': 'Expression Level'})
        plt.title('Tissue-Specific Gene Expression Patterns')
        plt.xlabel('Genes')
        plt.ylabel('Tissues')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_network_complexity(self, tissue_networks: Dict, output_path: str = None):
        """Plot network complexity by tissue."""
        tissues_net = list(tissue_networks.keys())
        complexities = [tissue_networks[t].get('total_pairs', 0) for t in tissues_net]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(tissues_net)), complexities, color='steelblue')
        plt.xticks(range(len(tissues_net)), [t[:50] + '...' for t in tissues_net])
        plt.ylabel('Number of Strong Co-expression Pairs')
        plt.title('Gene Network Complexity by Tissue')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_pca(self, expr_matrix: np.ndarray, tissue_labels: np.ndarray,
                 output_path: str = None):
        """Plot PCA of expression data."""
        pca_cells = min(2000, expr_matrix.shape[0])
        pca_indices = np.random.choice(expr_matrix.shape[0], pca_cells, replace=False)
        pca_data = expr_matrix[pca_indices, :50]
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pca_data)
        pca_tissues = tissue_labels[pca_indices]
        
        plt.figure(figsize=(10, 8))
        unique_tissues = np.unique(pca_tissues)[:8]
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_tissues)))
        
        for i, tissue in enumerate(unique_tissues):
            mask = pca_tissues == tissue
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1],
                       label=tissue[:15] + '...', alpha=0.6, s=20, color=colors[i])
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('PCA: Tissue Separation by Gene Expression')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_validation_rates(self, validation_results: Dict, output_path: str = None):
        """Plot database validation rates."""
        df = pd.DataFrame([
            {'Tissue': tissue, 'Validation_Rate': data['validation_rate']}
            for tissue, data in validation_results.items()
        ])
        
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df, x='Tissue', y='Validation_Rate', palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.title('Database Validation Rates for Genes Across Tissues')
        plt.xlabel('Tissue')
        plt.ylabel('Validation Rate')
        plt.ylim(0, 1)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
