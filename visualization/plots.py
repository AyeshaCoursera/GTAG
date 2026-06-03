"""Static visualization."""

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
        plt.b
