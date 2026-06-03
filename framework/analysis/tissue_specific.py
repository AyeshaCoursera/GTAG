"""Tissue-specific gene expression analysis."""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


class TissueSpecificAnalyzer:
    """Analyze tissue-specific gene expression patterns."""
    
    def __init__(self, expr_matrix: np.ndarray, gene_names: List[str], 
                 tissue_labels: np.ndarray):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        expr_matrix : np.ndarray
            Expression matrix (cells x genes)
        gene_names : list
            Gene names
        tissue_labels : np.ndarray
            Tissue labels for each cell
        """
        self.expr_matrix = expr_matrix
        self.gene_names = gene_names
        self.tissue_labels = tissue_labels
        
    def analyze_tissue_specificity(self, top_n: int = 15, 
                                   min_cells: int = 100) -> Dict:
        """
        Identify top tissue-specific genes.
        
        Parameters:
        -----------
        top_n : int
            Number of top genes to return per tissue
        min_cells : int
            Minimum number of cells required for tissue analysis
            
        Returns:
        --------
        dict: Tissue-specific gene results
        """
        tissues = np.unique(self.tissue_labels)
        tissue_results = {}
        
        for tissue in tissues:
            tissue_mask = self.tissue_labels == tissue
            if np.sum(tissue_mask) < min_cells:
                continue
                
            tissue_expr = self.expr_matrix[tissue_mask, :]
            mean_expression = np.mean(tissue_expr, axis=0)
            
            top_indices = np.argsort(mean_expression)[-top_n:][::-1]
            top_genes = [self.gene_names[i] for i in top_indices]
            top_values = mean_expression[top_indices]
            
            tissue_results[tissue] = {
                'top_genes': top_genes,
                'expression_values': top_values.tolist(),
                'cell_count': int(np.sum(tissue_mask)),
                'mean_expression': mean_expression.tolist()
            }
            
        return tissue_results
    
    def get_unique_genes(self, tissue_results: Dict) -> List[str]:
        """Get all unique genes across tissues."""
        all_genes = set()
        for data in tissue_results.values():
            all_genes.update(data['top_genes'])
        return list(all_genes)
    
    def generate_gene_descriptions(self, tissue_results: Dict, 
                                   max_genes: int = 50) -> List[Dict]:
        """Generate natural language descriptions for genes."""
        all_genes = self.get_unique_genes(tissue_results)
        gene_descriptions = []
        
        for gene in list(all_genes)[:max_genes]:
            tissues_with_gene = []
            for tissue, data in tissue_results.items():
                if gene in data['top_genes']:
                    idx = data['top_genes'].index(gene)
                    expr_level = data['expression_values'][idx]
                    tissues_with_gene.append((tissue, float(expr_level)))
            
            if tissues_with_gene:
                tissues_with_gene.sort(key=lambda x: x[1], reverse=True)
                
                if len(tissues_with_gene) == 1:
                    desc = f"The gene {gene} is specifically highly expressed in {tissues_with_gene[0][0]} tissue."
                elif len(tissues_with_gene) >= 3:
                    top_tissues = [t[0][:15] for t in tissues_with_gene[:3]]
                    desc = f"{gene} shows high expression in {len(tissues_with_gene)} tissues including {', '.join(top_tissues)}."
                else:
                    desc = f"{gene} is expressed in {len(tissues_with_gene)} different tissues."
                
                gene_descriptions.append({
                    'gene': gene,
                    'description': desc,
                    'tissues': [t[0] for t in tissues_with_gene],
                    'max_expression': float(max([t[1] for t in tissues_with_gene]))
                })
        
        return gene_descriptions
