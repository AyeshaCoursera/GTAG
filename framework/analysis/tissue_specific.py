"""Statistical analysis for tissue-specific gene expression."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple


class TissueSpecificAnalyzer:
    """
    Identify tissue-specific highly expressed genes using 
    fold-change and FDR-corrected statistical testing.
    """
    
    def __init__(self, lfc_threshold: float = 1.0, fdr_threshold: float = 0.05):
        self.lfc_threshold = lfc_threshold
        self.fdr_threshold = fdr_threshold
        self.epsilon = 1e-6
    
    def identify_tissue_specific_genes(self,
                                       expr_matrix: np.ndarray,
                                       gene_names: List[str],
                                       tissue_labels: np.ndarray,
                                       top_n: int = 15) -> Dict:
        """
        Identify top N highly expressed genes for each tissue.
        
        Parameters
        ----------
        expr_matrix : np.ndarray
            Expression matrix (cells × genes)
        gene_names : list
            List of gene names
        tissue_labels : np.ndarray
            Tissue labels for each cell
        top_n : int
            Number of top genes to return per tissue
            
        Returns
        -------
        dict : Tissue-specific expression results
        """
        tissues = np.unique(tissue_labels)
        tissue_results = {}
        
        for tissue in tissues:
            mask = tissue_labels == tissue
            if np.sum(mask) < 10:
                continue
                
            tissue_expr = expr_matrix[mask, :]
            mean_expr = np.mean(tissue_expr, axis=0)
            top_indices = np.argsort(mean_expr)[-top_n:][::-1]
            
            # Calculate LFC and significance for these genes
            all_other_mask = ~mask
            other_expr = expr_matrix[all_other_mask, :]
            mean_other = np.mean(other_expr, axis=0)
            
            lfc = np.log2((mean_expr + self.epsilon) / (mean_other + self.epsilon))
            
            # Simple t-test for significance
            p_values = []
            for i in top_indices:
                _, p = stats.ttest_ind(tissue_expr[:, i], other_expr[:, i])
                p_values.append(p)
            
            _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
            
            tissue_results[tissue] = {
                'top_genes': [gene_names[i] for i in top_indices],
                'expression_values': mean_expr[top_indices].tolist(),
                'lfc_values': lfc[top_indices].tolist(),
                'p_adjusted': p_adj.tolist(),
                'cell_count': int(np.sum(mask))
            }
        
        return tissue_results
    
    def filter_significant_genes(self, tissue_results: Dict) -> Dict:
        """Filter genes by FDR and LFC thresholds."""
        filtered = {}
        for tissue, data in tissue_results.items():
            significant_indices = [
                i for i, (lfc, p_adj) in enumerate(zip(data['lfc_values'], data['p_adjusted']))
                if lfc > self.lfc_threshold and p_adj < self.fdr_threshold
            ]
            if significant_indices:
                filtered[tissue] = {
                    'top_genes': [data['top_genes'][i] for i in significant_indices],
                    'expression_values': [data['expression_values'][i] for i in significant_indices],
                    'lfc_values': [data['lfc_values'][i] for i in significant_indices],
                    'p_adjusted': [data['p_adjusted'][i] for i in significant_indices],
                    'cell_count': data['cell_count']
                }
        return filtered


class CoexpressionAnalyzer:
    """Analyze gene co-expression networks within tissues."""
    
    def __init__(self, corr_threshold: float = 0.6, fdr_threshold: float = 0.05):
        self.corr_threshold = corr_threshold
        self.fdr_threshold = fdr_threshold
    
    def find_coexpression_pairs(self,
                                expr_matrix: np.ndarray,
                                gene_names: List[str],
                                tissue_labels: np.ndarray,
                                tissue_of_interest: str,
                                max_genes: int = 100) -> List[Dict]:
        """
        Find statistically significant co-expression pairs within a tissue.
        
        Parameters
        ----------
        expr_matrix : np.ndarray
            Expression matrix
        gene_names : list
            List of gene names
        tissue_labels : np.ndarray
            Tissue labels for each cell
        tissue_of_interest : str
            Name of tissue to analyze
        max_genes : int
            Maximum number of genes to consider for computational efficiency
            
        Returns
        -------
        list : Significant co-expression pairs with statistics
        """
        mask = tissue_labels == tissue_of_interest
        if np.sum(mask) < 100:
            return []
        
        tissue_expr = expr_matrix[mask, :]
        n_genes = min(max_genes, tissue_expr.shape[1])
        
        significant_pairs = []
        
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                r, p = stats.pearsonr(tissue_expr[:, i], tissue_expr[:, j])
                if abs(r) > self.corr_threshold and p < 0.05:
                    significant_pairs.append({
                        'gene1': gene_names[i],
                        'gene2': gene_names[j],
                        'correlation': float(r),
                        'p_value': float(p),
                        'tissue': tissue_of_interest,
                        'n_cells': int(np.sum(mask))
                    })
        
        # FDR correction
        if significant_pairs:
            p_vals = [pair['p_value'] for pair in significant_pairs]
            _, p_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
            for idx, pair in enumerate(significant_pairs):
                pair['fdr_corrected_p'] = float(p_corrected[idx])
                pair['significant_fdr'] = p_corrected[idx] < self.fdr_threshold
            
            significant_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return significant_pairs
