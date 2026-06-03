"""Co-expression network analysis."""

import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from typing import List, Dict, Tuple


class CoexpressionAnalyzer:
    """Analyze gene co-expression networks."""
    
    def __init__(self, expr_matrix: np.ndarray, gene_names: List[str],
                 tissue_labels: np.ndarray):
        """
        Initialize co-expression analyzer.
        
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
        
    def find_coexpression_networks(self, corr_threshold: float = 0.65,
                                   max_genes: int = 100) -> Dict:
        """
        Find co-expressed gene pairs in each tissue.
        
        Parameters:
        -----------
        corr_threshold : float
            Minimum absolute correlation for significant pairs
        max_genes : int
            Maximum number of genes to consider per tissue
            
        Returns:
        --------
        dict: Tissue-specific co-expression networks
        """
        tissues = np.unique(self.tissue_labels)
        tissue_networks = {}
        
        for tissue in tissues:
            tissue_mask = self.tissue_labels == tissue
            if np.sum(tissue_mask) < 500:
                continue
                
            tissue_expr = self.expr_matrix[tissue_mask, :]
            corr_matrix = np.corrcoef(tissue_expr.T)
            
            n_genes = min(max_genes, corr_matrix.shape[0])
            strong_pairs = []
            
            for i in range(n_genes):
                for j in range(i + 1, n_genes):
                    corr_val = corr_matrix[i, j]
                    if abs(corr_val) > corr_threshold:
                        strong_pairs.append({
                            'gene1': self.gene_names[i],
                            'gene2': self.gene_names[j],
                            'correlation': float(corr_val),
                            'tissue': tissue
                        })
            
            strong_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            tissue_networks[tissue] = {
                'strong_pairs': strong_pairs[:20],
                'total_pairs': int(len(strong_pairs)),
                'cell_count': int(np.sum(tissue_mask))
            }
            
        return tissue_networks
    
    def calculate_statistical_pairs(self, tissue_of_interest: str,
                                    corr_threshold: float = 0.6,
                                    max_genes: int = 100) -> List[Dict]:
        """
        Calculate statistically validated co-expression pairs.
        
        Parameters:
        -----------
        tissue_of_interest : str
            Tissue to analyze
        corr_threshold : float
            Minimum correlation threshold
        max_genes : int
            Maximum number of genes to consider
            
        Returns:
        --------
        list: Statistically significant co-expression pairs
        """
        tissue_mask = self.tissue_labels == tissue_of_interest
        if np.sum(tissue_mask) < 100:
            return []
        
        tissue_expr = self.expr_matrix[tissue_mask, :]
        n_cells = tissue_expr.shape[0]
        n_genes = min(max_genes, tissue_expr.shape[1])
        significant_pairs = []
        
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                gene1_expr = tissue_expr[:, i]
                gene2_expr = tissue_expr[:, j]
                r_value, p_value = stats.pearsonr(gene1_expr, gene2_expr)
                
                if abs(r_value) > corr_threshold and p_value < 0.05:
                    significant_pairs.append({
                        'gene1': self.gene_names[i],
                        'gene2': self.gene_names[j],
                        'correlation': float(r_value),
                        'p_value': float(p_value),
                        'tissue': tissue_of_interest,
                        'n_cells': n_cells
                    })
        
        # Apply FDR correction
        if significant_pairs:
            p_values = [pair['p_value'] for pair in significant_pairs]
            reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
            
            for idx, pair in enumerate(significant_pairs):
                pair['fdr_corrected_p'] = float(pvals_corrected[idx])
                pair['significant_fdr'] = bool(reject[idx])
        
        significant_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        return significant_pairs
