"""Data loading module for single-cell transcriptomic datasets."""

import anndata
import s3fs
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TranscriptomicDataLoader:
    """
    A reusable data loader for single-cell/nucleus transcriptomic data.
    Supports GTEx v9 from S3 and local files (e.g., TCGA).
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def load_gtex_v9(self, s3_path: str, n_cells: Optional[int] = None) -> anndata.AnnData:
        """
        Load GTEx v9 data from public S3 bucket.
        
        Parameters
        ----------
        s3_path : str
            S3 path to the .h5ad file
        n_cells : int, optional
            Number of cells to subsample (for computational efficiency)
            
        Returns
        -------
        anndata.AnnData
            Loaded and optionally subsampled AnnData object
        """
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open(s3_path, 'rb') as f:
            adata = anndata.read_h5ad(f)
        
        if n_cells and n_cells < adata.n_obs:
            indices = np.random.choice(adata.n_obs, n_cells, replace=False)
            adata = adata[indices, :]
        
        print(f"Loaded GTEx data: {adata.n_obs} cells × {adata.n_vars} genes")
        return adata
    
    def load_tcga_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """
        Load TCGA-STAD expression data.
        
        Parameters
        ----------
        filepath : str
            Path to the STAR counts TSV file
            
        Returns
        -------
        tuple: (X, y, gene_names)
            X: expression matrix (samples × genes)
            y: binary labels (1=tumor, 0=normal)
            gene_names: gene identifiers
        """
        counts = pd.read_csv(filepath, sep='\t', index_col=0)
        sample_type = counts.columns.str.split('-').str[-1]
        keep = sample_type.isin(['01A', '11A'])
        
        X = counts.loc[:, keep].T
        y = (sample_type[keep] == '01A').astype(int)
        
        # Fill missing values
        X = X.fillna(X.mean())
        
        print(f"Loaded TCGA data: {len(y)} samples, {X.shape[1]} genes")
        print(f"  Tumor: {sum(y)}, Normal: {len(y)-sum(y)}")
        
        return X.values, y.values, X.columns


class DataPreprocessor:
    """Preprocessing utilities for transcriptomic data."""
    
    @staticmethod
    def get_hvg_indices(adata: anndata.AnnData, n_hvg: int = 500) -> Tuple[np.ndarray, list]:
        """Get indices and names of highly variable genes."""
        hvg_mask = adata.var['hvg'].values if 'hvg' in adata.var.columns else np.ones(adata.n_vars, dtype=bool)
        hvg_indices = np.where(hvg_mask)[0][:n_hvg]
        hvg_names = adata.var_names[hvg_indices].tolist()
        return hvg_indices, hvg_names
    
    @staticmethod
    def extract_expression_matrix(adata: anndata.AnnData, 
                                  cell_indices: np.ndarray,
                                  gene_indices: np.ndarray,
                                  layer: str = 'normalized') -> np.ndarray:
        """Extract expression matrix from AnnData object."""
        expr = adata[cell_indices, gene_indices].layers[layer]
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        return expr
    
    @staticmethod
    def get_tissue_labels(adata: anndata.AnnData, cell_indices: np.ndarray) -> np.ndarray:
        """Extract tissue labels for selected cells."""
        return adata.obs['tissue'].iloc[cell_indices].values
