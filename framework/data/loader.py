"""Data loading utilities for GTEx and TCGA datasets."""

import anndata
import s3fs
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class GTExLoader:
    """Loader for GTEx v9 single-nucleus RNA-seq data."""
    
    def __init__(self, s3_path: str, use_anon: bool = True):
        """
        Initialize GTEx data loader.
        
        Parameters:
        -----------
        s3_path : str
            S3 path to the .h5ad file
        use_anon : bool
            Use anonymous S3 access
        """
        self.s3_path = s3_path
        self.fs = s3fs.S3FileSystem(anon=use_anon)
        self.adata = None
        
    def load(self) -> anndata.AnnData:
        """Load the AnnData object from S3."""
        with self.fs.open(self.s3_path, 'rb') as f:
            self.adata = anndata.read_h5ad(f)
        print(f"Loaded GTEx data: {self.adata.shape[0]} cells, {self.adata.shape[1]} genes")
        return self.adata
    
    def get_hvg_indices(self, n_genes: int = 500) -> Tuple[np.ndarray, list]:
        """Get indices and names of highly variable genes."""
        hvg_mask = self.adata.var['hvg'].values
        hvg_indices = np.where(hvg_mask)[0][:n_genes]
        hvg_names = self.adata.var['feature_name'].iloc[hvg_indices].tolist()
        return hvg_indices, hvg_names


class TCGALoader:
    """Loader for TCGA-STAD expression data."""
    
    def __init__(self, file_path: str):
        """
        Initialize TCGA data loader.
        
        Parameters:
        -----------
        file_path : str
            Path to the TCGA-STAD star_counts.tsv file
        """
        self.file_path = Path(file_path)
        self.data = None
        self.labels = None
        
    def load(self) -> pd.DataFrame:
        """Load TCGA expression data."""
        self.data = pd.read_csv(self.file_path, sep='\t', index_col=0)
        print(f"Loaded TCGA data: {self.data.shape[0]} genes, {self.data.shape[1]} samples")
        return self.data
    
    def get_labels(self) -> pd.Series:
        """Extract tumor/normal labels from sample IDs."""
        sample_types = self.data.columns.str.split('-').str[-1]
        keep = sample_types.isin(['01A', '11A'])
        labels = (sample_types[keep] == '01A').astype(int)
        return labels
