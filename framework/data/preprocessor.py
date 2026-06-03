"""Data preprocessing utilities."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from typing import Tuple, Optional


class ExpressionPreprocessor:
    """Preprocessor for expression data."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.var_threshold = VarianceThreshold(threshold=0.5)
        self.kbest = None
        
    def preprocess_gtex(self, expr_matrix: np.ndarray, n_cells: int = 10000,
                        random_sample: bool = True) -> np.ndarray:
        """
        Preprocess GTEx expression data.
        
        Parameters:
        -----------
        expr_matrix : np.ndarray
            Expression matrix (cells x genes)
        n_cells : int
            Number of cells to sample
        random_sample : bool
            Whether to randomly sample cells
            
        Returns:
        --------
        np.ndarray: Preprocessed expression matrix
        """
        if random_sample and expr_matrix.shape[0] > n_cells:
            indices = np.random.choice(expr_matrix.shape[0], n_cells, replace=False)
            expr_matrix = expr_matrix[indices, :]
        
        # Scale data
        expr_scaled = self.scaler.fit_transform(expr_matrix)
        return expr_scaled
    
    def preprocess_tcga(self, X: pd.DataFrame, y: pd.Series,
                        n_features: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess TCGA data with feature selection.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Expression data (samples x genes)
        y : pd.Series
            Labels (0: normal, 1: tumor)
        n_features : int
            Number of features to select
            
        Returns:
        --------
        Tuple of (X_selected, y, selected_feature_names)
        """
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Remove low-variance genes
        X_var = self.var_threshold.fit_transform(X)
        
        # Select top features with ANOVA
        self.kbest = SelectKBest(f_classif, k=n_features)
        X_sel = self.kbest.fit_transform(X_var, y)
        
        # Get selected feature names
        feature_mask = self.kbest.get_support()
        selected_features = X.columns[self.var_threshold.get_support()][feature_mask]
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_sel)
        
        return X_scaled, y.values, selected_features
    
    def get_donor_level_data(self, adata, hvg_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate single-cell data to donor level."""
        donor_sex = adata.obs[['donor_id', 'sex']].drop_duplicates()
        donor_expr = []
        donor_list = []
        
        for donor in donor_sex['donor_id'].unique():
            mask = adata.obs['donor_id'] == donor
            donor_expr_mean = adata[mask, hvg_indices].layers['normalized'].mean(axis=0)
            if isinstance(donor_expr_mean, np.matrix):
                donor_expr_mean = np.array(donor_expr_mean).flatten()
            donor_expr.append(donor_expr_mean)
            donor_list.append(donor)
        
        X_donor = np.vstack(donor_expr)
        y_donor = donor_sex.set_index('donor_id').loc[donor_list]['sex'].values
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_donor = le.fit_transform(y_donor)
        
        return X_donor, y_donor
