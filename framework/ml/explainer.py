"""SHAP explainability module."""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional


class SHAPExplainer:
    """SHAP-based model explainer."""
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize SHAP explainer.
        
        Parameters:
        -----------
        model : trained model
            Model to explain (must be tree-based)
        feature_names : list
            Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        
    def explain(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values."""
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get mean absolute SHAP values per feature."""
        shap_mean_abs = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'gene': self.feature_names,
            'shap_importance': shap_mean_abs
        })
        return importance_df.sort_values('shap_importance', ascending=False)
    
    def plot_summary(self, X: np.ndarray, save_path: Optional[str] = None):
        """Create SHAP summary plot."""
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Get top N features by importance."""
        importance_df = self.get_feature_importance()
        return importance_df.head(n)
