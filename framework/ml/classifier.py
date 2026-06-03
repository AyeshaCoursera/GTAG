"""Machine learning classifiers for transcriptomic data."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Tuple, Dict, Any


class TabTransformer(nn.Module):
    """TabTransformer for tabular data classification."""
    
    def __init__(self, input_dim: int, num_classes: int = 2,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.dropout(x)
        return self.fc_out(x)


class XGBoostWrapper:
    """Wrapper for XGBoost classifier."""
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.05, random_state: int = 42):
        self.model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_state,
            eval_metric='logloss', use_label_encoder=False
        )
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        self.model.fit(X_train, y_train, **kwargs)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_


class LASSOWrapper:
    """Wrapper for LASSO logistic regression."""
    
    def __init__(self, max_iter: int = 1000, cv_folds: int = 5,
                 random_state: int = 42):
        self.model = LogisticRegressionCV(
            penalty='l1', solver='saga', cv=cv_folds, random_state=random_state,
            max_iter=max_iter, class_weight='balanced', n_jobs=1
        )
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
