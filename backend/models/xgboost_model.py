"""
XGBoost leak detection model with SHAP explainability.

XGBoost is perfect for tabular sensor data:
  - Handles missing values natively
  - Fast training on the ~100k sample dataset
  - Built-in feature importance + SHAP for interpretability.
  
The model does binary classification: is there a leak at this timestep (1) or not (0)?
Since leaks are rare (<30% of timesteps), we use scale_pos_weight and F1-score evaluation.
"""

import numpy as np
import xgboost as xgb
import shap
import json
import pickle
from pathlib import Path
from sklearn.metrics import (
    classification_report, f1_score, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from typing import Dict, Any, Optional, Tuple, List


MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class LeakDetectorXGB:
    """
    XGBoost-based leak detector.
    
    Architecture decision: We use a single global model trained on ALL nodes
    simultaneously (each timestep = one sample with all-node features).
    This lets the model learn cross-node correlations (a leak at node 19
    also affects pressure at neighboring nodes 17, 20, etc.).
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 10.0,
        subsample: float = 0.7,
        colsample_bytree: float = 0.7,
        min_child_weight: int = 10,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "scale_pos_weight": scale_pos_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "eval_metric": "logloss",
            "objective": "binary:logistic",
            "tree_method": "hist",
            "use_label_encoder": False,
        }
        
        self.model = xgb.XGBClassifier(**self.params)
        self.feature_names: Optional[List[str]] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self._threshold: float = 0.4
        self._is_trained: bool = False
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) binary
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels
            feature_names: Human-readable feature names
            
        Returns:
            Dict of training metrics
        """
        self.feature_names = feature_names
        
        # Auto-compute scale_pos_weight from class balance
        n_pos = (y_train > 0).sum()
        n_neg = (y_train == 0).sum()
        if n_pos > 0:
            auto_weight = n_neg / n_pos
            self.model.set_params(scale_pos_weight=auto_weight)
            if verbose:
                print(f"  Class balance: {n_neg} normal / {n_pos} leak")
                print(f"  Auto scale_pos_weight: {auto_weight:.2f}")
        
        # Train with optional early stopping
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = verbose
        
        if verbose:
            print(f"  Training XGBoost ({X_train.shape[0]} samples, {X_train.shape[1]} features)")
        
        self.model.fit(X_train, y_train, **fit_params)
        self._is_trained = True
        
        # Optimize threshold on validation set FIRST
        if X_val is not None and y_val is not None:
            self._optimize_threshold(X_val, y_val)
            if verbose:
                print(f"  Optimized threshold: {self._threshold:.3f}")
        
        # Evaluate AFTER threshold optimization for accurate F1
        train_metrics = self.evaluate(X_train, y_train, prefix="train")
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val, prefix="val")
        
        # Initialize SHAP explainer
        if verbose:
            print("  Initializing SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
        metrics = {**train_metrics, **val_metrics, "threshold": self._threshold}
        if verbose:
            print(f"  Train F1: {train_metrics.get('train_f1', 0):.4f}")
            if val_metrics:
                print(f"  Val F1:   {val_metrics.get('val_f1', 0):.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using optimized threshold."""
        proba = self.predict_proba(X)
        return (proba >= self._threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict leak probability (0-1)."""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(
        self, X: np.ndarray, y: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        f1 = f1_score(y, y_pred, zero_division=0)
        ap = average_precision_score(y, y_proba) if y.sum() > 0 else 0.0
        cm = confusion_matrix(y, y_pred)
        
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        prefix_str = f"{prefix}_" if prefix else ""
        return {
            f"{prefix_str}f1": float(f1),
            f"{prefix_str}precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            f"{prefix_str}recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            f"{prefix_str}ap": float(ap),
            f"{prefix_str}tp": int(tp),
            f"{prefix_str}fp": int(fp),
            f"{prefix_str}fn": int(fn),
            f"{prefix_str}tn": int(tn),
        }
    
    def _optimize_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Find the probability threshold that maximizes F1 on validation set."""
        y_proba = self.predict_proba(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
        
        # Compute F1 for each threshold
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        if best_idx < len(thresholds):
            self._threshold = float(thresholds[best_idx])
        else:
            self._threshold = 0.5
    
    def explain(
        self,
        X: np.ndarray,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions.
        
        This tells the operator: "The model thinks there's a leak because
        pressure at Node_19 dropped 12.3 units below its rolling average,
        AND demand at Node_17 spiked."
        
        Args:
            X: Feature matrix to explain (can be a single sample or batch)
            top_k: Number of top features to return
            
        Returns:
            Dict with shap_values, top_features, and base_value
        """
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)
        
        shap_values = self.explainer.shap_values(X)
        
        # For binary classification, shap_values may be a list [neg, pos]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Get feature importance ranking
        if X.ndim == 1:
            X = X.reshape(1, -1)
            shap_values = shap_values.reshape(1, -1)
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
        
        feature_names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature": feature_names[idx] if idx < len(feature_names) else f"f{idx}",
                "importance": float(mean_abs_shap[idx]),
                "shap_value": float(shap_values[0, idx]) if shap_values.shape[0] == 1 else float(mean_abs_shap[idx]),
            })
        
        return {
            "shap_values": shap_values,
            "top_features": top_features,
            "base_value": float(self.explainer.expected_value) if not isinstance(self.explainer.expected_value, list) else float(self.explainer.expected_value[1]),
        }
    
    def explain_single(self, x: np.ndarray) -> Dict[str, Any]:
        """Explain a single prediction — returns per-feature SHAP breakdown."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        result = self.explain(x, top_k=15)
        proba = float(self.predict_proba(x)[0])
        
        return {
            "probability": proba,
            "prediction": int(proba >= self._threshold),
            "threshold": self._threshold,
            "top_features": result["top_features"],
            "base_value": result["base_value"],
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get XGBoost native feature importance (gain-based)."""
        importance = self.model.feature_importances_
        names = self.feature_names or [f"f{i}" for i in range(len(importance))]
        return dict(sorted(
            zip(names, importance.tolist()),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def save(self, path: Optional[str] = None) -> str:
        """Save model to disk."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = str(MODEL_DIR / "xgboost_leak_detector.pkl")
        
        save_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "threshold": self._threshold,
            "params": self.params,
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        
        print(f"  Model saved to {path}")
        return path
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> "LeakDetectorXGB":
        """Load a trained model from disk."""
        if path is None:
            path = str(MODEL_DIR / "xgboost_leak_detector.pkl")
        
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        
        detector = cls()
        detector.model = save_data["model"]
        detector.feature_names = save_data["feature_names"]
        detector._threshold = save_data["threshold"]
        detector.params = save_data["params"]
        detector._is_trained = True
        detector.explainer = shap.TreeExplainer(detector.model)
        
        return detector


def generate_feature_names(n_nodes: int, include_demand: bool = True) -> List[str]:
    """
    Generate human-readable feature names for the model.
    
    Maps back from column indices to meaningful names like:
      "Node_1_pressure", "Node_5_delta", "Node_19_rolling_mean", etc.
    """
    node_names = [f"Node_{i}" for i in range(1, n_nodes + 1)]
    
    groups = [
        ("pressure", node_names),
        ("delta", node_names),
        ("roll_mean", node_names),
        ("roll_std", node_names),
        ("residual", node_names),
    ]
    
    if include_demand:
        groups.append(("demand", node_names))
    
    feature_names = []
    for suffix, nodes in groups:
        for node in nodes:
            feature_names.append(f"{node}_{suffix}")
    
    return feature_names
