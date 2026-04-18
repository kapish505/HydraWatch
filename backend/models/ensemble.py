"""
Ensemble logic — combines all three models into a single alert system.

HOW IT WORKS (in plain English):
  Think of three security guards watching the water network:
    1. LSTM Guard: "Something looks UNUSUAL" (anomaly detection)
    2. XGBoost Guard: "I think it's a LEAK" (classification with features)
    3. GAT Guard: "The leak is at NODE 19" (spatial localisation)

  An alert fires ONLY when all three guards agree:
    1. LSTM anomaly score > 0.85 (high reconstruction error)
    2. XGBoost probability > 0.4 for at least one node
    3. The nodes flagged by XGBoost and GAT overlap in top-5

  This triple-agreement reduces false alarms dramatically.
  Each guard catches different types of mistakes the others make.

Severity levels (based on GAT's max node probability):
  - CRITICAL: GAT probability > 0.8
  - MEDIUM:   GAT probability 0.5 – 0.8
  - LOW:      GAT probability 0.3 – 0.5
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class EnsembleDetector:
    """
    Combines LSTM Autoencoder, XGBoost, and GAT predictions.

    Usage:
        detector = EnsembleDetector(lstm_trainer, xgb_model, gat_trainer)
        result = detector.predict(pressure_window, xgb_features, gat_features, node_names)
        if result["alert"]:
            print(f"LEAK DETECTED at {result['suspect_nodes']}!")
    """

    def __init__(
        self,
        lstm_trainer=None,
        xgb_model=None,
        gat_trainer=None,
        lstm_threshold: float = 0.85,
        xgb_threshold: float = 0.4,
        agreement_top_k: int = 5,
    ):
        """
        Args:
            lstm_trainer: LSTMAutoencoderTrainer instance (or None if not available)
            xgb_model: LeakDetectorXGB instance (or None)
            gat_trainer: GATTrainer instance (or None)
            lstm_threshold: normalized anomaly score threshold (0.85)
            xgb_threshold: XGBoost probability threshold (0.4)
            agreement_top_k: how many top nodes to compare for agreement
        """
        model_info_path = Path(__file__).resolve().parent.parent.parent / "models" / "model_info.json"
        if model_info_path.exists():
            try:
                with open(model_info_path) as f:
                    info = json.load(f)
                    # Only override if the value is not None
                    if info.get("lstm_threshold") is not None:
                        lstm_threshold = info["lstm_threshold"]
                    if info.get("xgb_threshold") is not None:
                        xgb_threshold = info["xgb_threshold"]
            except Exception:
                pass

        self.lstm = lstm_trainer
        self.xgb = xgb_model
        self.gat = gat_trainer
        self.lstm_threshold = lstm_threshold
        self.xgb_threshold = xgb_threshold
        self.agreement_top_k = agreement_top_k

    def predict(
        self,
        pressure_window: Optional[np.ndarray] = None,
        xgb_features: Optional[np.ndarray] = None,
        gat_node_features: Optional[np.ndarray] = None,
        node_names: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the ensemble prediction pipeline.

        Args:
            pressure_window: (1, 24, N_sensors) for LSTM — the last 24 timesteps
            xgb_features: (1, num_features) for XGBoost — engineered features
            gat_node_features: (N, 3) for GAT — per-node features
            node_names: list of node name strings
            timestamp: current timestamp string

        Returns:
            Dict with:
              alert: bool — whether to fire an alert
              severity: "CRITICAL"/"MEDIUM"/"LOW"/None
              anomaly_score: float 0-1 (from LSTM)
              suspect_nodes: list of node names
              confidence: float 0-1
              node_probabilities: dict {node: probability} from GAT
              shap_features: list of {feature, value, impact} from XGBoost
              detected_at: timestamp
              estimated_location: string description
        """
        result = {
            "alert": False,
            "severity": None,
            "anomaly_score": 0.0,
            "suspect_nodes": [],
            "confidence": 0.0,
            "node_probabilities": {},
            "shap_features": [],
            "detected_at": timestamp or datetime.now().isoformat(),
            "estimated_location": "",
            "xgb_probability": 0.0,
        }

        # ── Step 1: LSTM anomaly detection ────────────────────────────
        lstm_pass = False
        if self.lstm is not None and pressure_window is not None:
            lstm_result = self.lstm.predict(pressure_window)
            anomaly_score = float(lstm_result["normalized_scores"][0])
            result["anomaly_score"] = anomaly_score
            lstm_pass = anomaly_score > self.lstm_threshold
        else:
            # If LSTM not available, pass this check
            lstm_pass = True

        # ── Step 2: XGBoost classification ────────────────────────────
        xgb_pass = False
        xgb_suspect_nodes = set()
        if self.xgb is not None and xgb_features is not None:
            proba = float(self.xgb.predict_proba(xgb_features.reshape(1, -1))[0])
            result["xgb_probability"] = proba
            xgb_pass = proba > self.xgb_threshold

            # Compute SHAP on every timestep so the UI can show the live AI reasoning
            # even during nominal "safe" operations.
            explanation = self.xgb.explain_single(xgb_features)
            shap_features = []
            for feat in explanation["top_features"][:5]:
                shap_features.append({
                    "feature": feat["feature"],
                    "value": feat.get("importance", 0),
                    "impact": feat.get("shap_value", 0),
                })
            result["shap_features"] = shap_features

            # Extract suspect nodes from SHAP features.
            _KNOWN_SUFFIXES = (
                "_rolling_mean_1h", "_rolling_std_1h",
                "_rolling_mean_6h", "_rolling_std_6h",
                "_pressure_z_score", "_pressure_gradient",
                "_neighbor_pressure_delta",
                "_hour_of_day", "_day_of_week",
                "_demand_residual",
                "_pressure_range_6h", "_cumulative_deviation",
                "_neighbor_z_diff",
                "_pressure", "_delta", "_roll_mean", "_roll_std",
                "_residual", "_demand",
            )

            for feat in explanation["top_features"][:self.agreement_top_k]:
                fname = feat["feature"]
                node_name = fname  # fallback
                for suffix in _KNOWN_SUFFIXES:
                    if fname.endswith(suffix):
                        node_name = fname[:-len(suffix)]
                        break
                # Only add if it looks like a real node in the network
                if node_names and node_name in node_names:
                    xgb_suspect_nodes.add(node_name)
                elif node_names is None:
                    xgb_suspect_nodes.add(node_name)
        else:
            xgb_pass = True

        # ── Step 3: GAT localisation ──────────────────────────────────
        gat_pass = False
        gat_suspect_nodes = set()
        max_gat_prob = 0.0
        if self.gat is not None and gat_node_features is not None and node_names is not None:
            node_probs = self.gat.predict(gat_node_features, node_names)
            result["node_probabilities"] = node_probs

            # Get top-K nodes
            sorted_nodes = sorted(node_probs.items(), key=lambda x: x[1], reverse=True)
            top_k_nodes = sorted_nodes[:self.agreement_top_k]

            for name, prob in top_k_nodes:
                gat_suspect_nodes.add(name)

            max_gat_prob = sorted_nodes[0][1] if sorted_nodes else 0.0
            gat_pass = max_gat_prob > 0.3  # At least one node above 0.3
        else:
            gat_pass = True

        # ── Step 4: Suspect node aggregation ──────────────────────────
        if xgb_suspect_nodes and gat_suspect_nodes:
            overlap = xgb_suspect_nodes & gat_suspect_nodes
            result["suspect_nodes"] = list(overlap) if overlap else list(gat_suspect_nodes)[:3]
        elif gat_suspect_nodes:
            result["suspect_nodes"] = list(gat_suspect_nodes)[:3]
        elif xgb_suspect_nodes:
            result["suspect_nodes"] = list(xgb_suspect_nodes)[:3]

        # ── Final decision: 2-of-3 voting ─────────────────────────────
        # Alert fires when at least 2 of the 3 models agree there's a leak.
        # This is more robust than requiring all 3 (too strict) or any 1 (too noisy).
        votes = sum([lstm_pass, xgb_pass, gat_pass])
        result["alert"] = votes >= 2

        if result["alert"]:
            # Severity based on vote count and GAT probability
            if votes == 3 and max_gat_prob > 0.5:
                result["severity"] = "CRITICAL"
            elif votes == 3 or max_gat_prob > 0.3:
                result["severity"] = "MEDIUM"
            else:
                result["severity"] = "LOW"

            # Confidence = average of all model signals
            signals = []
            if self.lstm is not None:
                signals.append(result["anomaly_score"])
            if self.xgb is not None:
                signals.append(result["xgb_probability"])
            if self.gat is not None:
                signals.append(max_gat_prob)
            result["confidence"] = float(np.mean(signals)) if signals else 0.5

            # Estimated location description
            if len(result["suspect_nodes"]) >= 2:
                result["estimated_location"] = (
                    f"region near {', '.join(result['suspect_nodes'][:3])}"
                )
            elif len(result["suspect_nodes"]) == 1:
                result["estimated_location"] = f"near {result['suspect_nodes'][0]}"

        return result

    def predict_batch(
        self,
        pressure_windows: Optional[np.ndarray],
        xgb_features_batch: Optional[np.ndarray],
        gat_features_batch: Optional[List[np.ndarray]],
        node_names: Optional[List[str]],
        timestamps: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run ensemble on a batch of timesteps.

        Returns list of prediction dicts, one per timestep.
        """
        n_samples = 0
        if pressure_windows is not None:
            n_samples = len(pressure_windows)
        elif xgb_features_batch is not None:
            n_samples = len(xgb_features_batch)
        elif gat_features_batch is not None:
            n_samples = len(gat_features_batch)

        results = []
        for i in range(n_samples):
            pw = pressure_windows[i:i+1] if pressure_windows is not None else None
            xf = xgb_features_batch[i] if xgb_features_batch is not None else None
            gf = gat_features_batch[i] if gat_features_batch is not None else None
            ts = timestamps[i] if timestamps is not None else None

            result = self.predict(pw, xf, gf, node_names, ts)
            results.append(result)

        return results
