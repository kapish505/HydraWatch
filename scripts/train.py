"""
Complete training pipeline for HydraWatch — all three ML models.

This script does everything:
  1. Loads ALL 500 LeakDB Hanoi scenarios
  2. Builds engineered features (10 per node)
  3. Trains XGBoost classifier with SHAP explanations
  4. Trains LSTM Autoencoder on normal-only data
  5. Trains GAT for per-node leak localisation
  6. Exports all models (pickle + ONNX)
  7. Saves metrics + comparison to published baselines

Usage:
  python scripts/train.py                    # Train all 3 models
  python scripts/train.py --no-gat           # Skip GAT (faster iteration)
  python scripts/train.py --no-lstm          # Skip LSTM
  python scripts/train.py --scenarios 50     # Use fewer scenarios (faster)

Build order matters:
  1. XGBoost first (fastest, gives immediate metrics)
  2. LSTM second (needs normal-only windows)
  3. GAT third (needs node-level labels + network graph)
"""

import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.loader import (
    find_leakdb_hanoi_dir,
    load_leakdb_scenario,
    load_all_scenarios,
    build_pressure_matrix,
    get_node_names,
    get_hanoi_inp_path,
)
from backend.data.features import (
    build_xgboost_features,
    build_xgboost_dataset,
    build_lstm_windows,
    build_gat_node_features,
    get_node_leak_labels,
)
from backend.models.xgboost_model import LeakDetectorXGB
from backend.network import load_network, get_edge_index_and_features, get_junction_names, get_adjacency


MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _get_existing_lstm_threshold():
    """Retrieve the LSTM threshold from a previous training run's model_info.json."""
    info_path = MODELS_DIR / "model_info.json"
    if info_path.exists():
        try:
            with open(info_path) as f:
                info = json.load(f)
            return info.get("lstm_threshold")
        except Exception:
            pass
    return None


def train_xgboost(scenarios, adjacency, node_names, train_ids, val_ids, verbose=True):
    """
    Train XGBoost leak detector with engineered features.

    This is the "workhorse" model — fast to train, easy to interpret,
    and serves as the primary classification signal.
    """
    print("\n" + "=" * 60)
    print("MODEL 1: XGBoost Leak Detector")
    print("=" * 60)

    t0 = time.time()

    # Build full dataset with 13-feature engineering
    print("\n  Building feature matrix (13 features × 31 nodes = 403 features)...")
    X, y, scenario_ids, feature_names = build_xgboost_dataset(
        scenarios, adjacency, verbose=verbose
    )

    # Split by scenario ID
    train_mask = np.isin(scenario_ids, list(train_ids))
    val_mask = np.isin(scenario_ids, list(val_ids))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"\n  Train split: {X_train.shape[0]} samples ({(y_train > 0).sum()} leaks)")
    print(f"  Val split:   {X_val.shape[0]} samples ({(y_val > 0).sum()} leaks)")

    # Train using the parameters explicitly tuned inside the XGBoost class
    detector = LeakDetectorXGB()

    metrics = detector.train(
        X_train, y_train,
        X_val, y_val,
        feature_names=feature_names,
        verbose=verbose,
    )

    # Save
    detector.save()

    # Metrics
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            clean_metrics[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean_metrics[k] = v.tolist()
        else:
            clean_metrics[k] = v

    with open(OUTPUTS_DIR / "xgboost_metrics.json", "w") as f:
        json.dump(clean_metrics, f, indent=2)

    # Feature importance
    importance = detector.get_feature_importance()
    top_10 = list(importance.items())[:10]
    print("\n  Top 10 features:")
    for name, score in top_10:
        print(f"    {name}: {score:.4f}")

    # SHAP sample
    print("\n  Computing SHAP explanations on sample...")
    sample_idx = np.where(y_val > 0)[0][:1]
    if len(sample_idx) > 0:
        explanation = detector.explain_single(X_val[sample_idx[0]])
        print(f"  Sample prediction: {explanation['probability']:.3f}")
        print(f"  Top SHAP features:")
        for feat in explanation["top_features"][:5]:
            print(f"    {feat['feature']}: SHAP={feat['shap_value']:.4f}")

    elapsed = time.time() - t0
    print(f"\n  XGBoost training completed in {elapsed:.1f}s")

    return detector, clean_metrics


def train_lstm(scenarios, train_ids, val_ids, verbose=True):
    """
    Train LSTM Autoencoder on normal-only pressure windows.

    The key insight: we train ONLY on data where no leak is happening.
    The model learns what "normal" looks like. At inference time, if it
    can't reconstruct the input well, that input must be abnormal.
    """
    print("\n" + "=" * 60)
    print("MODEL 2: LSTM Autoencoder (Anomaly Detection)")
    print("=" * 60)

    from backend.models.lstm_ae import LSTMAutoencoderTrainer

    t0 = time.time()

    # Build normal-only windows from training scenarios
    print("\n  Building sliding windows (24 timesteps each)...")

    train_windows_list = []
    val_windows_list = []
    # Also build windows WITH leak data for validation threshold testing
    val_all_windows = []
    val_all_labels = []

    # To prevent Out-Of-Memory (OOM) errors, we don't need all 400 scenarios for LSTM.
    # Learning "normal" base pressure dynamics works perfectly fine on a subset.
    max_train_scenarios = 50
    max_val_scenarios = 10
    train_used = 0
    val_used = 0
    
    print(f"  Note: Limiting LSTM training to {max_train_scenarios} scenarios to prevent RAM exhaustion.")

    for sc in scenarios:
        pressures = build_pressure_matrix(sc)
        labels = sc["labels"][:len(pressures)]
        sid = sc["scenario_id"]

        if sid in train_ids:
            if train_used < max_train_scenarios:
                # Normal-only windows for training
                windows, _ = build_lstm_windows(pressures, labels, window_size=24, normal_only=True)
                if len(windows) > 0:
                    train_windows_list.append(windows)
                train_used += 1

        elif sid in val_ids:
            if val_used < max_val_scenarios:
                # Normal-only windows for validation loss
                normal_windows, _ = build_lstm_windows(pressures, labels, window_size=24, normal_only=True)
                if len(normal_windows) > 0:
                    val_windows_list.append(normal_windows)

                # ALL windows for threshold calibration
                all_windows, all_labels = build_lstm_windows(pressures, labels, window_size=24, normal_only=False)
                if len(all_windows) > 0:
                    val_all_windows.append(all_windows)
                    val_all_labels.append(all_labels)
                val_used += 1
                
        if train_used >= max_train_scenarios and val_used >= max_val_scenarios:
            break

    train_windows = np.concatenate(train_windows_list, axis=0) if train_windows_list else np.array([])
    val_windows = np.concatenate(val_windows_list, axis=0) if val_windows_list else np.array([])

    if len(train_windows) == 0:
        print("  ✗ No training windows! Cannot train LSTM.")
        return None

    n_sensors = train_windows.shape[2]
    print(f"  Train windows (normal only): {len(train_windows)}")
    print(f"  Val windows (normal only):   {len(val_windows)}")
    print(f"  Sensors per window: {n_sensors}")

    # If val_windows is empty, use a portion of training windows
    if len(val_windows) == 0:
        split = int(len(train_windows) * 0.8)
        val_windows = train_windows[split:]
        train_windows = train_windows[:split]
        print(f"  (Using time-based split: {len(train_windows)} train / {len(val_windows)} val)")

    # Train
    trainer = LSTMAutoencoderTrainer(
        n_sensors=n_sensors,
        hidden_size=64,
        bottleneck_size=8,
        lr=1e-3,
    )

    if val_all_windows:
        all_w = np.concatenate(val_all_windows, axis=0)
        all_l = np.concatenate(val_all_labels, axis=0)
    else:
        all_w, all_l = None, None

    metrics = trainer.train(
        train_windows, val_windows,
        epochs=50,
        batch_size=64,
        verbose=verbose,
        val_all_windows=all_w,
        val_all_labels=all_l,
    )

    # Save
    trainer.save()

    # Test anomaly detection on leak windows
    if val_all_windows:
        print("\n  Testing anomaly detection on validation data...")
        result = trainer.predict(all_w)
        scores = result["anomaly_scores"]
        is_anomaly = result["is_anomaly"]

        # Compute metrics
        tp = ((is_anomaly) & (all_l > 0)).sum()
        fp = ((is_anomaly) & (all_l == 0)).sum()
        fn = ((~is_anomaly) & (all_l > 0)).sum()
        tn = ((~is_anomaly) & (all_l == 0)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        print(f"  Anomaly Detection Results:")
        print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"    Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        metrics["detection_f1"] = float(f1)
        metrics["detection_precision"] = float(precision)
        metrics["detection_recall"] = float(recall)

    with open(OUTPUTS_DIR / "lstm_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "history"}, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  LSTM training completed in {elapsed:.1f}s")

    return trainer


def train_gat(scenarios, hanoi_inp, node_names, train_ids, val_ids, verbose=True):
    """
    Train GAT for per-node leak localisation.

    The GAT needs:
    - Node features: (N, 3) per timestep — current_pressure, rolling_mean_6h, z_score
    - Node labels: (N,) per timestep — 1.0 at the leaking node, 0.0 elsewhere
    - Edge index: fixed network topology from WNTR
    """
    print("\n" + "=" * 60)
    print("MODEL 3: GAT Leak Localiser")
    print("=" * 60)

    from backend.models.gat import GATTrainer

    t0 = time.time()

    # Load network graph
    print("\n  Loading network graph for edge connectivity...")
    wn = load_network(hanoi_inp)
    junction_names = get_junction_names(wn)
    edge_index, edge_attr = get_edge_index_and_features(wn, junction_names)

    print(f"  Junctions: {len(junction_names)}")
    print(f"  Edges: {edge_index.shape[1]}")

    # Build per-timestep node features and labels
    print("\n  Building per-timestep node features and labels...")

    train_feats, train_labels = [], []
    val_feats, val_labels = [], []

    # Limit GAT scenarios to avoid OOM and massively speed up training
    # 100 scenarios gives ~500k training timesteps
    max_train_scenarios = 100
    max_val_scenarios = 20
    train_used = 0
    val_used = 0
    
    print(f"  Note: Limiting GAT training to {max_train_scenarios} scenarios for speed/RAM.")

    for sc in scenarios:
        pressures = build_pressure_matrix(sc)
        labels = sc["labels"][:len(pressures)]
        node_labels = get_node_leak_labels(sc, node_names)  # (T, N)
        sid = sc["scenario_id"]

        T = len(labels)

        # Sample timesteps: always include leak steps, subsample normals every 6th step.
        # Hard cap total samples to keep epochs fast (~200k → ~3k batches at bs=64)
        max_train_samples = 200_000
        max_val_samples = 40_000

        for t in range(T):
            if labels[t] > 0 or t % 6 == 0:
                if sid in train_ids and train_used < max_train_scenarios and len(train_feats) < max_train_samples:
                    feat = build_gat_node_features(pressures, t, window_6h=12)  # (N, 13)
                    train_feats.append(feat)
                    train_labels.append(node_labels[t])
                elif sid in val_ids and val_used < max_val_scenarios and len(val_feats) < max_val_samples:
                    feat = build_gat_node_features(pressures, t, window_6h=12)  # (N, 13)
                    val_feats.append(feat)
                    val_labels.append(node_labels[t])

        if sid in train_ids:
            train_used += 1
        elif sid in val_ids:
            val_used += 1
            
        if train_used >= max_train_scenarios and val_used >= max_val_scenarios:
            break

    print(f"  Train samples: {len(train_feats)} timesteps")
    print(f"  Val samples:   {len(val_feats)} timesteps")

    if len(train_feats) == 0:
        print("  ✗ No training data! Cannot train GAT.")
        return None

    # Create trainer — empirically 2e-5 is the sweet spot for this network size
    # (anything higher forces the model to destroy its Loc.Acc to chase binary F1)
    trainer = GATTrainer(
        edge_index=edge_index,
        n_node_features=13,
        hidden_channels=64,
        heads=4,
        lr=2e-5,
    )

    print(f"\n  Training on device: {trainer.device}")
    params = sum(p.numel() for p in trainer.model.parameters())
    print(f"  Model parameters: {params:,}")

    # Store a representative sample from training data for ONNX export
    trainer.store_reference_sample(train_feats)

    # Pure cosine annealing — no warmup. Start at 5e-5, decay to 1e-6.
    import torch.optim as optim
    total_epochs = 150
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=total_epochs, eta_min=1e-6
    )

    # Training loop — track best by localisation accuracy (our actual target)
    best_f1 = 0.0
    best_loc_acc = 0.0
    patience_counter = 0
    max_patience = 30

    for epoch in range(1, total_epochs + 1):
        cosine_scheduler.step()

        loss = trainer.train_epoch(train_feats, train_labels, batch_size=64)

        if epoch % 3 == 0 or epoch == 1:
            val_metrics = trainer.evaluate(val_feats, val_labels)
            f1 = val_metrics["f1"]
            loc_acc = val_metrics["localisation_accuracy"]

            if verbose:
                lr_now = trainer.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {f1:.4f} | "
                      f"P: {val_metrics['precision']:.3f} R: {val_metrics['recall']:.3f} | "
                      f"Loc.Acc: {loc_acc:.3f} | LR: {lr_now:.2e}")

            # Save on best localisation accuracy (our primary target)
            if loc_acc > best_loc_acc:
                best_loc_acc = loc_acc
                best_f1 = f1
                trainer.save(skip_onnx=True)
                patience_counter = 0
            else:
                patience_counter += 3

            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n  Best val F1: {best_f1:.4f}")
    print(f"  Best Loc.Acc: {best_loc_acc:.3f}")

    # Final save with ONNX export attempt
    trainer.save()

    # Save metrics
    gat_metrics = {
        "best_f1": best_f1,
        "final_localisation_accuracy": best_loc_acc,
    }
    with open(OUTPUTS_DIR / "gat_metrics.json", "w") as f:
        json.dump(gat_metrics, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  GAT training completed in {elapsed:.1f}s")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train all HydraWatch models")
    parser.add_argument("--no-xgboost", action="store_true", help="Skip XGBoost training")
    parser.add_argument("--no-gat", action="store_true", help="Skip GAT training")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM training")
    parser.add_argument("--scenarios", type=int, default=0,
                        help="Number of scenarios to use (0 = all)")
    args = parser.parse_args()

    print("=" * 60)
    print("HydraWatch Complete Training Pipeline")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    print("\n  Loading LeakDB data...")
    hanoi_dir = find_leakdb_hanoi_dir()
    hanoi_inp = get_hanoi_inp_path()

    scenarios = load_all_scenarios(hanoi_dir)

    if args.scenarios > 0:
        scenarios = scenarios[:args.scenarios]
        print(f"  Using {len(scenarios)} scenarios")

    if not scenarios:
        print("  ✗ No scenarios loaded!")
        sys.exit(1)

    # Get node names and network info
    node_names = get_node_names(scenarios[0])
    print(f"  Nodes: {len(node_names)}")
    print(f"  Timesteps per scenario: {len(scenarios[0]['labels'])}")
    print(f"  Total scenarios: {len(scenarios)}")

    # Load network for adjacency info
    print("\n  Loading WNTR network for adjacency data...")
    wn = load_network(hanoi_inp)
    adjacency = get_adjacency(wn)
    print(f"  Adjacency computed for {len(adjacency)} nodes")

    # ── Train/val split by scenario ───────────────────────────────
    scenario_ids = np.array([sc["scenario_id"] for sc in scenarios])
    n_train = max(1, int(len(scenarios) * 0.8))  # 80/20 split = 400/100

    train_ids = set(scenario_ids[:n_train])
    val_ids = set(scenario_ids[n_train:])
    print(f"\n  Train scenarios: {len(train_ids)}")
    print(f"  Val scenarios:   {len(val_ids)}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Train XGBoost (Model 1) ───────────────────────────────────
    detector = None
    xgb_metrics = {}
    if not args.no_xgboost:
        detector, xgb_metrics = train_xgboost(
            scenarios, adjacency, node_names, train_ids, val_ids
        )
    else:
        print("\n  Skipping XGBoost training as requested.")
        try:
            with open(OUTPUTS_DIR / "xgboost_metrics.json") as f:
                xgb_metrics = json.load(f)
            print("  Loaded existing XGBoost metrics.")
        except:
            pass

    # ── Train LSTM (Model 2) ─────────────────────────────────────
    lstm_trainer = None
    if not args.no_lstm:
        lstm_trainer = train_lstm(scenarios, train_ids, val_ids)

    # ── Train GAT (Model 3) ──────────────────────────────────────
    gat_trainer = None
    if not args.no_gat:
        try:
            gat_trainer = train_gat(
                scenarios, hanoi_inp, node_names, train_ids, val_ids
            )
        except Exception as e:
            print(f"\n  ⚠ GAT training failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n  XGBoost:")
    print(f"    Train F1: {xgb_metrics.get('train_f1', 0):.4f}")
    print(f"    Val F1:   {xgb_metrics.get('val_f1', 0):.4f}")
    print(f"    Threshold: {xgb_metrics.get('threshold', 0):.4f}")

    if lstm_trainer:
        print(f"\n  LSTM Autoencoder:")
        print(f"    Anomaly threshold: {lstm_trainer.threshold:.6f}")

    if gat_trainer:
        gat_path = OUTPUTS_DIR / "gat_metrics.json"
        if gat_path.exists():
            gat_m = json.load(open(gat_path))
            print(f"\n  GAT Localiser:")
            print(f"    Best F1: {gat_m.get('best_f1', 0):.4f}")
            print(f"    Localisation Accuracy: {gat_m.get('final_localisation_accuracy', 0):.4f}")

    print(f"\n  Published baselines to beat:")
    print(f"    MNF detector F1 = 0.50, detection delay ~6 hours")
    if xgb_metrics:
        if xgb_metrics.get("val_f1", 0) > 0.50:
            print(f"    ✅ XGBoost F1 ({xgb_metrics['val_f1']:.4f}) BEATS baseline!")
        else:
            print(f"    ⚠ XGBoost F1 ({xgb_metrics.get('val_f1', 0):.4f}) below baseline")

    print(f"\n  Models saved to: {MODELS_DIR}")
    print(f"  Metrics saved to: {OUTPUTS_DIR}")

    # Save combined model info
    model_info = {
        "n_nodes": len(node_names),
        "node_names": node_names,
        "n_scenarios_trained": len(train_ids),
        "n_scenarios_val": len(val_ids),
        "xgb_threshold": xgb_metrics.get("threshold", 0.4),
        "xgb_val_f1": xgb_metrics.get("val_f1", 0),
        "lstm_threshold": lstm_trainer.threshold if lstm_trainer else _get_existing_lstm_threshold(),
        "gat_available": gat_trainer is not None,
        "hanoi_inp": hanoi_inp,
        "trained_at": __import__("datetime").datetime.now().isoformat(),
    }

    with open(MODELS_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)


if __name__ == "__main__":
    main()
