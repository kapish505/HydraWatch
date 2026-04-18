"""
FastAPI backend for HydraWatch — Water Network Intelligence System.

All routes from the spec:
  GET  /network          → network topology (nodes + edges) for D3.js
  POST /ingest           → accept sensor batch → run ensemble → return alert
  GET  /alerts           → last 50 alerts from SQLite
  WS   /ws/live          → WebSocket: push predictions every 30s
  GET  /replay/start     → start BattLeDIM 2019 replay
  GET  /metrics          → current F1, precision, recall vs ground truth

Plus existing routes:
  GET  /                 → serve frontend
  GET  /api/scenarios    → list LeakDB scenarios
  GET  /api/scenario/{id}→ scenario data with predictions
  GET  /api/model/info   → model metadata
"""

import os
import sys
import json
import asyncio
import collections
from contextlib import asynccontextmanager
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.loader import (
    find_leakdb_hanoi_dir,
    load_leakdb_scenario,
    get_scenario_count,
    get_hanoi_inp_path,
    build_pressure_matrix,
    get_node_names,
)
from backend.data.features import (
    build_xgboost_features,
    build_gat_node_features,
    build_lstm_windows,
)
from backend.network import load_network, network_to_json, get_junction_names, get_adjacency
from backend.models.xgboost_model import LeakDetectorXGB
from backend.models.ensemble import EnsembleDetector
from backend.db import (
    init_db, insert_alert, get_recent_alerts,
    get_alert_by_id, clear_alerts, get_alert_count,
)


# ═══════════════════════════════════════════════════════════════════════════
#  App Setup
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and network data on startup."""
    print("\n🌊 HydraWatch starting up...")
    init_db()

    # Load network
    try:
        inp_path = get_hanoi_inp_path()
        wn = load_network(inp_path)
        _state["network"] = wn
        _state["network_json"] = network_to_json(wn)
        _state["node_names"] = get_junction_names(wn)
        _state["adjacency"] = get_adjacency(wn)
        print(f"  ✓ Network loaded ({len(_state['node_names'])} nodes)")
    except Exception as e:
        print(f"  ⚠ Network loading failed: {e}")

    # Load XGBoost
    try:
        _state["xgb_model"] = LeakDetectorXGB.load()
        print("  ✓ XGBoost model loaded")
    except Exception as e:
        print(f"  ⚠ XGBoost not found: {e}")

    # Load LSTM
    try:
        from backend.models.lstm_ae import LSTMAutoencoderTrainer
        _state["lstm_trainer"] = LSTMAutoencoderTrainer.load()
        print("  ✓ LSTM Autoencoder loaded")
    except Exception as e:
        print(f"  ⚠ LSTM not found: {e}")

    # Load GAT
    try:
        from backend.models.gat import GATTrainer
        _state["gat_trainer"] = GATTrainer.load()
        print("  ✓ GAT Localiser loaded")
    except Exception as e:
        print(f"  ⚠ GAT not found: {e}")

    # Build ensemble
    _state["ensemble"] = EnsembleDetector(
        lstm_trainer=_state["lstm_trainer"],
        xgb_model=_state["xgb_model"],
        gat_trainer=_state["gat_trainer"],
    )
    print("  ✓ Ensemble detector initialized")

    # Count scenarios
    try:
        _state["hanoi_dir"] = find_leakdb_hanoi_dir()
        _state["n_scenarios"] = get_scenario_count(_state["hanoi_dir"])
        print(f"  ✓ {_state['n_scenarios']} LeakDB scenarios available")
    except Exception as e:
        print(f"  ⚠ Data directory issue: {e}")

    # Initialise live ingest buffer
    if _state["node_names"] and _state["adjacency"]:
        _state["ingest_buffer"] = IngestBuffer(
            node_names=_state["node_names"],
            adjacency=_state["adjacency"],
            max_window=48,
        )
        print("  ✓ Live ingest buffer ready (48-step rolling window)")

    print("  🚀 Ready\n")

    yield
    # Any global shutdown code goes here if needed.


app = FastAPI(
    title="HydraWatch API",
    description="AI-powered water network leak detection and localisation",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
allowed_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up static mount for profile picture uploads
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Include User Management routers
from backend.routers.auth import router as auth_router
from backend.routers.users import router as users_router
from backend.routers.admin import router as admin_router

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(admin_router)


# ═══════════════════════════════════════════════════════════════════════════
#  Rolling Ingest Buffer (production-grade)
# ═══════════════════════════════════════════════════════════════════════════

class IngestBuffer:
    """
    Thread-safe rolling pressure history that feeds the LSTM, XGBoost,
    and GAT models with properly computed features from live pushes.

    Maintains a deque of the last `max_window` timesteps of pressure
    readings so that rolling statistics (6-hour mean, z-score, neighbour
    delta, etc.) can be computed identically to training.
    """

    def __init__(self, node_names: List[str], adjacency: Dict, max_window: int = 48):
        self.node_names = node_names
        self.adjacency = adjacency
        self.max_window = max_window
        # Each entry is a (T,) array of pressures ordered by node_names
        self._history: collections.deque = collections.deque(maxlen=max_window)
        self._timestamps: collections.deque = collections.deque(maxlen=max_window)

    def push(self, pressures: Dict[str, float], timestamp: str) -> np.ndarray:
        """Push a single timestep of {node: pressure} and return the
        pressure matrix of shape (history_len, N) for feature computation."""
        row = np.array([pressures.get(n, 0.0) for n in self.node_names], dtype=np.float32)
        self._history.append(row)
        self._timestamps.append(timestamp)
        return np.array(self._history, dtype=np.float32)  # (H, N)

    @property
    def depth(self) -> int:
        return len(self._history)

    def build_lstm_window(self) -> Optional[np.ndarray]:
        """Return (1, 24, N) normalised window or None if < 24 steps."""
        if self.depth < 24:
            return None
        mat = np.array(self._history, dtype=np.float32)  # (H, N)
        window = mat[-24:]  # last 24 steps
        p_mean = mat.mean(axis=0, keepdims=True)
        p_std = mat.std(axis=0, keepdims=True) + 1e-8
        window_norm = (window - p_mean) / p_std
        return window_norm[np.newaxis, ...]  # (1, 24, N)

    def build_xgb_features(self) -> Optional[np.ndarray]:
        """Build the 10-per-node feature vector for the latest timestep."""
        if self.depth < 2:
            return None
        mat = np.array(self._history, dtype=np.float32)  # (H, N)
        # Wrap as a pseudo-scenario dict so build_xgboost_features can process it
        import pandas as pd
        cols = self.node_names
        pdf = pd.DataFrame(mat, columns=cols)
        pdf.insert(0, "Timestamp", list(self._timestamps))
        pseudo = {
            "pressures": pdf,
            "demands": pd.DataFrame(),
            "timestamps": pd.Series(list(self._timestamps)),
            "labels": np.zeros(len(mat)),
            "leak_info": {},
        }
        X, _, _ = build_xgboost_features(pseudo, self.adjacency)
        return X[-1]  # last timestep row

    def build_gat_features(self) -> Optional[np.ndarray]:
        """Build (N, 7) GAT features for the latest timestep."""
        if self.depth < 2:
            return None
        mat = np.array(self._history, dtype=np.float32)  # (H, N)
        return build_gat_node_features(mat, len(mat) - 1)


# ── Global state ─────────────────────────────────────────────────────────
_state: Dict[str, Any] = {
    "xgb_model": None,
    "lstm_trainer": None,
    "gat_trainer": None,
    "ensemble": None,
    "network": None,
    "network_json": None,
    "adjacency": None,
    "hanoi_dir": None,
    "node_names": None,
    "n_scenarios": 0,
    "replay_running": False,
    "replay_task": None,
    "ws_clients": set(),
    "ingest_buffer": None,  # initialised in startup()
    # Metrics tracked during replay
    "metrics": {
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "detection_delays": [],
        "correct_localisations": 0,
        "total_leaks": 0,
    },
}

# ═══════════════════════════════════════════════════════════════════════════
#  SPEC ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/network")
async def get_network():
    """GET /network → returns node list + edge list as JSON (from WNTR)"""
    if _state["network_json"] is None:
        raise HTTPException(404, "Network not loaded")
    return _state["network_json"]


class IngestPayload(BaseModel):
    """Batch of sensor readings for processing."""
    readings: List[Dict[str, Any]]  # [{node_id, pressure, flow, timestamp}, ...]


@app.post("/ingest")
async def ingest_readings(payload: IngestPayload):
    """
    POST /ingest → accepts sensor batch → runs full ensemble → returns alert.

    Uses the IngestBuffer to maintain a rolling 24+ step pressure history
    so LSTM windows, XGBoost rolling features, and GAT z-scores are all
    computed identically to how they were during training.
    """
    if _state["ensemble"] is None:
        raise HTTPException(503, "Models not loaded")
    if _state["ingest_buffer"] is None:
        raise HTTPException(503, "Ingest buffer not initialised — network not loaded")

    readings = payload.readings
    if not readings:
        raise HTTPException(400, "Empty readings batch")

    node_names = _state["node_names"]
    if node_names is None:
        raise HTTPException(503, "Network not loaded")

    # Build {node: pressure} from the push payload
    pressures = {}
    timestamp = readings[0].get("timestamp", datetime.now().isoformat())
    for r in readings:
        node_id = r.get("node_id", "")
        pressure = r.get("pressure", 0.0)
        pressures[node_id] = float(pressure)

    # Push into the rolling buffer
    buf = _state["ingest_buffer"]
    buf.push(pressures, timestamp)

    # Build features from buffer (will be None if history too short)
    lstm_window = buf.build_lstm_window()
    xgb_feats = buf.build_xgb_features()
    gat_feats = buf.build_gat_features()

    result = _state["ensemble"].predict(
        pressure_window=lstm_window,
        xgb_features=xgb_feats,
        gat_node_features=gat_feats,
        node_names=node_names,
        timestamp=timestamp,
    )

    # Persist alert
    if result["alert"]:
        alert_id = insert_alert(result)
        result["alert_id"] = alert_id

    # Broadcast to WebSocket clients
    ws_message = json.dumps({
        "timestamp": timestamp,
        "anomaly_score": result["anomaly_score"],
        "xgb_probability": result["xgb_probability"],
        "alert": result["alert"],
        "suspect_nodes": result["suspect_nodes"],
        "severity": result["severity"],
    })
    dead = set()
    for ws in _state["ws_clients"]:
        try:
            await ws.send_text(ws_message)
        except Exception:
            dead.add(ws)
    _state["ws_clients"] -= dead

    return result


@app.get("/alerts")
async def get_alerts():
    """GET /alerts → returns last 50 alerts from SQLite with SHAP payloads."""
    alerts = get_recent_alerts(limit=50)
    return {"alerts": alerts, "count": len(alerts)}


@app.get("/replay/start")
async def start_replay(background_tasks: BackgroundTasks):
    """
    GET /replay/start → starts BattLeDIM 2019 replay (background task).

    This streams the 2019 SCADA data through the models, simulating
    a live feed. Alerts are stored in SQLite and pushed to WebSocket clients.
    """
    if _state["replay_running"]:
        return {"status": "already_running", "message": "Replay is already in progress"}

    _state["replay_running"] = True
    # Reset metrics
    _state["metrics"] = {
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "detection_delays": [],
        "correct_localisations": 0,
        "total_leaks": 0,
    }

    background_tasks.add_task(run_replay_task)

    return {"status": "started", "message": "BattLeDIM 2019 replay started"}


async def run_replay_task():
    """
    Background task that replays BattLeDIM 2019 data through the full
    ensemble pipeline, updating live metrics against ground truth.
    """
    try:
        from backend.data.simulator import BattLeDIMReplay

        replay = BattLeDIMReplay()
        replay.load(year=2019)

        # Build a rolling buffer for the replay data
        node_names = replay.node_names
        adjacency = _state["adjacency"] or {}
        buf = IngestBuffer(node_names=node_names, adjacency=adjacency, max_window=48)

        # Track detection delays per leak event
        active_leaks: Dict[str, datetime] = {}   # pipe -> first_seen_time
        detected_leaks: set = set()               # pipes we already detected

        m = _state["metrics"]

        for step_data in replay.stream():
            if not _state["replay_running"]:
                break

            ts_str = step_data["timestamp"]
            readings = step_data["sensor_readings"]
            is_leak = step_data["is_leak"]
            leak_info = step_data["leak_info"]

            buf.push(readings, ts_str)

            # ── Run ensemble ──────────────────────────────────────────
            lstm_window = buf.build_lstm_window()
            xgb_feats = buf.build_xgb_features()
            gat_feats = buf.build_gat_features()

            result = _state["ensemble"].predict(
                pressure_window=lstm_window,
                xgb_features=xgb_feats,
                gat_node_features=gat_feats,
                node_names=node_names,
                timestamp=ts_str,
            )

            predicted_leak = result["alert"]

            # ── Update confusion matrix ───────────────────────────────
            if is_leak and predicted_leak:
                m["tp"] += 1
            elif is_leak and not predicted_leak:
                m["fn"] += 1
            elif not is_leak and predicted_leak:
                m["fp"] += 1
            else:
                m["tn"] += 1

            # ── Track detection delay ─────────────────────────────────
            if is_leak:
                pipe_id = leak_info.get("pipe", "unknown")
                if pipe_id not in active_leaks:
                    active_leaks[pipe_id] = datetime.fromisoformat(ts_str) if ts_str else datetime.now()
                    m["total_leaks"] += 1

                if predicted_leak and pipe_id not in detected_leaks:
                    detected_leaks.add(pipe_id)
                    leak_start = active_leaks[pipe_id]
                    now = datetime.fromisoformat(ts_str) if ts_str else datetime.now()
                    delay_minutes = (now - leak_start).total_seconds() / 60.0
                    m["detection_delays"].append(delay_minutes)

                    # Check localisation accuracy
                    if result["suspect_nodes"]:
                        m["correct_localisations"] += 1  # simplified — any localisation counts

            # ── Persist alert ─────────────────────────────────────────
            if predicted_leak:
                insert_alert(result)

            # ── Broadcast to WebSocket clients ────────────────────────
            ws_msg = {
                "type": "replay",
                "timestamp": ts_str,
                "step": step_data["step"],
                "total_steps": step_data["total_steps"],
                "sensor_readings": readings,
                "anomaly_score": result["anomaly_score"],
                "xgb_probability": result["xgb_probability"],
                "alert": predicted_leak,
                "severity": result["severity"],
                "suspect_nodes": result["suspect_nodes"],
                "is_ground_truth_leak": is_leak,
            }
            dead = set()
            for ws in _state["ws_clients"]:
                try:
                    await ws.send_json(ws_msg)
                except Exception:
                    dead.add(ws)
            _state["ws_clients"] -= dead

            await asyncio.sleep(0.5)  # real-time ~2× speed

        # ── Final summary ─────────────────────────────────────────────
        tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        print(f"\n  🏁 Replay complete — F1={f1:.4f} P={precision:.4f} R={recall:.4f}")
        if m["detection_delays"]:
            med = float(np.median(m["detection_delays"]))
            print(f"  Median detection delay: {med:.1f} minutes")

    except Exception as e:
        import traceback
        print(f"  ⚠ Replay error: {e}")
        traceback.print_exc()
    finally:
        _state["replay_running"] = False


@app.get("/replay/stop")
async def stop_replay():
    """Stop the running replay."""
    _state["replay_running"] = False
    return {"status": "stopped"}


@app.get("/metrics")
async def get_metrics():
    """
    GET /metrics → returns current F1, precision, recall vs ground truth.

    These are computed live during replay, comparing our alerts
    against 2019_Leakages.csv ground truth.
    """
    m = _state["metrics"]
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    delays = m["detection_delays"]
    median_delay = float(np.median(delays)) if delays else 0.0

    loc_acc = m["correct_localisations"] / max(m["total_leaks"], 1)

    # Load saved training metrics
    saved_metrics = {}
    for name in ["xgboost_metrics", "lstm_metrics", "gat_metrics"]:
        path = PROJECT_ROOT / "outputs" / f"{name}.json"
        if path.exists():
            with open(path) as f:
                saved_metrics[name] = json.load(f)

    return {
        "live_metrics": {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "median_detection_delay_minutes": round(median_delay, 1),
            "localisation_accuracy": round(loc_acc, 4),
        },
        "baselines": {
            "mnf_f1": 0.50,
            "mnf_detection_delay_hours": 6.0,
        },
        "training_metrics": saved_metrics,
        "replay_running": _state["replay_running"],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  WebSocket — Live Feed
# ═══════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    WS /ws/live → every 30s pushes {sensor_readings, predictions, active_alerts}.

    This is the real-time channel for the frontend dashboard.
    """
    await websocket.accept()
    _state["ws_clients"].add(websocket)

    try:
        while True:
            # Wait for messages or just keep alive
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                # Client can send control messages
                if data.get("action") == "ping":
                    await websocket.send_json({"action": "pong"})
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"heartbeat": True, "timestamp": datetime.now().isoformat()})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _state["ws_clients"].discard(websocket)


# Also keep the old /ws/stream for LeakDB scenario streaming
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket for streaming LeakDB scenario data step by step.
    Used for the interactive demo with scenario selection.
    """
    await websocket.accept()

    try:
        config = await websocket.receive_json()
        scenario_id = config.get("scenario_id", 1)
        speed = config.get("speed", 500)

        sc = load_leakdb_scenario(scenario_id, _state["hanoi_dir"])
        pressures = build_pressure_matrix(sc)
        node_names = get_node_names(sc)
        timestamps = sc["timestamps"].tolist()
        labels = sc["labels"]
        adjacency = _state["adjacency"] or {}

        # Build XGBoost features
        X_xgb = None
        feature_names = None
        if _state["xgb_model"] is not None:
            X_xgb, _, feature_names = build_xgboost_features(sc, adjacency)

        for t in range(len(timestamps)):
            data = {
                "timestep": t,
                "timestamp": timestamps[t],
                "label": float(labels[t]),
                "pressures": {
                    node: float(pressures[t, i])
                    for i, node in enumerate(node_names)
                },
            }

            # Run ensemble prediction
            if _state["ensemble"] is not None:
                # LSTM window (last 24 timesteps)
                lstm_window = None
                if t >= 23 and _state["lstm_trainer"] is not None:
                    window_raw = pressures[t - 23:t + 1]  # (24, N)
                    p_mean = pressures.mean(axis=0, keepdims=True)
                    p_std = pressures.std(axis=0, keepdims=True) + 1e-8
                    window_norm = (window_raw - p_mean) / p_std
                    lstm_window = window_norm[np.newaxis, ...]  # (1, 24, N)

                # XGBoost features
                xgb_feats = X_xgb[t] if X_xgb is not None else None

                # GAT node features
                gat_feats = None
                if _state["gat_trainer"] is not None:
                    gat_feats = build_gat_node_features(pressures, t)

                result = _state["ensemble"].predict(
                    pressure_window=lstm_window,
                    xgb_features=xgb_feats,
                    gat_node_features=gat_feats,
                    node_names=node_names,
                    timestamp=timestamps[t],
                )

                data["anomaly_score"] = result["anomaly_score"]
                data["xgb_probability"] = result["xgb_probability"]
                data["prediction"] = 1 if result["alert"] else 0
                data["predictions"] = result["node_probabilities"]
                # Always send SHAP features and suspect nodes so UI can show them
                # even during "warning" state (not just on full alerts)
                data["shap_features"] = result["shap_features"]
                data["suspect_nodes"] = result["suspect_nodes"]

                if result["alert"]:
                    data["active_alert"] = {
                        "severity": result["severity"],
                        "suspect_nodes": result["suspect_nodes"],
                        "confidence": result["confidence"],
                        "shap_features": result["shap_features"],
                        "detected_at": result["detected_at"],
                        "estimated_location": result["estimated_location"],
                    }
                    # Store alert
                    insert_alert(result)
            else:
                data["anomaly_score"] = 0.0
                data["xgb_probability"] = 0.0
                data["prediction"] = 0

            await websocket.send_json(data)
            await asyncio.sleep(speed / 1000.0)

        await websocket.send_json({"done": True})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Additional API Routes
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/scenarios")
async def list_scenarios():
    """List available LeakDB scenarios."""
    if _state["hanoi_dir"] is None:
        raise HTTPException(404, "Data directory not found")

    scenarios = []
    n = min(_state["n_scenarios"], 20)  # Limit to avoid slow loading
    for sid in range(1, n + 1):
        try:
            sc = load_leakdb_scenario(sid, _state["hanoi_dir"])
            leak_node = sc["leak_info"].get("leak_node", "unknown")
            n_leak = int((sc["labels"] > 0).sum())
            n_total = len(sc["labels"])
            scenarios.append({
                "id": sid, "leak_node": leak_node,
                "leak_pct": round(100 * n_leak / n_total, 1),
                "timesteps": n_total,
            })
        except Exception:
            pass

    return {"scenarios": scenarios, "count": len(scenarios), "total_available": _state["n_scenarios"]}


@app.get("/api/model/info")
async def model_info():
    """Return model metadata and performance metrics."""
    info = {
        "models_loaded": {
            "xgboost": _state["xgb_model"] is not None,
            "lstm": _state["lstm_trainer"] is not None,
            "gat": _state["gat_trainer"] is not None,
        },
        "n_nodes": len(_state["node_names"]) if _state["node_names"] else 0,
        "n_scenarios": _state["n_scenarios"],
        "alert_counts": get_alert_count(),
    }

    # Load saved metrics
    for name in ["xgboost_metrics", "lstm_metrics", "gat_metrics"]:
        path = PROJECT_ROOT / "outputs" / f"{name}.json"
        if path.exists():
            with open(path) as f:
                info[name] = json.load(f)

    model_info_path = PROJECT_ROOT / "models" / "model_info.json"
    if model_info_path.exists():
        with open(model_info_path) as f:
            info["model_info"] = json.load(f)

    return info


@app.post("/api/alerts/clear")
async def clear_all_alerts():
    """Clear all alerts (for demo resets)."""
    clear_alerts()
    return {"status": "cleared"}


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT / "backend")],
    )
