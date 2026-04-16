# System Architecture

HydraWatch is a full-stack, ML-driven application composed of a FastAPI backend and a React single-page frontend.

**1. Machine Learning Ensemble**
The application evaluates telemetry anomaly detection passing through an asynchronous `IngestBuffer` using a three-tier model ensemble (`models/ensemble.py`):
1. **LSTM Autoencoder**: Identifies temporal pattern anomalies via reconstruction loss (`models/lstm_ae.py`).
2. **XGBoost Classifier**: Analyzes 10-feature engineered representations to emit leak probabilities and TreeSHAP visual reasoning strings.
3. **Graph Attention Network (GAT)**: Computes spatial probability topologies using hydraulic network edge graphs (`models/gat.py`).

Final alerts are generated only when these signals find consensus.

**2. Backend Web Architecture (FastAPI)**
- `main.py` establishes the `lifespan` hook. Loads trained checkpoints (`.pt`, ONNX bindings).
- Manages an ephemerally cached (deque-based) `IngestBuffer` with a 48-step rolling window to ingest telemetry blocks and re-evaluate live rolling standards (std dev, mean variance).
- Uses `async` threads to push state chunks through WebSockets (`/ws/live` and `/ws/stream`).
- Provides a SQLite adapter (`backend/db.py`) for user registration, authenticating JWT tokens, and logging generated detection events.

**3. Frontend Architecture (React + Vite)**
- Adheres to the "Lunar Observatory" aesthetic using dark backgrounds, neon accents, and `framer-motion` for spatial transitions.
- Evaluates real-time WebSocket streams in `Dashboard.jsx`.
- Plots complex SVG nodal pressure structures using `D3.js` inside `NetworkGraph.jsx`, binding SHAP features and prediction states visually to the edges and nodes.
- Maps API URLs dynamically using `config.js` (`VITE_API_URL`, `VITE_WS_URL`).
