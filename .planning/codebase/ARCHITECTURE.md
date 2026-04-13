# Architecture

## System Pattern
**Monolithic FastAPI backend + React SPA frontend** with a 3-model ML ensemble.

```
┌─────────────────────────────────────────────────────────┐
│                     React SPA (Vite)                     │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Network  │ │  Sensor   │ │  Alert   │ │   SHAP   │  │
│  │  Graph   │ │  Chart    │ │  Panel   │ │Waterfall │  │
│  │  (D3.js) │ │(Chart.js) │ │          │ │          │  │
│  └────┬─────┘ └─────┬─────┘ └────┬─────┘ └────┬─────┘  │
│       └──────────────┴───────────┴─────────────┘        │
│                WebSocket + REST (fetch)                   │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────┐
│                 FastAPI Backend (main.py)                 │
│                                                          │
│  ┌─ Routers ──────────────────────────────────────────┐  │
│  │ auth.py   users.py   admin.py                      │  │
│  │ (JWT)     (profile)  (user mgmt)                   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─ Core API Routes ─────────────────────────────────┐   │
│  │ GET /network    POST /ingest    GET /alerts       │   │
│  │ WS  /ws/stream  WS   /ws/live   GET /metrics     │   │
│  │ GET /replay/start               GET /api/scenarios│   │
│  └───────────────────────────┬───────────────────────┘   │
│                              │                           │
│  ┌─ Ensemble (ensemble.py) ──┴───────────────────────┐   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │   │
│  │  │   LSTM   │  │ XGBoost  │  │     GAT      │    │   │
│  │  │Autoencod.│  │ +SHAP    │  │  Localiser   │    │   │
│  │  │(lstm_ae) │  │(xgboost) │  │   (gat.py)   │    │   │
│  │  └──────────┘  └──────────┘  └──────────────┘    │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─ Data Layer ──────────────────────────────────────┐   │
│  │ loader.py   features.py   simulator.py            │   │
│  │ (LeakDB +   (10 features  (BattLeDIM              │   │
│  │  BattLeDIM)  per node)     replay)                │   │
│  └───────────────────┬───────────────────────────────┘   │
│                      │                                   │
│  ┌─ Storage ─────────┴───────────────────────────────┐   │
│  │ db.py (SQLite: alerts, users, activity_logs)      │   │
│  │ network.py (WNTR → JSON graph)                    │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## Data Flow

### Inference Path (Real-time)
1. Sensor data arrives via `POST /ingest` or WebSocket replay
2. Feature engineering builds 3 model inputs simultaneously:
   - LSTM: sliding window `(1, 24, N_sensors)` → anomaly score
   - XGBoost: `(1, N*10)` engineered features → probability + SHAP
   - GAT: `(N, 3)` node features → per-node probability map
3. Ensemble checks triple agreement (LSTM > 0.85, XGB > 0.4, node overlap)
4. If alert fires → stored in SQLite, pushed to all WebSocket clients

### Training Path (Offline)
1. `scripts/train.py` loads all 500 LeakDB scenarios
2. Splits 80/20 by scenario ID (400 train / 100 val)
3. Trains XGBoost → LSTM → GAT sequentially
4. Saves `.pt`, `.pkl`, `.onnx` models to `models/`

## Authentication Middleware
- JWT-based via `backend/auth.py`
- `get_current_user()` dependency extracts token from `Authorization: Bearer` header
- Core ML routes (`/network`, `/ingest`, `/alerts`, `/ws/*`) are **not** behind auth
- User management routes (`/api/users/*`, `/api/admin/*`) require auth

## Entry Points
- **Backend:** `uvicorn backend.main:app --port 8000`
- **Frontend dev:** `cd frontend && npm run dev` (Vite dev server)
- **Training:** `python scripts/train.py`
- **Evaluation:** `python scripts/evaluate.py`
