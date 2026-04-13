# Directory Structure

```
HydraWatch/
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app, all routes, WebSocket handlers, startup loader
│   ├── auth.py                    # JWT auth: create_access_token, get_current_user, bootstrap_admin
│   ├── db.py                      # SQLite schema + CRUD (alerts, users, activity_logs)
│   ├── network.py                 # WNTR .inp → JSON (network_to_json), adjacency, edge_index
│   ├── api.py                     # (Legacy API module — routes now in main.py)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # LeakDB scenario loader + BattLeDIM SCADA/leakages parser
│   │   ├── features.py            # Feature engineering (10 features per node + LSTM windows + GAT features)
│   │   └── simulator.py           # BattLeDIMReplay class (streaming generator for live demo)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_ae.py             # LSTMAutoencoder + LSTMAutoencoderTrainer (anomaly detection)
│   │   ├── gat.py                 # LeakGAT + GATTrainer (per-node leak localisation)
│   │   ├── xgboost_model.py       # LeakDetectorXGB + SHAP explanations
│   │   └── ensemble.py            # EnsembleDetector (triple-agreement logic)
│   └── routers/
│       ├── auth.py                # POST /api/auth/login, POST /api/auth/register
│       ├── users.py               # GET /api/users/me, POST /api/users/me/picture
│       └── admin.py               # GET /api/admin/users, PUT /api/admin/users/{id}/status
├── frontend/
│   ├── index.html                 # Vite HTML entry
│   ├── package.json               # React 19, D3 v7, Chart.js v4, Tailwind v3
│   ├── vite.config.js
│   ├── tailwind.config.js         # Custom hw-* color tokens
│   ├── postcss.config.js
│   ├── eslint.config.js
│   ├── src/
│   │   ├── main.jsx               # React DOM entry (renders <App />)
│   │   ├── index.css              # Tailwind + glass-panel + D3 node styles + pulse animations
│   │   ├── App.jsx                # Router: Login, Dashboard, Profile, AdminDashboard + AuthProvider
│   │   ├── App.css                # Additional app-level styles
│   │   ├── NetworkGraph.jsx       # D3 force-directed network graph (node colors, pulse animation)
│   │   ├── SensorChart.jsx        # Chart.js pressure time-series (60 timesteps, red alert lines)
│   │   ├── AlertPanel.jsx         # Alert list (severity badge, suspect nodes, SHAP top-3)
│   │   ├── ShapWaterfall.jsx      # SHAP horizontal bar chart (red=leak, green=normal)
│   │   ├── contexts/
│   │   │   └── AuthContext.jsx    # Auth state: token, user, login/logout, JWT decode
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # Main dashboard (network graph + chart + alerts + SHAP)
│   │   │   ├── Login.jsx          # Login form page
│   │   │   ├── Profile.jsx        # User profile + picture upload
│   │   │   └── AdminDashboard.jsx # Admin user management panel
│   │   └── assets/
│   └── dist/                      # Production build output
├── models/                        # Trained model artifacts
│   ├── lstm_autoencoder.pt        # LSTM Autoencoder checkpoint (297KB)
│   ├── lstm_autoencoder.onnx      # ONNX export (300KB)
│   ├── gat_localiser.pt           # GAT checkpoint (144KB)
│   ├── xgboost_leak_detector.pkl  # XGBoost + threshold (1.4MB)
│   └── model_info.json            # Training metadata (node count, thresholds, timestamp)
├── data/
│   ├── hydrawatch.db              # SQLite database
│   ├── raw/
│   │   ├── leakdb/                # LeakDB Hanoi_CMH scenarios (500 dirs)
│   │   └── battledim/             # BattLeDIM SCADA xlsx + leakages csv + L-TOWN.inp
│   └── uploads/                   # User profile pictures
├── outputs/
│   ├── xgboost_metrics.json       # XGBoost training metrics
│   ├── lstm_metrics.json          # LSTM training metrics
│   └── gat_metrics.json           # GAT training metrics
├── scripts/
│   ├── download_data.py           # Download LeakDB + BattLeDIM from Zenodo
│   ├── extract_leakdb.py          # Extract LeakDB zip into scenario dirs
│   ├── train.py                   # Complete training pipeline (all 3 models)
│   └── evaluate.py                # Evaluation against BattLeDIM 2019 ground truth
├── requirements.txt               # Python dependencies (pinned)
├── README.md
└── venv/                          # Python virtual environment
```

## Key Locations
| What | Where |
|------|-------|
| Backend entry | `backend/main.py` |
| ML models | `backend/models/` (4 files) |
| Feature engineering | `backend/data/features.py` |
| Data loading | `backend/data/loader.py` |
| Network graph utils | `backend/network.py` |
| Auth system | `backend/auth.py` + `backend/routers/` |
| Frontend entry | `frontend/src/main.jsx` → `App.jsx` |
| Dashboard UI | `frontend/src/pages/Dashboard.jsx` |
| Visualisation components | `frontend/src/NetworkGraph.jsx`, `SensorChart.jsx`, `AlertPanel.jsx`, `ShapWaterfall.jsx` |
| Trained models | `models/` (3 model files + metadata) |
| Training pipeline | `scripts/train.py` |
