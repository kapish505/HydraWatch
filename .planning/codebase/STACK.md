# Technology Stack

## Languages & Runtimes
- **Backend:** Python 3.11
- **Frontend:** JavaScript (JSX) — React 19 + Vite 8
- **Config/Data:** JSON, CSV, XLSX, EPANET `.inp`

## Backend Framework
- **FastAPI 0.110.0** — REST API + WebSocket server
- **Uvicorn 0.27.1** — ASGI server (`uvicorn backend.main:app`)
- Entry point: `backend/main.py`

## ML / Data Science Stack
| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.2.0 | LSTM Autoencoder (anomaly detection) |
| `torch-geometric` | 2.5.0 | GAT model (leak localisation via GATConv) |
| `xgboost` | 2.0.3 | Explainable leak classification |
| `shap` | 0.44.1 | SHAP TreeExplainer for XGBoost |
| `scikit-learn` | 1.4.0 | Metrics (F1, precision, recall, confusion matrix) |
| `wntr` | 1.2.0 | Water network simulation + EPANET .inp loading |
| `pandas` | 2.2.0 | DataFrame operations for SCADA data |
| `numpy` | 1.26.4 | Numerical arrays for feature engineering |
| `matplotlib` | 3.8.3 | (Available but not primary — Charts use frontend) |
| `onnx` | 1.15.0 | Model export format |
| `onnxruntime` | 1.17.0 | (Available for ONNX inference) |

## Authentication Stack
| Library | Version | Purpose |
|---------|---------|---------|
| `python-jose[cryptography]` | 3.3.0 | JWT signing/verification (HS256) |
| `passlib[bcrypt]` | 1.7.4 | Password hashing (bcrypt) |
| `python-multipart` | 0.0.9 | Form data parsing (login, file uploads) |

## Frontend Dependencies
| Library | Version | Purpose |
|---------|---------|---------|
| `react` | ^19.2.4 | UI framework |
| `react-dom` | ^19.2.4 | DOM rendering |
| `react-router-dom` | ^7.14.0 | Client-side routing (Login, Dashboard, Profile, Admin) |
| `d3` | ^7.9.0 | Network graph visualisation (force-directed layout) |
| `chart.js` | ^4.5.1 | Pressure time-series line charts |
| `react-chartjs-2` | ^5.3.1 | React wrapper for Chart.js |
| `tailwindcss` | ^3.4.19 | Utility-first CSS framework |
| `jwt-decode` | ^4.0.0 | Client-side JWT decoding for AuthContext |
| `@tailwindcss/forms` | ^0.5.11 | Form styling plugin |

## Build & Dev Tools
- **Vite 8** — frontend build toolchain (`frontend/vite.config.js`)
- **PostCSS + Autoprefixer** — CSS processing (`frontend/postcss.config.js`)
- **ESLint 9** — JavaScript linting (`frontend/eslint.config.js`)

## Data Storage
- **SQLite** (stdlib) — alert storage + user management (`data/hydrawatch.db`)
- Schema tables: `alerts`, `users`, `activity_logs`
- Initialised via `backend/db.py:init_db()`

## Configuration
- `requirements.txt` — Python dependencies (pinned versions)
- `frontend/package.json` — Node.js dependencies
- `frontend/tailwind.config.js` — custom theme tokens (`hw-*` colors)
- No `.env` file — secrets currently hard-coded in `backend/auth.py` (⚠)
