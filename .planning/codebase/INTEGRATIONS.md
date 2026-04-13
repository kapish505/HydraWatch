# External Integrations & Services

## Datasets (External Data Sources)
| Dataset | Source | Usage |
|---------|--------|-------|
| **LeakDB** (Hanoi_CMH) | [github.com/KIOS-Research/LeakDB](https://github.com/KIOS-Research/LeakDB) / [Zenodo](https://zenodo.org/records/13985057) | 500 scenarios for training all 3 models. Stored in `data/raw/leakdb/` |
| **BattLeDIM** | [Zenodo](https://zenodo.org/records/4017659) | 2018 SCADA (normal baseline), 2019 SCADA (replay demo), 2019 Leakages (ground truth). Stored in `data/raw/battledim/` |

Download/extract scripts: `scripts/download_data.py`, `scripts/extract_leakdb.py`

## Internal Services

### JWT Authentication System
- **Provider:** Self-hosted (no external auth provider)
- **Implementation:** `backend/auth.py` — `create_access_token()`, `get_current_user()`
- **Algorithm:** HS256 with hard-coded `SECRET_KEY`
- **Token lifetime:** 7 days
- **Routes:** `backend/routers/auth.py` (login/logout), `backend/routers/users.py` (profile), `backend/routers/admin.py` (user management)

### SQLite Database
- **File:** `data/hydrawatch.db`
- **Schema:** Managed by `backend/db.py:init_db()`
  - `alerts` — leak detection alerts with SHAP payloads
  - `users` — user accounts (email, bcrypt password hash, role, status)
  - `activity_logs` — user action audit trail
- **Not an external DB** — local file-based, initialised on first startup

### WebSocket (Real-time Push)
- **Endpoint:** `ws://localhost:8000/ws/stream` — LeakDB scenario replay stream
- **Endpoint:** `ws://localhost:8000/ws/live` — BattLeDIM live replay push (30s intervals)
- **Client management:** In-memory set in `_state["ws_clients"]`
- **No external broker** — single-process only

### WNTR Hydraulic Simulation
- **Library:** `wntr` 1.2.0
- **Input:** EPANET `.inp` files (Hanoi_CMH.inp, L-TOWN.inp)
- **Usage:** Network graph extraction, adjacency computation, base demand lookup
- **Module:** `backend/network.py`

## External API Calls
- **None at runtime** — all models run locally, no cloud inference
- Download scripts use `requests` for initial dataset download only

## File Upload System
- **Endpoint:** `POST /api/users/me/picture`
- **Storage:** `data/uploads/` directory
- **Served via:** FastAPI `StaticFiles` mount at `/uploads`
