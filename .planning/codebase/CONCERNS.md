# Known Concerns & Tech Debt

## Security (High Priority)
- **Hard-coded SECRET_KEY** in `backend/auth.py` (`"hydrawatch-secret-key-change-in-production"`). Must be moved to environment variable before any non-local deployment.
- **SQLite credential storage** — user passwords (bcrypt hashed) and audit logs in a local `data/hydrawatch.db` file. The DB file itself has no access controls.
- **File upload validation** — `routers/users.py` saves uploaded profile pictures with minimal extension checking. No MIME type validation, no file size limit.
- **CORS wildcard** — `allow_origins=["*"]` in `main.py`. Acceptable for local dev, not for production.

## Reliability
- **Ensemble false negatives** — the triple-agreement requirement means a miscalibrated single model (e.g., LSTM threshold too high) can suppress valid detections from the other two models.
- **Missing sensor imputation** — feature engineering expects all N sensors to report values. A failed SCADA node (NaN values) will crash `compute_rolling_stats()` and downstream features.
- **WebSocket single-process** — `_state["ws_clients"]` is an in-memory `set()`. Cannot scale horizontally without a shared message broker (Redis pub/sub or similar).

## Performance
- **Feature engineering bottleneck** — `compute_rolling_stats()` and `compute_neighbor_delta()` in `backend/data/features.py` use sequential Python loops over `T` timesteps. For the full 500-scenario dataset (~168K timesteps), this is slow. Should be vectorized with pandas rolling or numpy stride tricks.
- **GAT ONNX export fails** — `gat.py:save()` attempts ONNX export but PyG's GATConv uses dynamic scatter operations that aren't supported by ONNX opset. The `.pt` checkpoint works fine; ONNX is a nice-to-have.
- **MPS fallback** — GAT training on macOS requires `PYTORCH_ENABLE_MPS_FALLBACK=1` because `scatter_reduce` isn't natively supported on MPS, causing automatic CPU fallback for those ops.

## Technical Debt
- **Legacy `api.py`** — `backend/api.py` (13KB) still exists but all core routes are in `main.py`. The file appears to be an older version that was superseded. Should be audited and removed if unused.
- **Legacy `__pycache__`** — `backend/models/__pycache__/` may contain stale `.pyc` for the deleted `gnn_model.py`. Should be cleared.
- **Hard-coded thresholds** — ensemble thresholds (`lstm_threshold=0.85`, `xgb_threshold=0.4`, severity boundaries `0.3/0.5/0.8`) are hard-coded in `ensemble.py`. Should be configurable via environment variables or `model_info.json`.
- **Format normalization** — `loader.py` requires `Node_N` format. LeakDB uses bare integers (`19`), BattLeDIM uses `P_N` or `n_N` conventions. The normalization is fragile and spread across `loader.py` and `features.py`.
- **Dashboard component in pages/** — `pages/Dashboard.jsx` duplicates some header/layout that's also in the parent `App.jsx` Layout component, causing potential visual conflicts when auth is active.

## Missing Features
- **No token revocation** — JWTs can't be invalidated before expiry (7 days). A compromised token remains valid until it expires.
- **No automated tests** — no pytest, jest, or CI/CD pipeline. Verification is manual via replay and evaluation script.
- **No Docker** — `docker-compose.yml` referenced in spec but not yet implemented (intentionally deferred).
