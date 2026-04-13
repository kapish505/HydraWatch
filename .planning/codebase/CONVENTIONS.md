# Code Conventions

## Python Backend

### Module Organisation
- **One model per file:** `lstm_ae.py`, `gat.py`, `xgboost_model.py`, `ensemble.py`
- **Model + Trainer pattern:** Each model file exports both the `nn.Module` class and a `Trainer` harness class
  - e.g. `LSTMAutoencoder` (nn.Module) + `LSTMAutoencoderTrainer` (train/predict/save/load)
- **Data layer:** `loader.py` (I/O), `features.py` (compute), `simulator.py` (replay)
- **Routers:** Prefixed with `/api/auth`, `/api/users`, `/api/admin` — each in `backend/routers/`

### Naming
- **Functions:** `snake_case` — `build_xgboost_features()`, `get_anomaly_score()`
- **Classes:** `PascalCase` — `LSTMAutoencoder`, `LeakDetectorXGB`, `EnsembleDetector`
- **Constants:** `UPPER_SNAKE_CASE` — `MODEL_DIR`, `PROJECT_ROOT`, `DATA_DIR`
- **Node references:** Always `Node_N` format (e.g., `Node_19`), normalised in loader

### Type Annotations
- Full type hints on all function signatures: `def predict(self, windows: np.ndarray) -> Dict[str, np.ndarray]:`
- Uses `from typing import Dict, Any, List, Optional, Tuple`
- Docstrings use Args/Returns format

### Error Handling
- `FileNotFoundError` for missing data/model files with helpful install instructions
- Graceful PyG import fallback with `HAS_PYG` flag in GAT module
- MPS/CUDA/CPU auto-detection: `if torch.backends.mps.is_available() ... elif torch.cuda.is_available() ... else cpu`

### Dependency Injection (FastAPI)
- `get_current_user` / `get_current_active_admin` as `Depends()` parameters
- Core ML routes are **not** behind auth dependencies
- Router-based auth routes use `Depends(get_current_user)` for protected endpoints

### Global State
- `_state` dict in `main.py` holds loaded models, network data, WebSocket clients
- Initialised in `@app.on_event("startup")` handler
- Models loaded lazily with try/except fallbacks

## JavaScript Frontend

### Component Pattern
- Functional components with hooks (`useState`, `useEffect`, `useRef`, `useCallback`)
- No class components anywhere
- All components are default exports

### State Management
- **AuthContext** (`contexts/AuthContext.jsx`) — global auth state via `useAuth()` hook
- **Local state** — `useState` in Dashboard for `currentData`, `alerts`, `pressureHistory`
- **WebSocket** — managed via `useRef(wsRef)` with `startSimulation`/`stopSimulation` callbacks

### Routing
- `react-router-dom` v7 with `<BrowserRouter>`, `<Routes>`, `<Route>`
- `<ProtectedRoute>` wrapper — redirects to `/login` if no token
- `<Layout>` wrapper — renders nav bar for authenticated users

### CSS/Styling
- **Tailwind CSS v3** with custom `hw-*` color tokens in `tailwind.config.js`
- **Glass panel effect** — `.glass-panel` class in `index.css` (rgba bg + backdrop-filter)
- **Component-level styling** — Tailwind utility classes inline on JSX
- **Fonts:** Inter (body) + JetBrains Mono (numbers, code)

### D3 Pattern
- D3 rendered into a `ref` container via `useEffect`
- Coordinate scaling from WNTR graph coordinates to pixel space
- Node updates via D3 transitions tied to `currentData` dependency
