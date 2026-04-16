# Coding Conventions

**Backend (Python)**
- **Framework**: Use `FastAPI`. Structure application entrypoints using the modern `lifespan` hook (via `asynccontextmanager`).
- **Dependencies**: Manage using `requirements.txt`.
- **Model Tracing**: Models must support ONNX export. Save/export pipelines (e.g. `save()` methods) should validate ONNX execution directly against PyTorch output immediately via `onnxruntime` during the export step, utilizing stored representative samples instead of random noise variables wherever possible.

**Frontend (React/JS)**
- **Styling**: Strictly utilize Tailwind CSS via the implemented "Lunar Observatory" Stitch guidelines (dark themes, translucent glass layers `bg-black/50 backdrop-blur`, neon borders).
- **Environment**: Abstract root URLs via `config.js` pointing to generic `import.meta.env.VITE...` configs. Avoid hardcoded `localhost:8000` URLs.
- **Component States**: Do not mutate D3 components by re-rendering entirely. Use D3 `.transition()` inside `useEffect` references while holding chart/data instances to reduce thrashing for high-frequency charts like `SensorChart.jsx` and `NetworkGraph.jsx`.
