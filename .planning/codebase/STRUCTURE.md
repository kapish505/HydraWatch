# Codebase Structure

```
├── .planning/                  # Project planning and codebase overview documents
├── backend/                    # FastAPI Server and Core ML logic
│   ├── data/                   # Data loaders, telemetry simulators, feature engineers
│   ├── models/                 # PyTorch & XGBoost model definitions, trainers, ONNX exporters
│   ├── routers/                # FastAPI routing logic (auth, admin, users)
│   ├── db.py                   # SQLite interface
│   ├── main.py                 # FastAPI application entrypoint and WebSockets
│   └── network.py              # WNTR-based network topology mapping
├── data/                       # Datasets, local SQLite DB, uploaded images
├── frontend/                   # React Single-Page Application
│   ├── public/                 # Static assets
│   ├── src/                    # App source
│   │   ├── contexts/           # React context providers (AuthContext)
│   │   ├── pages/              # Main view routes (Dashboard, Home, Admin)
│   │   ├── config.js           # Network environment configuration mappings
│   │   └── NetworkGraph.jsx    # D3.js implementation
│   ├── index.html              # HTML shell (Google Fonts overrides)
│   └── tailwind.config.js      # Global palette & theming variables (Stitch design)
├── scripts/                    # Sub-process routines
│   ├── evaluate.py             # Evaluation script for baseline metric comparisons
│   ├── extract_leakdb.py       # Dataset pre-processors
│   └── train.py                # Pipeline for training and tuning the three-model ensemble
├── stitch_designs/             # Inspiration maps for UI framework
└── requirements.txt            # Python dependencies
```
