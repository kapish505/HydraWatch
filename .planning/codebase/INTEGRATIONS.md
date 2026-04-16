# Integrations & External Interfaces

**Data Sources**
- **BattLeDIM 2019**: Replay data used to simulate live streaming telemetry (`/data/simulator.py`).
- **LeakDB**: KIOS LeakDB datasets used for model training (`/data/loader.py`).

**Local Databases**
- **SQLite**: Local file-based storage for user accounts, activity logs, and system alerts (`backend/db.py`). No external DB required.

**Client-Server Communication**
- **REST API**: FastAPI HTTP endpoints for authentication, network structure retrieval (`/network`), configuration, and starting specific replay scenarios.
- **WebSockets**:
  - `/ws/live`: Live event streaming for real-time telemetry updates.
  - `/ws/stream`: Stream specific playback sessions (e.g., LeakDB scenarios) to the frontend dashboard.
