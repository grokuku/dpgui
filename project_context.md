# Project Context: DPGui

## Current State
- **Phase**: Phase 3 Started (Queue Management & GUI Restructuring).
- **Status**: 
    - Full-stack launcher is operational (handles Node.js, Python env, Cloning).
    - Backend generates valid TOML files and resolves relative paths.
    - Training process runs in background with real-time WebSocket log streaming.
    - "Vendor" strategy implemented to handle complex dependency conflicts.

## Deployment Constraints
- **Remote/Automated First**: The project runs on a remote server. No manual CLI commands (npm install, pip install) should be expected from the user. 
- **Launcher Responsibility**: `launcher.sh` is the single source of truth for checking and installing all new dependencies automatically.

## Architecture & Workflow
1.  **Launcher (`launcher.sh`)**:
    - Checks/Installs Node.js (portable version).
    - Clones dependencies into `vendor/`.
    - Creates `vendor/libs` symlinks.
    - Sets `PYTHONPATH` to `vendor/libs`.
    - **Installs Frontend dependencies (React Router, Lucide, etc.) automatically.**
    - Starts Backend and Frontend.
2.  **Backend (FastAPI)**:
    - `process_manager.py`: Manages async subprocesses.
    - `config_gen.py`: Resolves relative paths.
    - WebSockets: Streams `stdout`/`stderr`.
3.  **Frontend (React/Vite)**:
    - Uses `vite.config.js` proxy.
    - **New Structure**: React Router based navigation (Dashboard, Jobs, Datasets, Settings).

## Dependencies Management
### Backend (Python)
- **Core**: `fastapi`, `uvicorn`, `python-multipart`, `toml`, `pydantic`.
- **Vendorized**: `diffusion-pipe` (Core only), `ComfyUI` (Disabled), `HunyuanVideo` (Disabled).

### Frontend (Node.js)
- **Runtime**: Node v22.12.0 (LTS) auto-installed.
- **Libs**: `axios`, `react-hook-form`, `react-router-dom` (Nav), `lucide-react` (Icons).

## Known Issues / Notes
- **Python 3.12 Compatibility**: Handled via `launcher.sh`.
- **Path Resolution**: Relative paths in UI are converted to absolute paths in TOML.