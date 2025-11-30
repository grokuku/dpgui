# Project Context: DPGui

## Current State
- **Phase**: Phase 3 Completed (Queue Logic & Z-Image Pivot). Ready for Phase 4.
- **Status**: 
    - Full-stack launcher is operational (handles Node.js, Python env, Cloning).
    - Backend generates valid TOML files and resolves relative paths.
    - **Job System**: `JobManager` implemented with persistence (JSON) and "Pool vs Queue" logic.
    - **Monitoring**: System resources (CPU/RAM/GPU) and real-time logs visible in Dashboard.
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
    - `job_manager.py`: Orchestrates jobs, manages `jobs/` (JSON) and `logs/` persistence.
    - `process_manager.py`: Manages async subprocesses (DeepSpeed).
    - `config_gen.py`: Resolves relative paths.
    - **Workflow**: Job Creation -> Pool (`STOPPED`) -> Queue (`PENDING`) -> Execution (`RUNNING`).
3.  **Frontend (React/Vite)**:
    - Uses `vite.config.js` proxy.
    - **New Structure**: React Router based navigation (Dashboard, Jobs, Datasets, Settings).
    - **Dashboard**: Split view "Active Queue" vs "Job Pool".

## Dependencies Management
### Backend (Python)
- **Core**: `fastapi`, `uvicorn`, `python-multipart`, `toml`, `pydantic` (V2 syntax), `psutil`.
- **Vendorized**: 
    - `diffusion-pipe` (Core).
    - `ComfyUI` (Enabled - Required for Z-Image).
    - `HunyuanVideo` (Disabled).

### Frontend (Node.js)
- **Runtime**: Node v22.12.0 (LTS) auto-installed.
- **Libs**: `axios`, `react-hook-form`, `react-router-dom` (Nav), `lucide-react` (Icons).

## Known Issues / Notes
- **Z-Image Priority**: The UI currently enforces **Z-Image** selection. Other models are disabled.
- **Python 3.12 Compatibility**: Handled via `launcher.sh` (setuptools upgrade).
- **Path Resolution**: Relative paths in UI are converted to absolute paths in TOML.