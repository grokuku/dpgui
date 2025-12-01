# Project Context: DPGui

## Current State
- **Phase**: Phase 4 Completed (Dataset Manager).
- **Status**: 
    - **Launcher**: Fully automated (Node.js, Python env, Cloning).
    - **Backend**: 
        - Generates valid TOML/DeepSpeed commands.
        - Job System: `JobManager` (Pool vs Queue).
        - **Dataset System**: `dataset_utils.py` handles CRUD, Batch Ops, and File I/O safely.
        - **CORS**: Correctly configured for frontend communication.
    - **Frontend**: 
        - React Router navigation.
        - **Dataset Manager**: Full IDE-like experience (Split view, Auto-save, Drag'n'Drop, Batch Tools).
        - Dashboard: Monitoring & Logs.

## Architecture & Workflow
1.  **Launcher (`launcher.sh`)**:
    - Single source of truth for dependencies.
    - Installs `Pillow` (New) for image processing.
    - Starts Backend (`uvicorn`) and Frontend (`vite`).
2.  **Backend (FastAPI)**:
    - `main.py`: Entry point, exposes API routes (Jobs & Datasets).
    - `dataset_utils.py`: **(New)** Manages file system, image resizing, zip export, and caption IO.
    - `job_manager.py`: Orchestrates training jobs.
    - `config_gen.py`: Resolves paths and generates TOML.
3.  **Frontend (React/Vite)**:
    - `Datasets.jsx`: **(New)** Complex UI for dataset management (Grid + Sidebar, Keyboard nav).
    - Uses `vite.config.js` proxy for WebSocket and API calls.

## Dependencies Management
### Backend (Python)
- **Core**: `fastapi`, `uvicorn`, `python-multipart`, `toml`, `pydantic` (V2), `psutil`.
- **Image Processing**: `Pillow` (Added in Phase 4).
- **Vendorized**: `diffusion-pipe`, `ComfyUI`.

### Frontend (Node.js)
- **Runtime**: Node v22.12.0 (LTS) auto-installed.
- **Libs**: `axios`, `react-hook-form`, `react-router-dom`, `lucide-react`.

## Known Issues / Notes
- **Auto-Tagging**: Placeholder button in UI. Requires external model integration (Phase 4.1 or 5).
- **Z-Image Priority**: UI enforces Z-Image model selection.
- **Hot-Reload**: Backend requires restart (`launcher.sh`) if python files are modified.