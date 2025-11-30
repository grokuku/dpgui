# Project Context: DPGui

## Current State
- **Phase**: Phase 2 Completed (Execution & Monitoring). Ready for Phase 3.
- **Status**: 
    - Full-stack launcher is operational (handles Node.js, Python env, Cloning).
    - Backend generates valid TOML files and resolves relative paths.
    - Training process runs in background with real-time WebSocket log streaming to Frontend.
    - "Vendor" strategy implemented to handle complex dependency conflicts (`utils` package collision).

## Architecture & Workflow
1.  **Launcher (`launcher.sh`)**:
    - Checks/Installs Node.js (portable version).
    - Clones dependencies into `vendor/` (`diffusion-pipe`, `ComfyUI`, `HunyuanVideo`).
    - **Crucial**: Creates `vendor/libs` with symlinks to `comfy` and `hyvideo`.
    - **Crucial**: Sets `PYTHONPATH` to `vendor/libs` to isolate modules and avoid loading ComfyUI's `utils` folder.
    - Starts Backend on a random free port.
    - Starts Frontend on fixed port (default 5173).
2.  **Backend (FastAPI)**:
    - `process_manager.py`: Manages async subprocesses (`deepspeed`). Uses `os.setsid` to handle process groups for clean termination.
    - `config_gen.py`: Resolves relative paths (`data`, `output`) to absolute paths for `deepspeed`.
    - WebSockets: Streams `stdout`/`stderr` combined.
3.  **Frontend (React/Vite)**:
    - Uses `vite.config.js` proxy to talk to the backend's random port.
    - Terminal component with auto-scroll for logs.

## Dependencies Management
### Backend (Python)
- **Core**: `fastapi`, `uvicorn`, `python-multipart`, `toml`, `pydantic`.
- **Vendorized**: 
    - `diffusion-pipe` (Main logic).
    - `ComfyUI` (Required for `comfy` module).
    - `HunyuanVideo` (Required for `hyvideo` module).
- **Fixes applied**:
    - `numpy` removed from `HunyuanVideo/requirements.txt` during install to prevent `ImpImporter` error on Python 3.12.
    - Build tools (`pip`, `setuptools`, `wheel`) are auto-upgraded.

### Frontend (Node.js)
- **Runtime**: Node v22.12.0 (LTS) auto-installed if missing.
- **Stack**: React, Vite.
- **Libs**: `axios`, `react-hook-form`.

## Known Issues / Notes
- **Python 3.12 Compatibility**: Requires `setuptools` upgrade and avoiding old `numpy` builds from source. Handled in `launcher.sh`.
- **Path Resolution**: The backend assumes execution from the project root. Relative paths in UI are converted to absolute paths in TOML.