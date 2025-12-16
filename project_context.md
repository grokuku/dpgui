# Project Context: DPGui

## ⚠️ AI KNOWLEDGE WARNING
**Crucial Note for Development:** The AI model used for this project has a knowledge cutoff (early 2023) that conflicts with the current environment (Late 2025).
*   **Reality:** PyTorch Stable is **2.9.1**, CUDA is **12.8** or **13.0**.
*   **AI Bias:** The AI tends to hallucinate that PyTorch 2.6 is "too new" or "nightly".
*   **Rule:** Always verify versions via `pip check` or official docs. Do not trust the AI's assertions on "current stable versions" without verification.

## Current State
- **Phase**: Phase 4 (Dataset Manager) **COMPLETED**.
- **Phase**: Phase 5 (Training Execution) **BLOCKED**.
- **Environment**: 
    - **Stable & Validated**: Python 3.11 + PyTorch 2.9.1 + CUDA 12.8.
    - **Dependencies**: All missing libs (wandb, tensorboard, etc.) are installed.
    - **Compat Patches**: 
        - `sqlite3` hotfix applied automatically for Python 3.11.
        - `node_helpers.py` symlink added for ComfyUI.

## Critical Blocking Issue: "The Zombie Port 29500"

### Symptoms
The training fails to start with the following error on **Rank 0**:
`torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. port: 29500 ... EADDRINUSE`

### Context
1.  **Zombie Processes**: Previous failed runs (due to code errors) left processes active on the GPUs. These processes are holding the default PyTorch Distributed port (**29500**).
2.  **Ignored Configuration**: 
    - The Job Manager explicitly generates a random port (e.g., `47066`).
    - The command line passes `--master_port=47066`.
    - The environment variable `MASTER_PORT` is forcibly set to `47066` in `job_manager.py`.
3.  **The Bug**: Despite all these overrides, `deepspeed.init_distributed()` or `train.py` **ignores the custom port** and falls back to the default `29500`, colliding with the zombie processes.

### Secondary Issue (Configuration)
- **Rank 1 Error**: `ValueError: Invalid pretrained_model_name_or_path`.
- **Cause**: The SDXL pipeline configuration in `train.py` expects a direct file path to a `.safetensors` file (via `from_single_file`), but the UI currently sends a Repository ID (e.g., `stabilityai/...`).
- **Fix Needed**: The UI `Jobs.jsx` was updated to support downloading specific files, but the user input needs to point to the local file, not the repo ID.

## Architecture Highlights
- **Launcher**: Now uses an **Active Check** strategy. It verifies if `torch` 2.9.1 is importable. If not, it forces a reinstall using the `cu128` index.
- **Job Manager**: 
    - Implements `_patch_compatibility()` to rewrite incompatible code in `vendor/diffusion-pipe` on the fly (Hotfix).
    - Implements `_fix_vendor_symlinks()` to handle `comfy`, `hyvideo`, and `node_helpers.py`.
- **Config Gen**: Now generates random ports to attempt to evade conflicts.

## Next Steps for Debugging
1.  **Kill Zombies**: Manually kill all python processes to free port 29500 (`pkill -f python`).
2.  **Fix Port Propagation**: Investigate why `train.py` ignores `MASTER_PORT`. It might be hardcoded or overridden inside the `diffusion-pipe` script logic.
3.  **Validate SDXL Path**: Ensure the configuration sent to the backend points to a valid local `.safetensors` file.