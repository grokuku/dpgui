# dpgui - Phased Development Plan

This document outlines a phased approach for developing the `diffusion-pipe-gui` (dpgui) application.

---

### **Phase 0: Project Setup & Automated Installation** [COMPLETED]
*   [x] Backend Dependencies (`requirements.txt` - Updated for 2025 Stack).
*   [x] Launcher Script (`scripts/launcher.sh`) with Auto-Healing & Active Version Check.

### **Phase 1: The Core Functional MVP (Web)** [COMPLETED]
*   [x] Backend API (`/generate-config`).
*   [x] Frontend UI (React) with `react-hook-form`.
*   [x] TOML Generation with absolute path resolution.

### **Phase 2: Integrated Execution & Monitoring** [COMPLETED]
*   [x] Process Manager (Async subprocess).
*   [x] Real-time Log Streaming (WebSockets - Non-blocking fixed).
*   [x] System Stats (GPU/CPU/RAM).

### **Phase 3: Queue Management** [COMPLETED]
*   [x] Job Manager (Persistence, JSON storage).
*   [x] Queue Logic (Pending -> Running -> Completed).
*   [x] Dashboard UI (Active Queue vs Job Pool).

### **Phase 4: The Dataset Manager** [COMPLETED]
**Objective:** Build a comprehensive tool for dataset preparation through the web interface.
*   [x] CRUD Operations & File Upload.
*   [x] Batch Operations (Resize, Trigger Words).
*   [x] Image/Caption Editor.
*   [x] Image Grid & Thumbnails.

### **Phase 5: Advanced Features & Refinement** [IN PROGRESS - BLOCKED]
**Objective:** Polish the experience and add intelligence.

1.  **Job Execution Reliability:** [BLOCKED]
    *   **Status**: Environment is healthy (Torch 2.9.1 / CUDA 12.8).
    *   **Blocker**: `DistNetworkError` (Address already in use). The system ignores the dynamic port assignment and collides with zombie processes from previous crashes.
    *   **Actions Taken**: Implemented Auto-Patching for Python 3.11 compatibility, Symlink repairs, and Environment Variable injection.

2.  **Model Management:** [PARTIAL]
    *   [x] Basic Local Model Scanning.
    *   [x] HuggingFace Downloader (Added support for single file download).
    *   [ ] Fix SDXL Configuration path logic (Repo ID vs File Path).

3.  **Training Visualization:**
    *   Embed TensorBoard or specialized graphs for Loss/Learning Rate.