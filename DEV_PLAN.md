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
*   [x] CRUD Operations & File Upload.
*   [x] Batch Operations (Resize, Trigger Words).
*   [x] Image/Caption Editor.
*   [x] Image Grid & Thumbnails.

### **Phase 5: Advanced Features & Refinement** [IN PROGRESS - BLOCKED]
**Objective:** Polish the experience and add intelligence.

1.  **Job Execution Reliability:** [IN PROGRESS]
    *   [x] **Zombie Ports**: Fixed via dynamic port reallocation.
    *   [x] **Config Ignored**: Fixed via Hot-Patching `train.py`.
    *   [ ] **Optimizer Compatibility**: Currently debugging `AssertionError: zero stage 2 requires an optimizer`.
    *   [ ] **Action**: Downgrade to ZeRO-1 or Refactor Optimizer init.

2.  **Model Management:** [PARTIAL]
    *   [x] Basic Local Model Scanning.
    *   [x] HuggingFace Downloader.
    *   [ ] Fix SDXL Configuration path logic (Repo ID vs File Path) - *Workaround applied*.

3.  **Training Visualization:**
    *   Embed TensorBoard or specialized graphs for Loss/Learning Rate.