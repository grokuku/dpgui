# dpgui - Phased Development Plan

This document outlines a phased approach for developing the `diffusion-pipe-gui` (dpgui) application.

---

### **Phase 0: Project Setup & Automated Installation** [COMPLETED]
*   [x] Backend Dependencies (`requirements.txt`).
*   [x] Launcher Script (`scripts/launcher.sh`) with auto-install for Node/Python/Libs.

### **Phase 1: The Core Functional MVP (Web)** [COMPLETED]
*   [x] Backend API (`/generate-config`).
*   [x] Frontend UI (React) with `react-hook-form`.
*   [x] TOML Generation with absolute path resolution.

### **Phase 2: Integrated Execution & Monitoring** [COMPLETED]
*   [x] Process Manager (Async subprocess).
*   [x] Real-time Log Streaming (WebSockets).
*   [x] System Stats (GPU/CPU/RAM).

### **Phase 3: Queue Management** [COMPLETED]
*   [x] Job Manager (Persistence, JSON storage).
*   [x] Queue Logic (Pending -> Running -> Completed).
*   [x] Dashboard UI (Active Queue vs Job Pool).

### **Phase 4: The Dataset Manager** [COMPLETED]
**Objective:** Build a comprehensive tool for dataset preparation through the web interface.

1.  **Dataset API (FastAPI) & Utils:** [x]
    *   CRUD Operations: Create, Rename, Delete, Clone Datasets.
    *   File Operations: Upload (Multi-file), Export (ZIP), Delete Images.
    *   Batch Operations: Resize Images, Add Trigger Words.
    *   Image Processing: Thumbnail generation via `Pillow`.

2.  **Frontend Dataset UI (React):** [x]
    *   **Navigation:** Grid view with keyboard support (Arrows), "None" state handling.
    *   **Editing:** Split-view (Grid + Editor Sidebar).
    *   **Captioning:** Auto-save functionality, quick text editor.
    *   **UX:** Drag'n'Drop uploads (Images + TXT), Fullscreen preview.

### **Phase 5: Advanced Features & Refinement** [NEXT]
**Objective:** Polish the experience and add intelligence.

1.  **Auto-Tagging:**
    *   Integrate a local Vision-LLM or Tagger (e.g., WD14 or JoyCaption) to auto-fill `.txt` files.
2.  **Model Management:**
    *   UI to manage/download base models (checkpoints, VAEs).
3.  **Training Visualization:**
    *   Embed TensorBoard or specialized graphs for Loss/Learning Rate.