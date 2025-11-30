# dpgui - Phased Development Plan

This document outlines a phased approach for developing the `diffusion-pipe-gui` (dpgui) application. The goal is to start with a Minimal Viable Product (MVP) and incrementally add more advanced features.

---

### **Phase 0: Project Setup & Automated Installation** [COMPLETED]

**Objective:** Create a robust, one-step process for setting up the entire application environment and launching the server.

1.  **Backend Dependencies (`requirements.txt`):** [x]
    *   Create and maintain a `requirements.txt` file for the FastAPI backend.
    *   Initial dependencies include: `fastapi`, `uvicorn[standard]`, `python-multipart`, `toml`.

2.  **Launcher Script (`scripts/launcher.sh`):** [x]
    *   Create a shell script that fully automates the setup and launch process.
    *   **Responsibilities:**
        1.  Check for and auto-install Node.js/NPM.
        2.  Clone `diffusion-pipe`, `ComfyUI`, and `HunyuanVideo` into `vendor/`.
        3.  Handle `PYTHONPATH` isolation via `vendor/libs` symlinks.
        4.  Install all Python dependencies (with Py3.12 fixes).
        5.  Start Backend (random port) and Frontend (proxy).

---

### **Phase 1: The Core Functional MVP (Web)** [COMPLETED]

**Objective:** Create a basic web application that generates configurations and prepares the launch command.

1.  **Backend API (FastAPI):** [x]
    *   Develop API endpoint (`/generate-config`) that receives JSON configuration.
    *   Generate `train.toml` and `dataset.toml` with absolute path resolution.

2.  **Frontend UI (React):** [x]
    *   Develop the web interface using `react-hook-form`.
    *   Dynamic management of dataset directories.
    *   Connection to backend via Vite Proxy.

---

### **Phase 2: Integrated Execution & Monitoring** [COMPLETED]

**Objective:** Launch and monitor training runs directly from the web application.

1.  **Backend Process Manager:** [x]
    *   `process_manager.py`: Async subprocess management.
    *   Endpoints: `/start-training` (with specific CWD), `/stop-training`.

2.  **Real-time Log Streaming (WebSockets):** [x]
    *   WebSocket endpoint `/ws/logs`.
    *   Frontend `Terminal` component to display streaming logs.

---

### **Phase 3: Queue Management & Results Visualization** [NEXT]

**Objective:** Manage multiple training runs and visualize the resulting test images via the web interface.

1.  **Backend Queue Manager:**
    *   Develop API endpoints for adding jobs to a queue, viewing the queue status, etc.
    *   The backend will manage the lifecycle of these jobs, running them sequentially.

2.  **Image Gallery API:**
    *   The backend will monitor the `output_dir` for new images.
    *   Create an API endpoint that allows the frontend to fetch the list of generated images and view them. The frontend will be updated in real-time via WebSocket notifications.

---

### **Phase 4: The Dataset Manager** [FUTURE]

**Objective:** Build a comprehensive tool for dataset preparation through the web interface.

1.  **Dataset API (FastAPI):**
    *   Create a full suite of CRUD (Create, Read, Update, Delete) API endpoints for managing dataset files.
    *   Endpoints for listing images, reading/writing caption files (`.txt`), and deleting files.

2.  **Frontend Dataset UI (React):**
    *   Build the corresponding user interface for the Dataset Manager.
    *   Allow users to browse image thumbnails, click to view/edit captions, and upload new images.

3.  **Auto-Tagging API:**
    *   Integrate a captioning model into the backend.
    *   Create an API endpoint that takes an image path and returns generated tags. The frontend will have an "Auto-Tag" button to call this endpoint.