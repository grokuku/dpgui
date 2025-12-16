# DPGui Features Overview

## 🖥️ Dashboard & Monitoring
- [x] **Real-time Stats**: Live CPU, RAM, and per-GPU (Temp, Fan, VRAM, Power) monitoring.
- [x] **Job Status**: Visual indicators for Pending, Running, Completed, and Failed jobs.
- [x] **Live Logs**: WebSocket-based terminal to view training logs in real-time (Non-blocking implementation).

## ⚙️ Job Management
- [x] **Configuration Generator**: GUI for `diffusion-pipe` TOML configuration (Z-Image focused).
- [x] **Job Pool**: Save configurations to a pool without running them immediately.
- [x] **Queue System**: Drag jobs from Pool to Queue. Auto-execution of the next job.
- [x] **History**: View logs and configurations of past jobs.
- [ ] **Execution Reliability**: currently blocked by dependency hell (Torch version mismatch).

## 📂 Dataset Manager
A full-featured IDE for preparing training data. **(COMPLETED Phase 4)**

### Organization
- [x] **CRUD**: Create, Rename, Clone, and Delete entire datasets.
- [x] **Export**: Download datasets as `.zip` files.
- [x] **Navigation**: Keyboard-friendly grid navigation (Arrows, Shift+Click selection).

### Image & Caption Editing
- [x] **Split-View**: Efficient workflow with image grid on the left and editor on the right.
- [x] **Auto-Save**: Captions save automatically 1 second after typing stops.
- [x] **Fullscreen**: Double-click to view images in high resolution.
- [x] **Drag'n'Drop**: Upload images and existing `.txt` captions directly into the browser.

### Batch Tools
- [x] **Batch Resize**: Resize multiple images to standard resolutions (512, 768, 1024) preserving aspect ratio.
- [x] **Trigger Words**: Inject trigger words into the start/end of caption files for selected images.
- [x] **Bulk Delete**: Remove multiple images and their associated text files instantly.
- [ ] **Auto-Tagging**: Integration with WD14/JoyCaption (Planned Phase 5).