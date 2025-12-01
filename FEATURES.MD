# DPGui Features Overview

## 🖥️ Dashboard & Monitoring
- **Real-time Stats**: Live CPU, RAM, and per-GPU (Temp, Fan, VRAM, Power) monitoring.
- **Job Status**: Visual indicators for Pending, Running, Completed, and Failed jobs.
- **Live Logs**: WebSocket-based terminal to view training logs in real-time.

## ⚙️ Job Management
- **Configuration Generator**: GUI for `diffusion-pipe` TOML configuration (Z-Image focused).
- **Job Pool**: Save configurations to a pool without running them immediately.
- **Queue System**: Drag jobs from Pool to Queue. Auto-execution of the next job.
- **History**: View logs and configurations of past jobs.

## 📂 Dataset Manager (New)
A full-featured IDE for preparing training data.

### Organization
- **CRUD**: Create, Rename, Clone, and Delete entire datasets.
- **Export**: Download datasets as `.zip` files.
- **Navigation**: Keyboard-friendly grid navigation (Arrows, Shift+Click selection).

### Image & Caption Editing
- **Split-View**: Efficient workflow with image grid on the left and editor on the right.
- **Auto-Save**: Captions save automatically 1 second after typing stops.
- **Fullscreen**: Double-click to view images in high resolution.
- **Drag'n'Drop**: Upload images and existing `.txt` captions directly into the browser.

### Batch Tools
- **Batch Resize**: Resize multiple images to standard resolutions (512, 768, 1024) preserving aspect ratio.
- **Trigger Words**: Inject trigger words into the start/end of caption files for selected images.
- **Bulk Delete**: Remove multiple images and their associated text files instantly.