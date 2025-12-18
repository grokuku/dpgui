# ZeRO-Pipe Features & Status

## Core System
| Feature | Status | Notes |
| :--- | :---: | :--- |
| **Launcher** | ✅ Ready | Auto-installs Node, PyTorch, CUDA deps. Handles Symlinks. |
| **Process Manager** | ✅ Ready | Async process handling, PID management, graceful shutdowns. |
| **Log Streaming** | ✅ Ready | WebSocket based, real-time updates. |
| **Port Management** | ✅ Ready | Auto-detection and conflict resolution for DDP ports. |

## Training Engine (ZeroPipe)
| Feature | Status | Notes |
| :--- | :---: | :--- |
| **Engine Core** | 🚧 Beta | `zeropipe.py` implements custom Flat Model wrapper. |
| **Multi-GPU** | ⚠️ Debug | OOM (System RAM) issues during dataset broadcast on 5 GPUs. |
| **ZeRO-2 Offload** | ⚙️ Ready | Configured in `config_gen.py`, waiting for stable run. |
| **Model Support** | ✅ Mapped | SDXL, Flux, Hunyuan, LTX, Wan, Cosmos, Lumina, Z-Image. |
| **LoRA/Full** | ✅ Ready | Engine supports both Adapter and Full Fine-Tuning modes. |

## User Interface
| Feature | Status | Notes |
| :--- | :---: | :--- |
| **Dashboard** | ✅ Ready | GPU monitoring, System stats. |
| **Job Wizard** | ✅ Ready | Complex form validation, TOML/JSON preview. |
| **Terminal** | ✅ Ready | XTerm.js integration. |
| **Dataset Manager** | 🚧 Pending | Basic path selection ready, advanced tools planned. |