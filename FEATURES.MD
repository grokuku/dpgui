--- START OF FILE FEATURES.MD ---

# DPGui Features & Status

## Core System
| Feature | Status | Notes |
| :--- | :---: | :--- |
| **Launcher** | ✅ Ready | Auto-installs Node, PyTorch, CUDA deps. Handles Symlinks. |
| **Process Manager** | ✅ Ready | Async process handling, PID management, graceful shutdowns. |
| **Log Streaming** | ✅ Ready | WebSocket based, real-time updates, circular buffer. |
| **Port Management** | ✅ Ready | Auto-detection and conflict resolution for DDP ports. |

## Training Engine (Strategic Pivot)
| Feature | Status | Notes |
| :--- | :---: | :--- |
| **Vendor Integration** | ⚠️ Partial | `diffusion-pipe` libs mapped. Script patching abandoned. |
| **DeepSpeed** | 🔄 Refactor | Moving to Custom Training Loop to support ZeRO-2 Offload. |
| **Memory Opt.** | ⚙️ Configured | ZeRO Stage 2 + CPU Offload enabled in config. |
| **Compatibility** | ✅ Fixed | Pydantic V2 & CUDA 13.0/12.8 mismatch solved via config/env vars. |

## User Interface
| Feature | Status | Notes |
| :--- | :---: | :--- |
| **Dashboard** | ✅ Ready | GPU monitoring placeholder, System stats. |
| **Job Wizard** | ✅ Ready | Complex form validation, TOML/JSON preview. |
| **Terminal** | ✅ Ready | XTerm.js integration, ANSI color support. |
| **Dataset Manager** | 🚧 Pending | Basic path selection ready, advanced tools planned. |

--- END OF FILE FEATURES.MD ---