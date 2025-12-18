# Development Plan: ZeRO-Pipe

## Phase 1: Environment & Setup (✅ Done)
- [x] Define architecture (React + Python/FastAPI)
- [x] Setup Git repository and initial structure
- [x] Create `launcher.sh` for dependency management
- [x] Validate PyTorch 2.9.1 + CUDA 12.8 environment

## Phase 2: Backend Core (✅ Done)
- [x] FastAPI implementation (`main.py`)
- [x] Job Manager (`job_manager.py`)
- [x] WebSocket Log Streaming (`process_manager.py`)
- [x] Configuration Generation (`config_gen.py`)

## Phase 3: Frontend Interface (✅ Done)
- [x] Dashboard (System status)
- [x] Job Creation Wizard (Forms, Validation)
- [x] Real-time Terminal (Logs)
- [x] History & Management

## Phase 4: Data & Configuration (✅ Done)
- [x] Dataset processing utilities
- [x] TOML Configuration mapping
- [x] DeepSpeed JSON Configuration generation

## Phase 5: Training Execution (🔄 IN PROGRESS - CRITICAL)
- [x] **Pivot Architecture**: Abandoned patching `train.py`.
- [x] **ZeroPipe Engine**: Created `scripts/zeropipe.py` implementing a standard DeepSpeed Engine with ZeRO-2 support.
- [x] **Model Mapping**: SDXL, Flux, Hunyuan, LTX, Wan, etc. mapped in the new engine.
- [ ] **RAM Optimization**: Fix OOM (SIGKILL) during dataset broadcasting on multi-GPU setups (64GB RAM limit).
- [ ] **Validation**: Successfully run SDXL training on 5 GPUs.
- [ ] **Advanced Models**: Validate Flux/Qwen training with ZeRO-Offload.

## Phase 6: Polish & Release (📅 Future)
- [ ] Rename internal variables from `dpgui` to `zeropipe` where appropriate.
- [ ] UX improvements (Loss Curves, Image Gallery).
- [ ] Documentation.