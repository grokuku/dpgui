--- START OF FILE DEV_PLAN.md ---

# Development Plan: DPGui

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

## Phase 5: Training Execution (🔄 IN PROGRESS - PIVOT)
- [x] Integration of `diffusion-pipe` vendor repository.
- [x] Dependency conflict resolution (Pydantic, DeepSpeed, CUDA).
- [x] **Failure Analysis**: Vendor script (`train.py`) is incompatible with ZeRO-2 (Offload).
- [ ] **Custom Engine Development**:
    - [ ] Create `train_dpgui_custom.py`.
    - [ ] Implement Standard DeepSpeed Engine (Non-Pipeline).
    - [ ] Reuse vendor DataLoaders.
    - [ ] Implement Training Loop with ZeRO-2 support.
- [ ] Verify Multi-GPU distributed training.
- [ ] Validate Checkpointing and LoRA saving.

## Phase 6: Polish & Release (📅 Future)
- [ ] Error handling refinement
- [ ] UX improvements
- [ ] Documentation

--- END OF FILE DEV_PLAN.md ---