# Project Context: ZeRO-Pipe (formerly DPGui)

## ⚠️ AI KNOWLEDGE WARNING
**Crucial Note:** The AI model used for this project has a knowledge cutoff (early 2023).
*   **Reality:** PyTorch Stable is **2.9.1**, CUDA is **12.8** or **13.0**.
*   **Rule:** Always verify versions via `pip check`.

## Project Identity
- **Name**: ZeRO-Pipe
- **Goal**: GUI and Orchestration layer for `diffusion-pipe` with native ZeRO-Stage 2/3 support.
- **Target Hardware**: Multi-GPU Consumer Clusters (e.g., 5x RTX 3090/4090).

## Current State
- **Phase**: Phase 5 (Training Execution - Custom Engine & Debugging).
- **Environment**: 
    - Python 3.11 + PyTorch 2.9.1 + CUDA 12.8.
    - DeepSpeed 0.17.0.
- **Engine**: `scripts/zeropipe.py` (Custom Flat Model Wrapper + ZeRO-3 Init).

## Critical Blocker: System RAM OOM (SIGKILL -9)
- **Status**: 🔴 BLOCKING
- **Symptoms**: 
    - **Rank 0 is killed (SIGKILL -9)** immediately after finishing model initialization (exiting `deepspeed.zero.Init` context).
    - Ranks 1-4 crash with `Connection reset by peer` / `Broken pipe` (TCPStore failure) because Rank 0 dies.
    - **Context**: 5x GPUs, SDXL Model, 64GB System RAM.
- **Progress**:
    - ✅ Fixed `TypeError` by manually calculating global batch size.
    - ✅ Fixed `NotImplementedError: Cannot copy out of meta tensor` via Monkeypatch.
    - ✅ Model structure initializes successfully on Meta device (`num_params = 1533`).
    - ❌ The actual partitioning or buffer allocation upon exiting `zero.Init` still spikes RAM beyond 64GB with 5 processes.

## Technical Decisions
1.  **Architecture**: Use `SingleStageModel` (Flat Wrapper) + Standard DeepSpeed Engine.
2.  **Compatibility**: Force `DS_SKIP_CUDA_CHECK=1`.
3.  **RAM Optimization Strategy (Active)**:
    - **ZeRO Stage 3**: Partitioning parameters to reduce redundancy.
    - **`deepspeed.zero.Init()`**: Allocating weights on "Meta" device during construction.
    - **Monkeypatch**: Patched `deepspeed.runtime.zero.partition_parameters.Init._post_init_method` to bypass transformer/diffusers meta-tensor access errors.
    - **`pin_memory=False`**: Disabled in config to save RAM.
    - **Dataset Workers**: Forced `NUM_PROC=1`.
    - **Lazy Loading**: Dataset is memory-mapped, not broadcast via TCP.

## Next Steps
1.  **Reduce Parallelism (Diagnosis)**: Run with **2 GPUs** instead of 5 to confirm if the OOM is strictly due to the number of processes (5x overhead) or a single-process leak.
2.  **Swap/Paged RAM**: Verify system swap usage.
3.  **Pre-Materialization**: Investigate if specific SDXL components (CLIP, VAE) are being materialized in RAM by `diffusers` *before* DeepSpeed can shard them, despite the `zero.Init` context.
4.  **Empty Cache Aggressively**: Force Python GC and CUDA cache clearing between every major initialization step.