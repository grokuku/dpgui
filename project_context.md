# Project Context: DPGui

## ⚠️ AI KNOWLEDGE WARNING
**Crucial Note for Development:** The AI model used for this project has a knowledge cutoff (early 2023) that conflicts with the current environment (Late 2025).
*   **Reality:** PyTorch Stable is **2.9.1**, CUDA is **12.8** or **13.0**.
*   **Rule:** Always verify versions via `pip check` or official docs.

## Current State
- **Phase**: Phase 5 (Training Execution) **STRATEGIC PIVOT**.
- **Environment**: 
    - **Stable**: Python 3.11 + PyTorch 2.9.1 + CUDA 12.8 (System CUDA 13.0).
    - **DeepSpeed**: 0.17.0.
    - **Hardware**: 5 GPUs cluster (Consumer Grade -> Requires ZeRO-Offload).
- **Status**: 
    - **Patched Vendor Script**: **ABANDONED**. The vendor script enforces `PipelineParallelism` which is architecturally incompatible with `ZeRO-Stage 2` (CPU Offload). We cannot use one without disabling the other, and we need ZeRO-2 for memory constraints.
    - **New Strategy**: **"Hybrid Rewrite"**. We will create a custom training script (`train_dpgui_custom.py`) starting from zero, but reusing the vendor's data loading and model loading modules.

## Technical Insights (Learned Hard Way)
1.  **ZeRO vs Pipeline**: DeepSpeed strictly forbids combining ZeRO-Stage 2 (Offload) with Pipeline Parallelism (`AssertionError: ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism`).
2.  **Pydantic V2 & DeepSpeed**: DeepSpeed 0.17+ fails with Pydantic V2 if using client-side optimizer objects + Offload due to strict config validation (`Extra inputs not permitted`).
    *   *Solution*: Do **not** instantiate the optimizer in Python. Configure it entirely via the `ds_config` dictionary (type: "AdamW") so DeepSpeed builds its native `DeepSpeedCPUAdam` internally.
3.  **CUDA Mismatch**: System CUDA (13.0) vs PyTorch CUDA (12.8) prevents JIT compilation of CPU Adam.
    *   *Solution*: Set env var `DS_SKIP_CUDA_CHECK=1`.

## Next Steps (Next Session)
1.  **Architecture**: Design `train_dpgui_custom.py`.
    -   **Import**: Reuse `utils.dataset` and model loading from vendor.
    -   **Engine**: Use Standard DeepSpeed Engine (NOT PipelineEngine).
    -   **Loop**: Implement a standard `for batch in dataloader` training loop.
2.  **Implementation**: Write and test the script on the 5-GPU cluster.
3.  **Validation**: Verify ZeRO-2 Offload is active and memory is stable.