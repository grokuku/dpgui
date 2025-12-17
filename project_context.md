# Project Context: DPGui

## ⚠️ AI KNOWLEDGE WARNING
**Crucial Note for Development:** The AI model used for this project has a knowledge cutoff (early 2023) that conflicts with the current environment (Late 2025).
*   **Reality:** PyTorch Stable is **2.9.1**, CUDA is **12.8** or **13.0**.
*   **Rule:** Always verify versions via `pip check` or official docs.

## Current State
- **Phase**: Phase 5 (Training Execution) **STRATEGIC PIVOT**.
- **Environment**: 
    - **Stable**: Python 3.11 + PyTorch 2.9.1 + CUDA 12.8.
    - **Hardware**: 5 GPUs setup.
- **Status**: 
    - The "Vendor/Non-destructive" approach for `diffusion-pipe` is officially **abandoned**.
    - Native Pipeline Parallelism failed (defaulted to Data Parallelism -> OOM).
    - ZeRO integration failed on original script (Architecture conflict).

## Strategic Change (Dec 2025)
**Decision**: We will no longer treat `diffusion-pipe` as a read-only dependency. We will implement a **"Heavy Modification"** strategy.
**Goal**: We will take ownership of the training logic to ensure it fits consumer/prosumer hardware constraints (ZeRO + Model Parallelism).

## Next Steps (Next Session)
1.  **Refactor `train.py` (The "Hard Patch")**:
    - Completely rewrite the initialization sequence.
    - Instantiate the Optimizer **BEFORE** `deepspeed.initialize()`.
    - Pass the optimizer explicitly to DeepSpeed to enable ZeRO (Stage 1/2).
2.  **Config Update**:
    - Re-enable `zero_optimization` in `config_gen.py`.
    - Ensure `pipeline_stages` maps correctly to GPU count in the generated config.
3.  **Execution**: Validate the new custom script on the 5-GPU cluster.