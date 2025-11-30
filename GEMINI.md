# diffusion-pipe-gui (dpgui) Project Overview

This project aims to provide a graphical user interface (GUI) to simplify the usage of `diffusion-pipe`, a powerful Python-based deep learning project for training diffusion models. The original `diffusion-pipe` (located in `ref/diffusion-pipe`) is configured through manual editing of TOML files, which can be complex and error-prone. The `dpgui` project's main goal is to streamline the creation of these configuration files and the launching of training processes, making `diffusion-pipe` more accessible to a wider range of users.

## diffusion-pipe Project (Reference)

`diffusion-pipe` is a deep learning framework designed for training various diffusion models, including those for image and video generation. It heavily utilizes distributed training capabilities through DeepSpeed and leverages the Hugging Face ecosystem for models and pipelines.

### Key Technologies in `diffusion-pipe`:

*   **Python:** The primary programming language.
*   **PyTorch:** The underlying deep learning framework.
*   **DeepSpeed:** For efficient distributed training, memory optimization, and pipeline parallelism.
*   **Hugging Face Transformers/Diffusers/PEFT:** For accessing pre-trained models, diffusion pipelines, and parameter-efficient fine-tuning (LoRA).
*   **TOML:** Used extensively for managing training, model, optimizer, and dataset configurations.
*   **argparse:** For command-line argument parsing.
*   **Weights & Biases (WandB), TensorBoard:** For experiment tracking and visualization.
*   **bitsandbytes:** For 8-bit optimizers and quantization.

### Configuration Structure in `diffusion-pipe`:

`diffusion-pipe` is highly configurable via TOML files. There are two primary types of configuration files:

1.  **Main Training Configuration (e.g., `examples/main_example.toml`):**
    This file defines global training parameters such as:
    *   Output directory (`output_dir`)
    *   Paths to dataset configurations (`dataset`, `eval_datasets`)
    *   Training hyper-parameters (`epochs`, `micro_batch_size_per_gpu`, `learning rate settings`)
    *   Evaluation frequency and settings
    *   Saving and checkpointing frequencies
    *   Model-specific parameters (`[model]` table), where the `type` field dynamically dictates other available settings (e.g., `hunyuan-video`, `flux`, `sdxl`).
    *   Adapter settings (`[adapter]` table) for techniques like LoRA.
    *   Optimizer settings (`[optimizer]` table), including `type`, learning rate, betas, etc.
    *   Monitoring settings (`[monitoring]` table) for WandB integration.

2.  **Dataset Configuration (e.g., `examples/dataset.toml`):**
    This file specifies how the training data is prepared and accessed:
    *   Resolution and aspect ratio bucketing settings (`resolutions`, `enable_ar_bucket`, `min_ar`, `max_ar`, `num_ar_buckets`, `ar_buckets`).
    *   Frame bucketing for video training (`frame_buckets`).
    *   Specification of data directories using `[[directory]]` table arrays, allowing for multiple data sources, each with its `path`, optional `mask_path`, and `num_repeats`.

### Building and Running `diffusion-pipe`:

To set up and run the `diffusion-pipe` training:

1.  **Install Dependencies:**
    Navigate to `ref/diffusion-pipe` and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Training:**
    Training is typically launched using the `deepspeed` command, pointing to the `train.py` script and a main configuration file.
    ```bash
    deepspeed --num_gpus=<NUM_GPUS> ref/diffusion-pipe/train.py --config <PATH_TO_YOUR_CONFIG.toml>
    ```
    *Replace `<NUM_GPUS>` with the number of GPUs you want to use, and `<PATH_TO_YOUR_CONFIG.toml>` with the path to your main training configuration file (e.g., `ref/diffusion-pipe/examples/main_example.toml`).*

### Development Conventions in `diffusion-pipe`:

*   **Configuration-driven:** All major aspects of training are controlled through external TOML configuration files.
*   **Modular Model Implementation:** Different diffusion models are implemented as separate modules (e.g., `models/flux.py`, `models/hunyuan_video.py`), which are dynamically loaded.
*   **DeepSpeed Integration:** DeepSpeed is deeply integrated for managing distributed training, pipeline parallelism, and low-level optimizations.
*   **Clear Argument Parsing:** `argparse` is used for a well-defined command-line interface.
*   **Utilities:** A `utils` directory provides common functionalities like dataset management, saving, and RNG isolation.

## dpgui Project - Next Steps

The `dpgui` project will need to:
1.  **Parse Configuration Schemas:** Dynamically read and understand the structure of `main_example.toml` and `dataset.toml` to generate appropriate GUI input fields.
2.  **Build a Dynamic GUI:** Create a user interface (likely using Dear PyGui) that allows users to:
    *   Select model types, optimizers, and other top-level parameters.
    *   Define and add multiple dataset directories.
    *   Input values for all configurable parameters, with appropriate validation.
    *   Generate a valid TOML configuration file based on user input.
    *   Provide a way to launch the `deepspeed` training command with the generated configuration.
3.  **Integrate with `diffusion-pipe` Execution:** Execute the `deepspeed` command with the generated configuration file.
4.  **Consider Advanced Features:** (Future) Live monitoring of training progress (e.g., by parsing TensorBoard logs or integrating with WandB).

This `GEMINI.md` file will serve as a foundational document for understanding the `diffusion-pipe` reference project and guiding the development of the `dpgui` interface.