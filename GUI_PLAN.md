# GUI Plan for diffusion-pipe

This document outlines all the configurable parameters for `diffusion-pipe`, extracted from `main_example.toml`, `dataset.toml`, and `train.py`. It is intended to be a blueprint for creating the graphical user interface.

## Main Configuration (`main_example.toml`)

This file contains the primary settings for a training run.

### Global Settings

| Parameter | Type | Example Value | Description / UI Element Suggestion |
|---|---|---|---|
| `output_dir` | String (Path) | `/data/training_runs` | File/Folder picker input. This is a mandatory field. |
| `dataset` | String (Path) | `examples/dataset.toml` | File picker input. Path to the dataset configuration file. Mandatory. |
| `epochs` | Integer | `1000` | Integer input field. |
| `max_steps` | Integer | `5000` | Integer input field (optional). |
| `micro_batch_size_per_gpu` | Integer or List of Lists | `1` or `[[512, 4], [1024, 1]]` | Can be a simple integer input or a more complex list editor for per-resolution batch sizes. |
| `image_micro_batch_size_per_gpu` | Integer | `4` | Integer input field (optional). For mixed video/image training. |
| `pipeline_stages` | Integer | `1` | Integer input field. |
| `gradient_accumulation_steps` | Integer | `1` | Integer input field. |
| `gradient_clipping` | Float | `1.0` | Float input field. |
| `warmup_steps` | Integer | `100` | Integer input field. |
| `force_constant_lr` | Float | `1e-5` | Float input field (optional). |
| `lr_scheduler` | String | `linear` | Dropdown with options: `constant`, `linear`. |
| `blocks_to_swap` | Integer | `20` | Integer input field (optional). For block swapping optimization. |
| `pseudo_huber_c` | Float | `0.5` | Float input field (optional). |

### Evaluation Settings

| Parameter | Type | Example Value | Description / UI Element Suggestion |
|---|---|---|---|
| `eval_datasets` | List of Objects | `[{name = '...', config = '...'}]` | A list editor where user can add multiple evaluation datasets, each with a `name` (text input) and a `config` file path (file picker). |
| `eval_every_n_epochs` | Integer | `1` | Integer input field. |
| `eval_every_n_steps` | Integer | `100` | Integer input field (optional). |
| `eval_every_n_examples` | Integer | `1000` | Integer input field (optional). |
| `eval_before_first_step` | Boolean | `true` | Checkbox. |
| `eval_micro_batch_size_per_gpu` | Integer | `1` | Integer input field. |
| `image_eval_micro_batch_size_per_gpu` | Integer | `4` | Integer input field (optional). |
| `eval_gradient_accumulation_steps` | Integer | `1` | Integer input field. |
| `disable_block_swap_for_eval` | Boolean | `true` | Checkbox (optional). |

### Miscellaneous Settings

| Parameter | Type | Example Value | Description / UI Element Suggestion |
|---|---|---|---|
| `save_every_n_epochs` | Integer | `2` | Integer input field. |
| `save_every_n_steps` | Integer | `100` | Integer input field (optional). |
| `save_every_n_examples` | Integer | `1000` | Integer input field (optional). |
| `checkpoint_every_n_epochs` | Integer | `1` | Integer input field (optional). |
| `checkpoint_every_n_minutes` | Integer | `120` | Integer input field (optional). |
| `activation_checkpointing` | Boolean or String | `true` or `'unsloth'` | Checkbox for `true`/`false`, maybe a dropdown for `true`/`unsloth`/`false`. |
| `reentrant_activation_checkpointing` | Boolean | `true` | Checkbox (optional). |
| `partition_method` | String | `parameters` | Dropdown with options: `parameters`, `manual`. |
| `partition_split` | List of Integers | `[10]` or `[10, 20]` | List editor for integers, visible when `partition_method` is `manual`. |
| `save_dtype` | String | `bfloat16` | Dropdown with common dtypes: `float32`, `float16`, `bfloat16`. |
| `caching_batch_size` | Integer | `1` | Integer input field. |
| `map_num_proc` | Integer | `32` | Integer input field (optional). |
| `compile` | Boolean | `true` | Checkbox for `torch.compile`. |
| `steps_per_print` | Integer | `1` | Integer input field. |
| `video_clip_mode` | String | `single_beginning` | Dropdown with options: `single_beginning`, `single_middle`. |
| `x_axis_examples` | Boolean | `true` | Checkbox. Changes Tensorboard/WandB x-axis from steps to examples. |

---

## Model Configuration (`[model]`)

This section is highly dynamic and depends on the selected model `type`.

### Model `type` (Selector)

A dropdown menu should allow the user to select one of the following model types. The choice will determine which subsequent fields are shown.

**Available Model Types (from `train.py`):**
- `flux`
- `ltx-video`
- `hunyuan-video`
- `sdxl`
- `cosmos`
- `lumina_2`
- `wan`
- `chroma`
- `hidream`
- `sd3`
- `cosmos_predict2`
- `omnigen2`
- `qwen_image`
- `hunyuan_image`
- `auraflow`

### Model Parameters (Example for `hunyuan-video`)

| Parameter | Type | Example Value | Description / UI Element Suggestion |
|---|---|---|---|
| `ckpt_path` | String (Path) | `/path/to/ckpts` | Folder picker (optional, alternative to separate paths). |
| `transformer_path`| String (Path) | `.../transformer.safetensors` | File picker. |
| `vae_path` | String (Path) | `.../vae.safetensors` | File picker. |
| `llm_path` | String (Path) | `.../llava-llama-3-8b...` | Folder picker. |
| `clip_path` | String (Path) | `.../clip-vit-large-patch14` | Folder picker. |
| `dtype` | String | `bfloat16` | Dropdown with dtypes: `float32`, `float16`, `bfloat16`. |
| `transformer_dtype`| String | `float8` | Dropdown with dtypes, including `float8`. |
| `timestep_sample_method`| String | `logit_normal` | Dropdown: `logit_normal`, `uniform`. |

*Note: Each model type will have its own set of specific parameters. The GUI should adapt to show the correct fields based on the selection.*

---

## Adapter Configuration (`[adapter]`)

This section is optional and used for fine-tuning methods like LoRA. The entire section can be enabled/disabled with a checkbox.

### Adapter Parameters (for `type = 'lora'`)

| Parameter | Type | Example Value | Description / UI Element Suggestion |
|---|---|---|---|
| `type` | String | `lora` | Currently only `lora` is implemented. Could be a dropdown for future expansion. |
| `rank` | Integer | `32` | Integer input field. |
| `dtype` | String | `bfloat16` | Dropdown with dtypes: `float32`, `float16`, `bfloat16`. |
| `init_from_existing` | String (Path) | `/path/to/lora/epoch50` | Folder picker (optional). |
| `fuse_adapters` | List of Objects | `[{path = '...', weight = 1.0}]` | List editor. Each item has a `path` (file picker) and `weight` (float input). Experimental. |

---

## Optimizer Configuration (`[optimizer]`)

This section is also dynamic based on the selected optimizer `type`.

### Optimizer `type` (Selector)

A dropdown should show common optimizers, with an option for a custom text input for any optimizer from the `pytorch-optimizer` library.

**Available Optimizer Types (from `train.py`):**
- `adamw`
- `adamw8bit`
- `adamw_optimi`
- `stableadamw`
- `sgd`
- `adamw8bitkahan`
- `offload`
- `automagic`
- `genericoptim`
- `Prodigy` (from `pytorch-optimizer`)
- (Custom text input)

### Optimizer Parameters (Common)

| Parameter | Type | Example Value | Description / UI Element Suggestion |
|---|---|---|---|
| `lr` | Float | `2e-5` | Float input field. |
| `betas` | List of Floats | `[0.9, 0.99]` | List editor for two float values. |
| `weight_decay`| Float | `0.01` | Float input field. |
| `eps` | Float | `1e-8` | Float input field. |

---

## Monitoring Configuration (`[monitoring]`)

This section is for enabling and configuring Weights & Biases. Can be enabled/disabled with a checkbox.

| Parameter | Type | Example Value | Description / UI Element Suggestion |
|---|---|---|---|
| `enable_wandb`| Boolean | `false` | Checkbox. |
| `wandb_api_key`| String (Secret) | ` ` | Password/secret input field. |
| `wandb_tracker_name`| String | ` ` | Text input field. |
| `wandb_run_name`| String | ` ` | Text input field. |

---

## Dataset Configuration (`dataset.toml`)

This file defines the data sources and processing. The GUI should allow creating/editing this file.

### General Dataset Settings

| Parameter | Type | Example Value | Description / UI Element Suggestion |
|---|---|---|---|
| `resolutions` | List of Integers/Lists | `[512]` or `[[1280, 720]]`| List editor for integers or `[width, height]` pairs. |
| `enable_ar_bucket` | Boolean | `true` | Checkbox for Aspect Ratio Bucketing. |
| `min_ar` | Float | `0.5` | Float input, visible if `enable_ar_bucket` is true. |
| `max_ar` | Float | `2.0` | Float input, visible if `enable_ar_bucket` is true. |
| `num_ar_buckets`| Integer | `7` | Integer input, visible if `enable_ar_bucket` is true. |
| `ar_buckets` | List of Floats/Lists | `[1.0, 1.5]` or `[[512, 512]]`| Alternative to `min/max/num` settings. List editor. |
| `frame_buckets`| List of Integers | `[1, 33]` | List editor for frame counts (for video). |
| `cache_shuffle_num` | Integer | `10` | Integer input (optional). |
| `cache_shuffle_delimiter` | String | `, ` | Text input, visible if `cache_shuffle_num` > 0. |

### Directory Blocks (`[[directory]]`)

The GUI needs a way to manage a list of data directories. Each entry in the list is a "directory block" with its own parameters.

| Parameter (per directory) | Type | Example Value | Description / UI Element Suggestion |
|---|---|---|---|
| `path` | String (Path) | `/home/anon/data/images/grayscale` | Folder picker. |
| `mask_path` | String (Path) | `/home/anon/data/images/grayscale/masks` | Folder picker (optional). |
| `num_repeats`| Integer | `1` | Integer input. |
| *Overrides* | | | The user should also be able to override settings like `ar_buckets`, `resolutions`, and `frame_buckets` on a per-directory basis. This could be an "Advanced" section for each directory entry. |
