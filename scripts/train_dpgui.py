import sys
import os

# --- DPGUI HEADER: PATH INJECTION & ENV SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
vendor_path = os.path.join(project_root, "vendor", "diffusion-pipe")

if os.path.exists(vendor_path):
    sys.path.insert(0, vendor_path)
    print(f"[DPGui] Injected vendor path: {vendor_path}")
else:
    print(f"[DPGui] WARNING: Vendor path not found at {vendor_path}")

# [CRITICAL FIX] CUDA Version Mismatch Bypass
# System CUDA (13.0) != PyTorch CUDA (12.8).
# We force DeepSpeed to compile its JIT kernels anyway.
os.environ["DS_SKIP_CUDA_CHECK"] = "1"
print("[DPGui] Env Var set: DS_SKIP_CUDA_CHECK=1")
# ------------------------------------------------

import argparse
import wandb
from datetime import datetime, timezone
import shutil
import glob
import time
import random
import json
import inspect
from pathlib import Path
from collections import defaultdict

import toml
import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.pipe import module as ds_pipe_module
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocess as mp
import numpy as np

from utils import dataset as dataset_util
from utils import common
from utils.common import is_main_process, get_rank, DTYPE_MAP, empty_cuda_cache
import utils.saver
from utils.isolate_rng import isolate_rng
from utils.patches import apply_patches
from utils.unsloth_utils import unsloth_checkpoint
from utils.pipeline import ManualPipelineModule

# needed for broadcasting Queue in dataset.py
mp.current_process().authkey = b'afsaskgfdjh4'

wandb_enable = False

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', nargs='?', const=True, default=None,
                    help='resume training from checkpoint. If no value is provided, resume from the most recent checkpoint. If a folder name is provided, resume from that specific folder.')
parser.add_argument('--reset_dataloader', action='store_true', help='Start dataloader from scratch when resuming from checkpoint, i.e. only load the optimizer states.')
parser.add_argument('--reset_optimizer', action='store_true')
parser.add_argument('--reset_optimizer_params', action='store_true')
parser.add_argument('--regenerate_cache', action='store_true', help='Force regenerate cache.')
parser.add_argument('--cache_only', action='store_true', help='Cache model inputs then exit.')
parser.add_argument('--trust_cache', action='store_true', help='Load from metadata cache files if they exist, without checking if any fingerprints have changed. Can make loading much faster for large datasets.')
parser.add_argument('--i_know_what_i_am_doing', action='store_true', help="Skip certain checks and overrides. You may end up using settings that won't work.")
parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
parser.add_argument('--dump_dataset', type=Path, default=None, help='Decode cached latents and dump the dataset to this directory.')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts
ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


# --- DPGUI FIX: SINGLE STAGE MODEL WRAPPER ---
# This class wraps the layers in a standard nn.Module to avoid triggering
# DeepSpeed's PipelineEngine when using ZeRO-2 (which are incompatible).
class SingleStageModel(nn.Module):
    def __init__(self, layers, loss_fn):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.loss_fn = loss_fn
        
    def forward(self, batch):
        # DeepSpeed PipelineDataLoader yields ((inputs,), (labels,)) tuples
        # We need to unwrap them for standard sequential execution
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, labels = batch
            # Unwrap inner tuples if present (Pipeline style: ((x,), (y,)))
            if isinstance(inputs, (tuple, list)): inputs = inputs[0]
            if isinstance(labels, (tuple, list)): labels = labels[0]
        else:
            inputs = batch
            labels = None

        x = inputs
        # Execute layers sequentially
        for layer in self.layers:
            if isinstance(x, tuple):
                x = layer(*x)
            else:
                x = layer(x)
        
        if labels is not None:
            return self.loss_fn(x, labels)
        return x


def set_config_defaults(config):
    # Force the user to set this. If we made it a default of 1, it might use a lot of disk space.
    assert 'save_every_n_epochs' in config or 'save_every_n_steps' in config or 'save_every_n_examples' in config

    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config.setdefault('reentrant_activation_checkpointing', False)
    if config['activation_checkpointing'] == 'unsloth':
        config['reentrant_activation_checkpointing'] = True
    config.setdefault('warmup_steps', 0)
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := model_config.get('transformer_dtype', None):
        model_config['transformer_dtype'] = DTYPE_MAP[transformer_dtype]
    if diffusion_model_dtype := model_config.get('diffusion_model_dtype', None):
        model_config['diffusion_model_dtype'] = DTYPE_MAP[diffusion_model_dtype]
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if adapter_config['type'] == 'lora':
            if 'alpha' in adapter_config:
                raise NotImplementedError(
                    'This script forces alpha=rank to make the saved LoRA format simpler and more predictable with downstream inference programs. Please remove alpha from the config.'
                )
            adapter_config['alpha'] = adapter_config['rank']
            adapter_config.setdefault('dropout', 0.0)
            adapter_config.setdefault('dtype', model_dtype_str)
            adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    config.setdefault('logging_steps', 1)
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('eval_every_n_steps', None)
    config.setdefault('eval_every_n_epochs', None)
    config.setdefault('eval_every_n_examples', None)
    config.setdefault('eval_before_first_step', True)
    config.setdefault('compile', False)
    config.setdefault('x_axis_examples', False)


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


# Need to preload all micro batches since pulling from the dataloader does IPC between the
# first and last stage.
def get_data_iterator_for_step(dataloader, engine, num_micro_batches=None):
    # DPGUI: Standard Engine compatibility (no micro_batches attr on standard engine)
    num_micro_batches = num_micro_batches or getattr(engine, 'micro_batches', 1)
    
    # Only check stages if we are actually using the Pipeline Engine
    if hasattr(engine, 'is_first_stage'):
        if not (engine.is_first_stage() or engine.is_last_stage()):
            return None
            
    dataloader_iter = iter(dataloader)
    items = [next(dataloader_iter) for _ in range(num_micro_batches)]
    return iter(items)


def evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None):
    eval_dataloader.set_eval_quantile(quantile)
    total_loss = 0
    count = 0
    while True:
        # Standard engine polyfill for reset_activation_shape
        if hasattr(model_engine, 'reset_activation_shape'):
            model_engine.reset_activation_shape()
            
        iterator = get_data_iterator_for_step(eval_dataloader, model_engine, num_micro_batches=eval_gradient_accumulation_steps)
        
        # DPGUI: Compatibility for Standard Engine (doesn't have eval_batch)
        if hasattr(model_engine, 'eval_batch'):
            loss = model_engine.eval_batch(iterator, num_micro_batches=eval_gradient_accumulation_steps).item()
        else:
            # Manual evaluation loop for Standard Engine
            loss_acc = 0.0
            steps = eval_gradient_accumulation_steps
            model_engine.eval()
            with torch.no_grad():
                for _ in range(steps):
                    batch = next(iterator)
                    loss_item = model_engine.module(batch).item() # SingleStageModel forward returns loss
                    loss_acc += loss_item
            model_engine.train()
            loss = loss_acc / steps

        eval_dataloader.sync_epoch()
        if pbar:
            pbar.update(1)
        total_loss += loss
        count += 1
        if eval_dataloader.epoch == 2:
            break

    eval_dataloader.reset()
    return total_loss / count


def _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    pbar_total = 0
    for eval_dataloader in eval_dataloaders.values():
        pbar_total += len(eval_dataloader) * len(TIMESTEP_QUANTILES_FOR_EVAL) // eval_gradient_accumulation_steps
    if is_main_process():
        print('Running eval')
        pbar = tqdm(total=pbar_total)
    else:
        pbar = None

    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        losses = []
        for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
            loss = evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=pbar)
            losses.append(loss)
            if is_main_process():
                tb_writer.add_scalar(f'{name}/loss_quantile_{quantile:.2f}', loss, step)
                if wandb_enable:
                    wandb.log({f'{name}/loss_quantile_{quantile:.2f}': loss, 'step': step})
        avg_loss = sum(losses) / len(losses)
        if is_main_process():
            tb_writer.add_scalar(f'{name}/loss', avg_loss, step)
            if wandb_enable:
                wandb.log({f'{name}/loss': avg_loss, 'step': step})

    duration = time.time() - start
    if is_main_process():
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)
        if wandb_enable:
            wandb.log({'eval/eval_time_sec': duration, 'step': step})
        pbar.close()


def evaluate(model, model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps, disable_block_swap):
    if len(eval_dataloaders) == 0:
        return
    empty_cuda_cache()
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
    with torch.no_grad(), isolate_rng():
        seed = get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)
    empty_cuda_cache()
    model.prepare_block_swap_training()


def distributed_init(args):
    """Initialize distributed training environment."""
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = args.local_rank

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(args.master_port)

    return world_size, rank, local_rank


def get_prodigy_d(optimizer):
    d = 0
    for group in optimizer.param_groups:
        d += group['d']
    return d / len(optimizer.param_groups)


def _get_automagic_lrs(optimizer):
    lrs = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            lr = optimizer._get_lr(group, state)
            lrs.append(lr)
    lrs = torch.stack(lrs)
    return lrs, lrs.mean()


if __name__ == '__main__':
    apply_patches()

    with open(args.config) as f:
        # Inline TOML tables are not pickleable, which messes up the multiprocessing dataset stuff. This is a workaround.
        config = json.loads(json.dumps(toml.load(f)))

    set_config_defaults(config)
    common.AUTOCAST_DTYPE = config['model']['dtype']
    dataset_util.UNCOND_FRACTION = config.get('uncond_fraction', 0.0)
    if map_num_proc := config.get('map_num_proc', None):
        dataset_util.NUM_PROC = map_num_proc

    # Initialize distributed environment before deepspeed
    world_size, rank, local_rank = distributed_init(args)

    # Now initialize deepspeed
    deepspeed.init_distributed()

    # needed for broadcasting Queue in dataset.py
    torch.cuda.set_device(dist.get_rank())

    resume_from_checkpoint = (
        args.resume_from_checkpoint if args.resume_from_checkpoint is not None
        else config.get('resume_from_checkpoint', False)
    )
    regenerate_cache = (
        args.regenerate_cache if args.regenerate_cache is not None
        else config.get('regenerate_cache', False)
    )

    model_type = config['model']['type']

    # --- MODEL LOADING (unchanged) ---
    if model_type == 'flux':
        from models import flux
        model = flux.FluxPipeline(config)
    elif model_type == 'ltx-video':
        from models import ltx_video
        model = ltx_video.LTXVideoPipeline(config)
    elif model_type == 'hunyuan-video':
        from models import hunyuan_video
        model = hunyuan_video.HunyuanVideoPipeline(config)
    elif model_type == 'sdxl':
        from models import sdxl
        model = sdxl.SDXLPipeline(config)
    elif model_type == 'cosmos':
        from models import cosmos
        model = cosmos.CosmosPipeline(config)
    elif model_type == 'lumina_2':
        from models import lumina_2
        model = lumina_2.Lumina2Pipeline(config)
    elif model_type == 'wan':
        from models.wan import wan
        model = wan.WanPipeline(config)
    elif model_type == 'chroma':
        from models import chroma
        model = chroma.ChromaPipeline(config)
    elif model_type == 'hidream':
        from models import hidream
        model = hidream.HiDreamPipeline(config)
    elif model_type == 'sd3':
        from models import sd3
        model = sd3.SD3Pipeline(config)
    elif model_type == 'cosmos_predict2':
        from models import cosmos_predict2
        model = cosmos_predict2.CosmosPredict2Pipeline(config)
    elif model_type == 'omnigen2':
        from models import omnigen2
        model = omnigen2.OmniGen2Pipeline(config)
    elif model_type == 'qwen_image':
        from models import qwen_image
        model = qwen_image.QwenImagePipeline(config)
    elif model_type == 'hunyuan_image':
        from models import hunyuan_image
        model = hunyuan_image.HunyuanImagePipeline(config)
    elif model_type == 'auraflow':
        from models import auraflow
        model = auraflow.AuraFlowPipeline(config)
    elif model_type == 'z_image':
        from models import z_image
        model = z_image.ZImagePipeline(config)
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented')

    with open(config['dataset']) as f:
        dataset_config = toml.load(f)

    micro_batch_size_per_gpu = config.get('micro_batch_size_per_gpu', 1)
    if isinstance(micro_batch_size_per_gpu, int):
        micro_batch_size_per_gpu = {None: micro_batch_size_per_gpu}
    elif isinstance(micro_batch_size_per_gpu, list):
        micro_batch_size_per_gpu = {x[0]: x[1] for x in micro_batch_size_per_gpu}

    eval_micro_batch_size_per_gpu = config.get('eval_micro_batch_size_per_gpu', micro_batch_size_per_gpu)
    if isinstance(eval_micro_batch_size_per_gpu, int):
        eval_micro_batch_size_per_gpu = {None: eval_micro_batch_size_per_gpu}
    elif isinstance(eval_micro_batch_size_per_gpu, list):
        eval_micro_batch_size_per_gpu = {x[0]: x[1] for x in eval_micro_batch_size_per_gpu}

    image_micro_batch_size_per_gpu = config.get('image_micro_batch_size_per_gpu', micro_batch_size_per_gpu)
    if isinstance(image_micro_batch_size_per_gpu, int):
        image_micro_batch_size_per_gpu = {None: image_micro_batch_size_per_gpu}
    elif isinstance(image_micro_batch_size_per_gpu, list):
        image_micro_batch_size_per_gpu = {x[0]: x[1] for x in image_micro_batch_size_per_gpu}

    eval_image_micro_batch_size_per_gpu = config.get('eval_image_micro_batch_size_per_gpu', eval_micro_batch_size_per_gpu)
    if isinstance(eval_image_micro_batch_size_per_gpu, int):
        eval_image_micro_batch_size_per_gpu = {None: eval_image_micro_batch_size_per_gpu}
    elif isinstance(eval_image_micro_batch_size_per_gpu, list):
        eval_image_micro_batch_size_per_gpu = {x[0]: x[1] for x in eval_image_micro_batch_size_per_gpu}

    default_micro_batch_size_per_gpu = list(micro_batch_size_per_gpu.values())[0]

    gradient_release = config['optimizer'].get('gradient_release', False)
    ds_config = {
        'train_micro_batch_size_per_gpu': default_micro_batch_size_per_gpu,
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
        'gradient_clipping': 0. if gradient_release else config.get('gradient_clipping', 1.0),
        'steps_per_print': config.get('steps_per_print', 1),
    }

    # --- DPGUI FIX: ZeRO Configuration ---
    if 'zero_optimization' in config:
        ds_config['zero_optimization'] = config['zero_optimization']
        # Clean config for native optimizer usage
        if 'zero_force_ds_cpu_optimizer' in ds_config['zero_optimization']:
            del ds_config['zero_optimization']['zero_force_ds_cpu_optimizer']

    # --- DPGUI FIX: Native DeepSpeed Optimizer Setup ---
    optim_config = config.get('optimizer', {})
    lr = optim_config.get('lr', 1e-4) 
    ds_config['optimizer'] = {
        "type": "AdamW", 
        "params": {
            "lr": lr,
            "betas": optim_config.get('betas', [0.9, 0.999]),
            "eps": optim_config.get('eps', 1e-8),
            "weight_decay": optim_config.get('weight_decay', 1e-2)
        }
    }

    caching_batch_size = config.get('caching_batch_size', 1)
    dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=regenerate_cache, trust_cache=args.trust_cache, caching_batch_size=caching_batch_size)

    train_data = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
    dataset_manager.register(train_data)

    eval_data_map = {}
    for i, eval_dataset in enumerate(config['eval_datasets']):
        if type(eval_dataset) == str:
            name = f'eval{i}'
            config_path = eval_dataset
        else:
            name = eval_dataset['name']
            config_path = eval_dataset['config']
        with open(config_path) as f:
            eval_dataset_config = toml.load(f)
        eval_data_map[name] = dataset_util.Dataset(eval_dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
        dataset_manager.register(eval_data_map[name])

    if args.dump_dataset:
        import torchvision
        dataset_manager.cache(unload_models=False)
        if is_main_process():
            with torch.no_grad():
                os.makedirs(args.dump_dataset, exist_ok=True)
                vae = model.vae.to('cuda')
                train_data.post_init(0, 1, 1, 1, 1)
                for i, item in enumerate(train_data):
                    latents = item['latents']
                    latents = latents / vae.config.scaling_factor
                    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                        latents = latents + vae.config.shift_factor
                    img = vae.decode(latents.to(vae.device, vae.dtype)).sample.to(torch.float32)
                    img = img.squeeze(0)
                    img = ((img + 1) / 2).clamp(0, 1)
                    pil_img = torchvision.transforms.functional.to_pil_image(img)
                    pil_img.save(args.dump_dataset / f'{i}.png')
                    if i >= 100:
                        break
        dist.barrier()
        quit()

    dataset_manager.cache()
    if args.cache_only:
        quit()

    model.load_diffusion_model()

    if adapter_config := config.get('adapter', None):
        model.configure_adapter(adapter_config)
        is_adapter = True
        if init_from_existing := adapter_config.get('init_from_existing', None):
            model.load_adapter_weights(init_from_existing)
    else:
        is_adapter = False

    if not resume_from_checkpoint and is_main_process():
        run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
        shutil.copy(config['dataset'], run_dir)
        for eval_dataset in config['eval_datasets']:
            shutil.copy(eval_dataset['config'], run_dir)
    dist.barrier()
    if resume_from_checkpoint is True:
        run_dir = get_most_recent_run_dir(config['output_dir'])
    elif isinstance(resume_from_checkpoint, str):
        run_dir = os.path.join(config['output_dir'], resume_from_checkpoint)
        if not os.path.exists(run_dir):
            raise ValueError(f"Checkpoint directory {run_dir} does not exist")
    else:
        run_dir = get_most_recent_run_dir(config['output_dir'])

    wandb_enable = config.get('monitoring', {}).get('enable_wandb', False)
    if wandb_enable and is_main_process():
        wandb_api_key     = config['monitoring']['wandb_api_key']
        wandb_tracker     = config['monitoring']['wandb_tracker_name']
        wandb_run_name    = config['monitoring']['wandb_run_name']
        logging_dir       = run_dir
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_tracker,
            name=wandb_run_name,
            config=config,
            dir=logging_dir
        )

    if blocks_to_swap := config.get('blocks_to_swap', 0):
        assert config['pipeline_stages'] == 1, 'Block swapping only works with pipeline_stages=1'
        assert 'adapter' in config, 'Block swapping only works when training LoRA'
        def to(self, *args, **kwargs):
            pass
        deepspeed.pipe.PipelineModule.to = to
        model.enable_block_swap(blocks_to_swap)

    layers = model.to_layers()
    additional_pipeline_module_kwargs = {}
    activation_checkpointing = config['activation_checkpointing']
    if activation_checkpointing:
        if activation_checkpointing == True:
            from functools import partial
            checkpoint_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=config['reentrant_activation_checkpointing'])
        elif activation_checkpointing == 'unsloth':
            checkpoint_func = unsloth_checkpoint
        else:
            raise NotImplementedError(f'activation_checkpointing={activation_checkpointing} is not implemented')
        additional_pipeline_module_kwargs.update({
            'activation_checkpoint_interval': 1,
            'checkpointable_layers': model.checkpointable_layers,
            'activation_checkpoint_func': checkpoint_func,
        })

    num_stages = config.get('pipeline_stages', 1)
    pp_world_size = num_stages
    dp_world_size = world_size // num_stages
    if dp_world_size < 1: dp_world_size = 1
    global_batch_size = ds_config['train_micro_batch_size_per_gpu'] * ds_config['gradient_accumulation_steps'] * dp_world_size
    print(f'[DPGui] Calculated global_batch_size = {global_batch_size}')

    # --- DPGUI FIX: Select Correct Engine Strategy ---
    # We strictly check if we are in a Pipeline Parallelism scenario.
    # If num_stages is 1, we MUST use the Standard Engine (SingleStageModel) to support ZeRO-2.
    if num_stages > 1:
        print("[DPGui] Using Pipeline Parallelism Engine (ZeRO-2 disabled)")
        partition_method=config.get('partition_method', 'parameters')
        partition_split = config.get('partition_split',[len(layers) / num_stages])
        final_model = ManualPipelineModule(
            layers=layers,
            num_stages=num_stages,
            partition_method=partition_method,
            manual_partition_split=partition_split,
            loss_fn=model.get_loss_fn(),
            **additional_pipeline_module_kwargs
        )
        if config['compile']: final_model.compile()
    else:
        print("[DPGui] Using Standard Engine (ZeRO-2/3 Compatible)")
        final_model = SingleStageModel(layers, model.get_loss_fn())

    parameters_to_train = [p for p in final_model.parameters() if p.requires_grad]

    print("[DPGui] Initializing DeepSpeed...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=final_model, 
        optimizer=None, # Use config defined optimizer
        model_parameters=parameters_to_train,
        config=ds_config,
    )

    # --- DPGUI FIX: ENGINE POLYFILLS ---
    # Since we might be using the Standard Engine which lacks 'train_batch', we polyfill it.
    if num_stages == 1:
        # Polyfill train_batch to emulate pipeline behavior (acc steps + step)
        def train_batch_polyfill(data_iter):
            accum_loss = 0.0
            steps = model_engine.gradient_accumulation_steps()
            for _ in range(steps):
                batch = next(data_iter)
                loss = model_engine(batch)
                model_engine.backward(loss)
                model_engine.step()
                accum_loss += loss.item()
            return torch.tensor(accum_loss / steps)
            
        def reset_activation_shape_polyfill():
            pass
            
        # Attach methods dynamically to the engine instance
        model_engine.train_batch = train_batch_polyfill
        model_engine.reset_activation_shape = reset_activation_shape_polyfill
    # -----------------------------------

    model.model_engine = model_engine
    if model_engine.is_pipe_parallel:
         grid = model_engine.grid
         model_engine.first_last_stage_group = dist.new_group(ranks=[grid.pp_group[0], grid.pp_group[-1]])

    train_data.post_init(
        model_engine.grid.get_data_parallel_rank(),
        model_engine.grid.get_data_parallel_world_size(),
        micro_batch_size_per_gpu,
        model_engine.gradient_accumulation_steps(),
        image_micro_batch_size_per_gpu,
    )
    for eval_data in eval_data_map.values():
        eval_data.post_init(
            model_engine.grid.get_data_parallel_rank(),
            model_engine.grid.get_data_parallel_world_size(),
            eval_micro_batch_size_per_gpu,
            config['eval_gradient_accumulation_steps'],
            eval_image_micro_batch_size_per_gpu,
        )

    communication_data_type = config['lora']['dtype'] if 'lora' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    train_dataloader = dataset_util.PipelineDataLoader(train_data, model_engine, model_engine.gradient_accumulation_steps(), model)
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()

    scheduler_type = config.get('lr_scheduler', 'constant')
    if scheduler_type == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=config['epochs'] * steps_per_epoch)
    else:
        raise NotImplementedError(f'Unknown lr_scheduler: {scheduler_type}')
    
    if config['warmup_steps'] > 0:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])
    model_engine.lr_scheduler = lr_scheduler

    step = 1
    examples = global_batch_size
    
    if resume_from_checkpoint:
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            load_module_strict=False,
            load_lr_scheduler_states='force_constant_lr' not in config and not args.reset_optimizer and not args.reset_optimizer_params,
            load_optimizer_states=not args.reset_optimizer,
        )
        dist.barrier()
        assert load_path is not None
        if args.reset_dataloader:
            train_dataloader.epoch = client_state['custom_loader']['epoch']
        else:
            train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1
        if 'examples' in client_state:
            examples = client_state['examples'] + global_batch_size
        else:
            examples = step * global_batch_size
        del client_state
        if is_main_process():
            print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')

    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']

    eval_dataloaders = {
        name: dataset_util.PipelineDataLoader(eval_data, model_engine, config['eval_gradient_accumulation_steps'], model, num_dataloader_workers=0)
        for name, eval_data in eval_data_map.items()
    }

    epoch = train_dataloader.epoch
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
    saver = utils.saver.Saver(args, config, is_adapter, run_dir, model, train_dataloader, model_engine, pipeline_model)

    disable_block_swap_for_eval = config.get('disable_block_swap_for_eval', False)
    if config['eval_before_first_step'] and not resume_from_checkpoint:
        evaluate(model, model_engine, eval_dataloaders, tb_writer, 0, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

    epoch_loss = 0
    num_steps = 0
    empty_cuda_cache()
    while True:
        # Ensure compatibility with Standard Engine which might lack this method
        if hasattr(model_engine, 'reset_activation_shape'):
            model_engine.reset_activation_shape()
            
        iterator = get_data_iterator_for_step(train_dataloader, model_engine)
        loss = model_engine.train_batch(iterator).item()
        epoch_loss += loss
        num_steps += 1
        train_dataloader.sync_epoch()

        new_epoch, checkpointed, saved = saver.process_epoch(epoch, step, examples)
        finished_epoch = True if new_epoch != epoch else False

        x_axis = examples if config['x_axis_examples'] else step

        if is_main_process() and step % config['logging_steps'] == 0:
            tb_writer.add_scalar(f'train/loss', loss, x_axis)
            if wandb_enable:
                wandb.log({'train/loss': loss, 'step': x_axis})

        if (config['eval_every_n_steps'] and step % config['eval_every_n_steps'] == 0) or (finished_epoch and config['eval_every_n_epochs'] and epoch % config['eval_every_n_epochs'] == 0):
            evaluate(model, model_engine, eval_dataloaders, tb_writer, x_axis, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

        if finished_epoch:
            if is_main_process():
                tb_writer.add_scalar(f'train/epoch_loss', epoch_loss/num_steps, epoch)
                if wandb_enable:
                    wandb.log({'train/epoch_loss': epoch_loss/num_steps, 'epoch': epoch})
            epoch_loss = 0
            num_steps = 0
            if new_epoch is None:
                final_model_name = f'epoch{epoch}'
                break
            epoch = new_epoch

        checkpointed, saved = saver.process_step(step, examples)
        if 'max_steps' in config and step >= config['max_steps']:
            final_model_name = f'step{step}'
            break
        step += 1
        examples += global_batch_size

    if not checkpointed:
        saver.save_checkpoint(step, examples)
    if not saved:
        saver.save_model(final_model_name)

    if is_main_process():
        print('TRAINING COMPLETE!')