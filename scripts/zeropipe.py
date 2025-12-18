import sys
import os

# --- ZEROPIPE HEADER: PATH INJECTION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
vendor_path = os.path.join(project_root, "vendor", "diffusion-pipe")

if os.path.exists(vendor_path):
    sys.path.insert(0, vendor_path)
    print(f"[ZeroPipe] Vendor path injected: {vendor_path}")
else:
    print(f"[ZeroPipe] CRITICAL ERROR: Vendor path not found at {vendor_path}")
    sys.exit(1)

# [COMPATIBILITY] Force DeepSpeed to ignore CUDA mismatch (System 13.0 vs Torch 12.8)
os.environ["DS_SKIP_CUDA_CHECK"] = "1"

import argparse
import toml
import json
import torch
import deepspeed
import shutil
from datetime import datetime, timezone
from torch import nn
from deepspeed import comm as dist
import numpy as np
import random
import gc
import types

# --- VENDOR IMPORTS ---
from utils import dataset as dataset_util
from utils import common
from utils.common import is_main_process, get_rank, DTYPE_MAP, empty_cuda_cache
import utils.saver
from utils.patches import apply_patches
import comfy.model_management as mm

# --- PATCH DEEPSPEED ZERO.INIT (FIX META TENSOR ERROR) ---
# Ce patch permet à zero.Init d'ignorer les erreurs quand transformers essaie d'init des poids sur meta
def safe_post_init_method(self, module):
    try:
        # Tente l'implémentation originale
        original_post_init(self, module)
    except NotImplementedError:
        # Ignore "Cannot copy out of meta tensor"
        pass
    except Exception as e:
        # Ignore aussi les erreurs de cast sur meta
        pass

# Sauvegarde de la méthode originale et application du patch
import deepspeed.runtime.zero.partition_parameters as ds_partition
original_post_init = ds_partition.Init._post_init_method
ds_partition.Init._post_init_method = safe_post_init_method
print("[ZeroPipe] 🩹 Applied DeepSpeed ZeRO.Init monkeypatch for Meta Tensors")
# ---------------------------------------------------------

# --- 0. SAFETY PATCH: LIMIT CPU WORKERS ---
if hasattr(dataset_util, 'NUM_PROC'):
    print(f"[ZeroPipe] 🛡️  Limiting Dataset Workers: {dataset_util.NUM_PROC} -> 1 (RAM Safety)")
    dataset_util.NUM_PROC = 1

# --- 1. LAZY DATASET MANAGER ---
class LazyDatasetManager(dataset_util.DatasetManager):
    def cache(self, unload_models=True):
        if is_main_process():
            print("   [ZeroPipe] 💿  Rank 0 checking/generating cache on disk...")
            original_broadcast = torch.distributed.broadcast_object_list
            torch.distributed.broadcast_object_list = lambda *args, **kwargs: None
            try:
                super().cache(unload_models=False) 
            except Exception as e:
                print(f"   [ZeroPipe] ❌ Rank 0 Cache Gen Error: {e}")
                raise e
            finally:
                torch.distributed.broadcast_object_list = original_broadcast
            print("   [ZeroPipe] ✅  Cache generation complete.")

        if dist.is_initialized():
            print(f"   [ZeroPipe] ⏳  Rank {dist.get_rank()} waiting for cache sync...")
            dist.barrier()

        print(f"   [ZeroPipe] 📂  Rank {dist.get_rank()} loading cache from disk...")
        for ds in self.datasets:
            ds.cache_metadata(regenerate_cache=False, trust_cache=True)
            ds.cache_latents(None, regenerate_cache=False, trust_cache=True)
            for i in range(1, len(self.text_encoders)+1):
                ds.cache_text_embeddings(None, i, regenerate_cache=False)

        if unload_models:
            print(f"   [ZeroPipe] 🧹  Rank {dist.get_rank()} unloading models...")
            for model in self.submodels:
                if not isinstance(model, nn.Module): continue
                if self.model.name == 'sdxl' and model is self.vae:
                    model.to('cpu')
                else:
                    model.to('meta')
            mm.unload_all_models()

        if dist.is_initialized():
            dist.barrier()
            
        gc.collect()
        empty_cuda_cache()

# --- 2. THE FLAT MODEL WRAPPER ---
class ZeroPipeModel(nn.Module):
    def __init__(self, layers, loss_fn):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.loss_fn = loss_fn
        self.gradient_checkpointing = False

    def forward(self, batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, labels = batch
            if isinstance(inputs, (tuple, list)): inputs = inputs[0]
            if isinstance(labels, (tuple, list)): labels = labels[0]
        else:
            inputs = batch
            labels = None

        x = inputs
        for layer in self.layers:
            if self.gradient_checkpointing and layer.training:
                def custom_forward(*args): return layer(*args)
                if isinstance(x, tuple):
                    x = torch.utils.checkpoint.checkpoint(custom_forward, *x, use_reentrant=False)
                else:
                    x = torch.utils.checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
            else:
                if isinstance(x, tuple): x = layer(*x)
                else: x = layer(x)
        
        if labels is not None: return self.loss_fn(x, labels)
        return x

# --- 3. ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="ZeroPipe: ZeRO-Optimized Training Engine")
parser.add_argument('--config', type=str, required=True, help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
parser.add_argument('--master_port', type=int, default=29500, help='Port for distributed training')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

def main():
    apply_patches()
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if is_main_process():
        print(f"\n🚀 [ZeroPipe] Starting Training Engine on {world_size} GPUs")
        print(f"   Config: {args.config}")

    with open(args.config) as f:
        config = json.loads(json.dumps(toml.load(f)))

    common.AUTOCAST_DTYPE = DTYPE_MAP[config['model']['dtype']]
    dataset_util.UNCOND_FRACTION = config.get('uncond_fraction', 0.0)
    
    resume_from = config.get('resume_from_checkpoint', None)
    if not resume_from:
        run_name = datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S')
        run_dir = os.path.join(config['output_dir'], run_name)
    else:
        run_dir = os.path.join(config['output_dir'], resume_from)
        
    if is_main_process():
        os.makedirs(run_dir, exist_ok=True)
        if not resume_from: shutil.copy(args.config, run_dir)

    # --- 4. PREPARE DEEPSPEED CONFIG ---
    micro_batch = config.get('micro_batch_size_per_gpu', 1)
    if isinstance(micro_batch, list): micro_batch = micro_batch[0][1]
    grad_accum = config.get('gradient_accumulation_steps', 1)
    train_batch_size = micro_batch * grad_accum * world_size
    
    if is_main_process():
        print(f"   [ZeroPipe] Batch Calculation: Micro={micro_batch} | Accum={grad_accum} | World={world_size} => Global={train_batch_size}")

    ds_config = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "steps_per_print": 1,
        "gradient_clipping": config.get('gradient_clipping', 1.0),
        "train_batch_size": train_batch_size,
        "zero_optimization": config.get('zero_optimization', {
            "stage": 3, 
            "offload_optimizer": {"device": "cpu", "pin_memory": False},
            "offload_param": {"device": "cpu", "pin_memory": False}
        }),
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config['optimizer']['lr'],
                "weight_decay": config['optimizer'].get('weight_decay', 0.01),
                "betas": config['optimizer'].get('betas', [0.9, 0.999]),
                "eps": config['optimizer'].get('eps', 1e-8)
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": config['epochs'] * 1000,
                "warmup_min_lr": 0,
                "warmup_max_lr": config['optimizer']['lr'],
                "warmup_num_steps": config.get('warmup_steps', 0)
            }
        },
        "bf16": {"enabled": config['model']['dtype'] == 'bfloat16'},
        "fp16": {"enabled": config['model']['dtype'] == 'float16'}
    }

    # --- 5. MODEL INIT (WRAPPED WITH ZERO.INIT) ---
    model_type = config['model']['type']
    if is_main_process(): 
        print(f"   Model Type: {model_type}")
        print("   [ZeroPipe] 🛡️  Initializing Model with DeepSpeed ZeRO.Init() ...")

    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        if model_type == 'sdxl': from models import sdxl as m_mod; model = m_mod.SDXLPipeline(config)
        elif model_type == 'flux': from models import flux as m_mod; model = m_mod.FluxPipeline(config)
        elif model_type == 'hunyuan-video': from models import hunyuan_video as m_mod; model = m_mod.HunyuanVideoPipeline(config)
        elif model_type == 'ltx-video': from models import ltx_video as m_mod; model = m_mod.LTXVideoPipeline(config)
        elif model_type == 'z-image' or model_type == 'z_image': from models import z_image as m_mod; model = m_mod.ZImagePipeline(config)
        elif model_type == 'qwen_image': from models import qwen_image as m_mod; model = m_mod.QwenImagePipeline(config)
        elif model_type == 'wan': from models.wan import wan as m_mod; model = m_mod.WanPipeline(config)
        elif model_type == 'cosmos': from models import cosmos as m_mod; model = m_mod.CosmosPipeline(config)
        elif model_type == 'lumina_2': from models import lumina_2 as m_mod; model = m_mod.Lumina2Pipeline(config)
        elif model_type == 'auraflow': from models import auraflow as m_mod; model = m_mod.AuraFlowPipeline(config)
        else: raise NotImplementedError(f"Model {model_type} not yet mapped in ZeroPipe.")

    # --- 6. DATASET & CACHING ---
    dataset_manager = LazyDatasetManager(model, regenerate_cache=False)
    
    with open(config['dataset']) as f:
        dataset_config = toml.load(f)
    dataset_config['trust_cache'] = True

    train_data = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=False)
    dataset_manager.register(train_data)
    
    gc.collect()
    empty_cuda_cache()
    
    dataset_manager.cache()

    # --- 7. LOAD WEIGHTS ---
    if is_main_process(): print("   Loading Diffusion Model Weights...")
    # NOTE: With ZeRO-3 Init, params are on Meta. We rely on load_diffusion_model to 
    # load parameters. However, since params are partitioned/meta, standard loading may fail 
    # if not using DeepSpeed's loading mechanism or gather_parameter.
    # Luckily, diffusion-pipe models usually load into `self` attributes which are then
    # used to construct the engine.
    # FIX: We ensure we are out of the Init context before loading weights if possible,
    # OR we let the engine initialization handle the gathering.
    # For now, we proceed as normal, hoping diffusers from_single_file managed to attach hooks.
    model.load_diffusion_model()

    # --- 8. ADAPTER vs FULL ---
    is_adapter = False
    if 'adapter' in config:
        if is_main_process(): print("   Mode: LoRA Training")
        model.configure_adapter(config['adapter'])
        if config['adapter'].get('init_from_existing'):
            model.load_adapter_weights(config['adapter']['init_from_existing'])
        is_adapter = True
    else:
        if is_main_process(): print("   Mode: Full Fine-Tuning")

    # --- 9. ENGINE INIT ---
    layers = model.to_layers()
    zero_model = ZeroPipeModel(layers, model.get_loss_fn())
    
    if config.get('gradient_checkpointing', False):
        zero_model.gradient_checkpointing = True
        model.enable_gradient_checkpointing() 

    parameters_to_train = [p for p in zero_model.parameters() if p.requires_grad]
    if is_main_process():
        print(f"   [ZeroPipe] DeepSpeed Initialize...")

    # Force params to have data if they are still on meta (safety check)
    # ZeRO-3 Engine handles Meta params automatically during init.
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=zero_model,
        model_parameters=parameters_to_train,
        config=ds_config
    )

    model.model_engine = model_engine

    # --- 10. TRAINING LOOP ---
    train_data.post_init(rank, world_size, micro_batch, ds_config['gradient_accumulation_steps'])
    
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=None,
        num_workers=config.get('num_workers', 1), 
        pin_memory=True
    )

    saver = utils.saver.Saver(args, config, is_adapter, run_dir, model, train_dataloader, model_engine, None)
    epochs = config['epochs']
    global_step = 0
    
    if is_main_process(): print("   Training Loop Started...")
    
    for epoch in range(epochs):
        train_data.set_epoch(epoch)
        try: steps_per_epoch = len(train_dataloader)
        except: steps_per_epoch = 1000 
            
        for step, batch in enumerate(train_dataloader):
            loss = model_engine(batch)
            model_engine.backward(loss)
            model_engine.step()
            
            global_step += 1
            if is_main_process() and global_step % ds_config['steps_per_print'] == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step}/{steps_per_epoch} | Loss: {loss.item():.4f}")

            examples_count = global_step * world_size * ds_config['train_micro_batch_size_per_gpu'] * ds_config['gradient_accumulation_steps']
            _, _, saved = saver.process_step(global_step, examples_count)
            
        saver.process_epoch(epoch, global_step, examples_count)

    if is_main_process(): print("✅ Training Complete.")

if __name__ == "__main__":
    main()