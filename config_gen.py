import toml
import os
import shlex
import random
from schemas import TrainingConfig

def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.getcwd(), path))

def generate_toml_files(config: TrainingConfig, base_path: str = "generated_configs"):
    """
    Generates dataset.toml and main.toml based on the provided configuration.
    """
    
    # Ensure output directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # --- 1. Generate Dataset TOML ---
    dataset_toml_path = os.path.join(base_path, "dataset.toml")
    dataset_data = config.dataset_config.model_dump(exclude={"directories"})
    
    # Traitement des répertoires : résolution des chemins
    processed_directories = []
    for d in config.dataset_config.directories:
        dir_dict = d.model_dump()
        dir_dict["path"] = resolve_project_path(dir_dict["path"])
        if dir_dict.get("mask_path"):
            dir_dict["mask_path"] = resolve_project_path(dir_dict["mask_path"])
        processed_directories.append(dir_dict)

    dataset_data["directory"] = processed_directories
    
    # Remove None values
    dataset_data = {k: v for k, v in dataset_data.items() if v is not None}
    
    with open(dataset_toml_path, "w") as f:
        toml.dump(dataset_data, f)
        
    # --- 2. Generate Main TOML ---
    main_toml_path = os.path.join(base_path, "train.toml")
    
    main_data = config.model_dump(exclude={
        "dataset_config", "model", "adapter", "optimizer", "monitoring", "evaluation"
    })
    
    # Résolution du dossier de sortie
    main_data["output_dir"] = resolve_project_path(main_data["output_dir"])

    # Link the dataset config file (Always absolute)
    main_data["dataset"] = os.path.abspath(dataset_toml_path)
    
    # Résolution des chemins dans le modèle
    model_params = config.model.params.copy()
    for key, value in model_params.items():
        if ("path" in key or "file" in key) and isinstance(value, str):
            model_params[key] = resolve_project_path(value)

    main_data["model"] = model_params
    main_data["model"]["type"] = config.model.type
    main_data["model"]["dtype"] = config.model.dtype
    
    if config.adapter.enabled:
        main_data["adapter"] = config.adapter.model_dump(exclude={"enabled"})
        if main_data["adapter"].get("init_from_existing"):
            main_data["adapter"]["init_from_existing"] = resolve_project_path(main_data["adapter"]["init_from_existing"])
        main_data["adapter"] = {k: v for k, v in main_data["adapter"].items() if v is not None}
        
    main_data["optimizer"] = config.optimizer.model_dump()
    
    if config.monitoring.enable_wandb:
        main_data["monitoring"] = config.monitoring.model_dump(exclude={"enable_wandb"})
        main_data["monitoring"] = {k: v for k, v in main_data["monitoring"].items() if v is not None}

    # --- OPTIMISATION VRAM AUTOMATIQUE ---
    
    # 1. Force Gradient Checkpointing
    main_data["gradient_checkpointing"] = True
    
    # 2. ZeRO Optimization (STRATEGIC PIVOT: ENABLED)
    main_data["zero_optimization"] = {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "overlap_comm": True,
        "contiguous_gradients": True,
        # --- AJOUT CRITIQUE POUR FIXER L'ERREUR ---
        "zero_force_ds_cpu_optimizer": False 
        # ------------------------------------------
    }
    
    # --------------------------------------------------
    
    # Nettoyage global des None pour éviter les erreurs TOML
    main_data = {k: v for k, v in main_data.items() if v is not None}

    with open(main_toml_path, "w") as f:
        toml.dump(main_data, f)
        
    return os.path.abspath(main_toml_path)

def generate_deepspeed_command(main_config_path: str, num_gpus: int = 1) -> str:
    script_path = resolve_project_path("scripts/train_dpgui.py")
    safe_config_path = shlex.quote(main_config_path)
    safe_script_path = shlex.quote(script_path)
    port = random.randint(30000, 50000)
    cmd = f"deepspeed --num_gpus={num_gpus} --master_port={port} {safe_script_path} --config {safe_config_path}"
    return cmd