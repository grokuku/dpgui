import toml
import os
import shlex
from schemas import TrainingConfig

def resolve_project_path(path: str) -> str:
    """
    Si le chemin est absolu, on le garde.
    Si le chemin est relatif (ex: 'output', 'data/images'), on le rend absolu
    par rapport à la racine du projet (là où tourne le backend).
    """
    if os.path.isabs(path):
        return path
    # os.getcwd() retourne la racine du projet car main.py est lancé depuis là
    return os.path.abspath(os.path.join(os.getcwd(), path))

def generate_toml_files(config: TrainingConfig, base_path: str = "generated_configs"):
    """
    Generates dataset.toml and main.toml based on the provided configuration.
    Returns the path to the main config file.
    """
    
    # Ensure output directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # --- 1. Generate Dataset TOML ---
    dataset_toml_path = os.path.join(base_path, "dataset.toml")
    dataset_data = config.dataset_config.dict(exclude={"directories"})
    
    # Traitement des répertoires : résolution des chemins
    processed_directories = []
    for d in config.dataset_config.directories:
        dir_dict = d.dict()
        # On force le chemin absolu pour 'path'
        dir_dict["path"] = resolve_project_path(dir_dict["path"])
        # On force le chemin absolu pour 'mask_path' s'il existe
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
    
    main_data = config.dict(exclude={
        "dataset_config", "model", "adapter", "optimizer", "monitoring", "evaluation"
    })
    
    # Résolution du dossier de sortie
    main_data["output_dir"] = resolve_project_path(main_data["output_dir"])

    # Link the dataset config file (Always absolute)
    main_data["dataset"] = os.path.abspath(dataset_toml_path)
    
    # Résolution des chemins dans le modèle (ckpt, vae, etc)
    # On parcourt les params du modèle pour voir s'il y a des chemins à résoudre
    model_params = config.model.params.copy()
    for key, value in model_params.items():
        # Si la clé contient "path" ou "file" et que la valeur est une string, on tente de résoudre
        if ("path" in key or "file" in key) and isinstance(value, str):
            model_params[key] = resolve_project_path(value)

    main_data["model"] = model_params
    main_data["model"]["type"] = config.model.type
    
    if config.adapter.enabled:
        main_data["adapter"] = config.adapter.dict(exclude={"enabled"})
        # Resolve paths in adapter if needed
        if main_data["adapter"].get("init_from_existing"):
            main_data["adapter"]["init_from_existing"] = resolve_project_path(main_data["adapter"]["init_from_existing"])
        # Clean None values
        main_data["adapter"] = {k: v for k, v in main_data["adapter"].items() if v is not None}
        
    main_data["optimizer"] = config.optimizer.dict()
    
    if config.monitoring.enable_wandb:
        main_data["monitoring"] = config.monitoring.dict(exclude={"enable_wandb"})
        main_data["monitoring"] = {k: v for k, v in main_data["monitoring"].items() if v is not None}

    main_data.update(config.evaluation.dict())
    main_data = {k: v for k, v in main_data.items() if v is not None}

    with open(main_toml_path, "w") as f:
        toml.dump(main_data, f)
        
    return os.path.abspath(main_toml_path)

def generate_deepspeed_command(main_config_path: str, num_gpus: int = 1) -> str:
    """
    Generates the command line string to launch training.
    """
    # Since we are executing inside the vendor/diffusion-pipe directory,
    # we point directly to train.py without the folder prefix.
    script_path = "train.py" 
    
    # Escaping paths for safety
    safe_config_path = shlex.quote(main_config_path)
    safe_script_path = shlex.quote(script_path)
    
    cmd = f"deepspeed --num_gpus={num_gpus} {safe_script_path} --config {safe_config_path}"
    return cmd