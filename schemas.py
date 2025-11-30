from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
from enum import Enum
import time

# --- Enums ---
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

# --- Job Models ---
class JobBase(BaseModel):
    # Ce qui est nécessaire pour créer un job
    config: 'TrainingConfig' 
    command: str

class Job(JobBase):
    id: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[str] = None
    log_path: Optional[str] = None

# --- Dataset Section ---

class DirectoryBlock(BaseModel):
    path: str
    mask_path: Optional[str] = None
    num_repeats: int = 1
    resolutions: Optional[List[Union[int, List[int]]]] = None
    ar_buckets: Optional[List[Union[float, List[float]]]] = None
    frame_buckets: Optional[List[int]] = None

class DatasetConfig(BaseModel):
    resolutions: List[Union[int, List[int]]] = [512]
    enable_ar_bucket: bool = True
    min_ar: Optional[float] = 0.5
    max_ar: Optional[float] = 2.0
    num_ar_buckets: Optional[int] = 7
    ar_buckets: Optional[List[Union[float, List[float]]]] = None
    frame_buckets: Optional[List[int]] = None
    cache_shuffle_num: Optional[int] = 10
    cache_shuffle_delimiter: Optional[str] = ", "
    directories: List[DirectoryBlock] = []

# --- Main Configuration Section ---

class ModelConfig(BaseModel):
    type: str
    params: Dict[str, Any] = {} 

class AdapterConfig(BaseModel):
    enabled: bool = False
    type: str = "lora"
    rank: int = 32
    dtype: str = "bfloat16"
    init_from_existing: Optional[str] = None

class OptimizerConfig(BaseModel):
    type: str = "adamw"
    lr: float = 2e-5
    betas: List[float] = [0.9, 0.99]
    weight_decay: float = 0.01
    eps: float = 1e-8

class MonitoringConfig(BaseModel):
    enable_wandb: bool = False
    wandb_api_key: Optional[str] = None
    wandb_tracker_name: Optional[str] = None
    wandb_run_name: Optional[str] = None

class EvaluationConfig(BaseModel):
    eval_every_n_epochs: int = 1
    eval_every_n_steps: Optional[int] = None
    eval_before_first_step: bool = True
    eval_micro_batch_size_per_gpu: int = 1
    eval_gradient_accumulation_steps: int = 1
    eval_datasets: List[Dict[str, str]] = []

class TrainingConfig(BaseModel):
    output_dir: str
    epochs: int = 1000
    max_steps: Optional[int] = 5000
    micro_batch_size_per_gpu: Union[int, List[List[int]]] = 1
    pipeline_stages: int = 1
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    warmup_steps: int = 100
    lr_scheduler: str = "linear"
    save_dtype: str = "bfloat16"
    compile: bool = True
    
    dataset_config: DatasetConfig
    model: ModelConfig
    adapter: AdapterConfig
    optimizer: OptimizerConfig
    monitoring: MonitoringConfig
    evaluation: EvaluationConfig

class TrainingCommand(BaseModel):
    command: str
    config: TrainingConfig

# FIX PYDANTIC V2: model_rebuild() au lieu de update_forward_refs()
JobBase.model_rebuild()
Job.model_rebuild()