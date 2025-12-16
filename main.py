from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from schemas import TrainingConfig, TrainingCommand, Job
from config_gen import generate_toml_files, generate_deepspeed_command
from job_manager import JobManager
import dataset_utils
import os
import asyncio
import psutil
import shutil
import subprocess
import threading
import time
from typing import List, Optional
from huggingface_hub import snapshot_download

app = FastAPI(title="DPGui Backend", version="0.8.0")
job_manager = JobManager()

origins = ["http://localhost:5173", "http://127.0.0.1:5173", "*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
    expose_headers=["*"]
)

# --- GPU Stats ---
def get_gpu_stats():
    if shutil.which("nvidia-smi") is None: return None
    try:
        cmd = [
            'nvidia-smi', 
            '--query-gpu=index,utilization.gpu,memory.total,memory.used,temperature.gpu,fan.speed,clocks.current.graphics,power.draw,power.limit', 
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0: return None
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 9: continue
            idx, util, mem_total, mem_used, temp, fan, clock, power_draw, power_limit = parts
            def safe_float(val):
                try: return float(val)
                except: return 0.0
            gpus.append({
                "id": idx, "usage": safe_float(util), "vram_total": safe_float(mem_total),
                "vram_used": safe_float(mem_used), "vram_percent": round((safe_float(mem_used) / max(1, safe_float(mem_total))) * 100, 1),
                "temp": safe_float(temp), "fan": safe_float(fan), "clock": safe_float(clock),
                "power_draw": safe_float(power_draw), "power_limit": safe_float(power_limit)
            })
        return gpus
    except Exception as e:
        print(f"GPU Stats Error: {e}")
        return None

# --- MODEL DOWNLOAD MANAGER ---

class DownloadRequest(BaseModel):
    repo_id: str
    filename: Optional[str] = None
    hf_token: Optional[str] = None

# Global state simple pour le suivi
download_state = {
    "status": "idle", # idle, downloading, completed, error
    "current_repo": None,
    "local_path": None,
    "downloaded_size": "0 MB",
    "error": None
}

def get_folder_size_mb(path):
    """Calcule la taille d'un dossier en MB."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except Exception:
        pass
    return total_size / (1024 * 1024)

def _monitor_download(target_dir, stop_event):
    """Thread qui surveille la taille du dossier."""
    while not stop_event.is_set():
        size_mb = get_folder_size_mb(target_dir)
        if size_mb > 1024:
            download_state["downloaded_size"] = f"{size_mb/1024:.2f} GB"
        else:
            download_state["downloaded_size"] = f"{size_mb:.2f} MB"
        time.sleep(1)

def _background_download_model(req: DownloadRequest):
    global download_state
    download_state["status"] = "downloading"
    download_state["current_repo"] = req.repo_id
    download_state["error"] = None
    download_state["local_path"] = None
    download_state["downloaded_size"] = "0 MB"
    
    # Event pour arrêter le monitoring
    stop_monitor = threading.Event()
    
    try:
        # Dossier de destination : models/Createur_RepoName
        safe_name = req.repo_id.replace("/", "_")
        target_dir = os.path.abspath(os.path.join("models", safe_name))
        
        print(f"[ModelManager] Starting download: {req.repo_id} -> {target_dir}")
        
        # Lancer le moniteur
        monitor_thread = threading.Thread(target=_monitor_download, args=(target_dir, stop_monitor))
        monitor_thread.start()
        
        local_path = snapshot_download(
            repo_id=req.repo_id,
            local_dir=target_dir,
            token=req.hf_token if req.hf_token else None,
            allow_patterns=req.filename if req.filename else None,
            # ignore_patterns=["*.msgpack", "*.bin"] 
        )
        
        download_state["status"] = "completed"
        download_state["local_path"] = local_path
        print(f"[ModelManager] Download finished: {local_path}")
        
    except Exception as e:
        download_state["status"] = "error"
        download_state["error"] = str(e)
        print(f"[ModelManager] Download failed: {e}")
    finally:
        stop_monitor.set()

@app.post("/models/download")
async def start_model_download(req: DownloadRequest, background_tasks: BackgroundTasks):
    if download_state["status"] == "downloading":
        raise HTTPException(status_code=400, detail="A download is already in progress")
    
    background_tasks.add_task(_background_download_model, req)
    return {"status": "started", "repo_id": req.repo_id}

@app.get("/models/status")
async def get_download_status():
    return download_state

@app.get("/models")
async def list_local_models():
    """Returns a list of folder names in the models/ directory."""
    if not os.path.exists("models"):
        return []
    return [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]


# --- General Routes ---
@app.get("/")
async def read_root():
    return {"status": "active", "service": "dpgui-backend", "version": "0.8.0", "active_job": job_manager.active_job_id}

@app.get("/system-stats")
async def get_system_stats():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    return {"cpu": cpu, "ram": {"total": mem.total, "used": mem.used, "percent": mem.percent}, "gpu": get_gpu_stats() or []}

@app.post("/generate-config")
async def generate_config(config: TrainingConfig):
    try:
        output_path = os.path.abspath("generated_configs")
        main_toml_path = generate_toml_files(config, base_path=output_path)
        cmd = generate_deepspeed_command(main_toml_path, num_gpus=1)
        return {"status": "success", "files": {"main_config": main_toml_path}, "command": cmd}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# --- DATASET MANAGEMENT API ---

class DatasetAction(BaseModel):
    name: str
    new_name: Optional[str] = None

class BatchAction(BaseModel):
    dataset: str
    images: List[str] 
    action: str 
    payload: Optional[str] = None 

class DeleteImagesRequest(BaseModel):
    dataset: str
    images: List[str]

class CaptionRequest(BaseModel):
    image_path: str 
    content: str

@app.get("/datasets")
async def get_datasets_list():
    return dataset_utils.list_datasets()

@app.post("/datasets/create")
async def create_dataset_endpoint(data: DatasetAction):
    if dataset_utils.create_dataset(data.name):
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Dataset already exists or invalid name")

@app.post("/datasets/delete")
async def delete_dataset_endpoint(data: DatasetAction):
    if dataset_utils.delete_dataset(data.name):
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Failed to delete")

@app.post("/datasets/rename")
async def rename_dataset_endpoint(data: DatasetAction):
    if not data.new_name: raise HTTPException(status_code=400, detail="New name missing")
    if dataset_utils.rename_dataset(data.name, data.new_name):
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Failed to rename")

@app.post("/datasets/clone")
async def clone_dataset_endpoint(data: DatasetAction):
    if not data.new_name: raise HTTPException(status_code=400, detail="New name missing")
    if dataset_utils.clone_dataset(data.name, data.new_name):
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Failed to clone")

@app.post("/datasets/delete_images")
async def delete_images_endpoint(req: DeleteImagesRequest):
    count = dataset_utils.delete_dataset_images(req.dataset, req.images)
    return {"deleted": count}

@app.get("/datasets/{name}/export")
async def export_dataset(name: str):
    zip_path = dataset_utils.zip_dataset(name)
    if not zip_path:
        raise HTTPException(status_code=404, detail="Dataset not found or failed to zip")
    
    def cleanup():
        try: os.remove(zip_path)
        except: pass
        
    return FileResponse(zip_path, filename=f"{name}.zip", background=BackgroundTask(cleanup))

@app.get("/datasets/{name}/images")
async def get_dataset_images(name: str):
    return dataset_utils.list_images(name)

@app.get("/datasets/thumbnail")
async def get_thumbnail(path: str):
    data = dataset_utils.get_thumbnail(path)
    if data:
        return Response(content=data, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/datasets/image_raw")
async def get_raw_image(path: str):
    full_path = os.path.join(dataset_utils.DATA_ROOT, path)
    if not dataset_utils.is_safe_path(full_path) or not os.path.exists(full_path):
         raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(full_path)

@app.get("/datasets/caption")
async def get_caption(path: str):
    content = dataset_utils.read_caption(path)
    return {"content": content}

@app.post("/datasets/caption")
async def update_caption(req: CaptionRequest):
    if dataset_utils.save_caption(req.image_path, req.content):
        return {"status": "success"}
    raise HTTPException(status_code=500, detail="Failed to save caption")

@app.post("/datasets/batch")
async def batch_operation(req: BatchAction):
    if req.action == "trigger_word":
        count = dataset_utils.batch_add_trigger(req.dataset, req.images, req.payload, position="start")
        return {"processed": count}
    elif req.action == "resize":
        size = int(req.payload) if req.payload else 1024
        count = dataset_utils.batch_resize_images(req.dataset, req.images, resolution=size)
        return {"processed": count}
    elif req.action == "autotag":
        return {"processed": 0, "message": "Auto-tagging requires external model (Coming in Phase 4.1)"}
    
    raise HTTPException(status_code=400, detail="Unknown action")

@app.post("/datasets/{name}/upload")
async def upload_images(name: str, files: List[UploadFile] = File(...)):
    count = 0
    for file in files:
        content = await file.read()
        if dataset_utils.save_image(name, content, file.filename):
            count += 1
    return {"uploaded": count}

# --- Job Routes ---
@app.get("/jobs", response_model=List[Job])
async def list_jobs(): return job_manager.list_jobs()

@app.post("/jobs", response_model=Job)
async def create_job(cmd_data: TrainingCommand): return job_manager.create_job(cmd_data.config, cmd_data.command)

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    job_manager.delete_job(job_id)
    return {"status": "deleted"}

@app.post("/jobs/{job_id}/queue")
async def queue_job(job_id: str):
    job_manager.queue_job(job_id)
    return {"status": "queued"}

@app.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    await job_manager.stop_job(job_id)
    return {"status": "stopped"}

@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    content = job_manager.read_log_file(job_id)
    return {"content": content}

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    try:
        # On enregistre le socket auprès du ProcessManager qui lui poussera les logs
        await job_manager.process_manager.register_websocket(websocket)
        
        # On garde la connexion ouverte pour recevoir le flux
        while True:
            # On attend juste un message pour garder la connexion vivante ou détecter la fermeture
            data = await websocket.receive_text()
            if data == "PING":
                pass
    except WebSocketDisconnect:
        job_manager.process_manager._websockets.discard(websocket)
    except Exception as e:
        print(f"WebSocket Error: {e}")
        job_manager.process_manager._websockets.discard(websocket)