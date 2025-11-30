from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from schemas import TrainingConfig, TrainingCommand
from config_gen import generate_toml_files, generate_deepspeed_command
from process_manager import ProcessManager
import os
import asyncio
import psutil
import shutil
import subprocess

app = FastAPI(
    title="DPGui Backend",
    description="Backend API for diffusion-pipe-gui",
    version="0.3.0"
)

# --- State ---
training_manager = ProcessManager()

# --- Configuration CORS ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utils ---
def get_gpu_stats():
    """Récupère les stats GPU via nvidia-smi si disponible."""
    if shutil.which("nvidia-smi") is None:
        return None
    
    try:
        # On demande: index, util.gpu, memory.total, memory.used
        # Format CSV sans header ni unités
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.total,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return None
            
        gpus = []
        for line in result.stdout.strip().split('\n'):
            idx, util, mem_total, mem_used = line.split(', ')
            gpus.append({
                "id": idx,
                "usage": float(util),
                "vram_total": float(mem_total),
                "vram_used": float(mem_used),
                "vram_percent": round((float(mem_used) / float(mem_total)) * 100, 1)
            })
        return gpus
    except Exception:
        return None

# --- Routes ---

@app.get("/")
async def read_root():
    return {
        "status": "active",
        "service": "dpgui-backend",
        "version": "0.3.0",
        "training_active": training_manager.is_running
    }

@app.get("/system-stats")
async def get_system_stats():
    """Renvoie l'utilisation CPU, RAM et GPU."""
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    gpu = get_gpu_stats()
    
    return {
        "cpu": cpu,
        "ram": {
            "total": mem.total,
            "used": mem.used,
            "percent": mem.percent
        },
        "gpu": gpu or [] # Liste vide si pas de GPU
    }

@app.post("/generate-config")
async def generate_config(config: TrainingConfig):
    try:
        output_path = os.path.abspath("generated_configs")
        main_toml_path = generate_toml_files(config, base_path=output_path)
        cmd = generate_deepspeed_command(main_toml_path, num_gpus=1)
        
        return {
            "status": "success",
            "message": "Configuration files generated successfully.",
            "files": {
                "main_config": main_toml_path,
                "dataset_config": os.path.join(output_path, "dataset.toml")
            },
            "command": cmd
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration generation failed: {str(e)}")

@app.post("/start-training")
async def start_training(cmd_data: TrainingCommand):
    try:
        work_dir = os.path.abspath("vendor/diffusion-pipe")
        if not os.path.isdir(work_dir):
             raise Exception(f"Working directory not found: {work_dir}")
        
        pid = await training_manager.start(cmd_data.command, cwd=work_dir)
        return {"status": "started", "pid": pid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/stop-training")
async def stop_training():
    await training_manager.stop()
    return {"status": "stopped"}

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    try:
        if not training_manager.is_running or not training_manager.process:
            await websocket.send_text("INFO: No training process running.")
            while not training_manager.is_running:
                await asyncio.sleep(1)
                try: await websocket.send_text("PING") 
                except: return

        async for line in training_manager.get_output_generator():
            await websocket.send_text(line)
        await websocket.send_text("INFO: Process finished.")
        await websocket.close()
    except WebSocketDisconnect:
        print("Client disconnected from logs.")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        try: await websocket.close()
        except: pass