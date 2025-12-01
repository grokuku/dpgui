from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from schemas import TrainingConfig, TrainingCommand, Job
from config_gen import generate_toml_files, generate_deepspeed_command
from job_manager import JobManager
import os
import asyncio
import psutil
import shutil
import subprocess
from typing import List

app = FastAPI(title="DPGui Backend", version="0.5.1")
job_manager = JobManager()

origins = ["http://localhost:5173", "http://127.0.0.1:5173", "*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Utils (GPU) ---
def get_gpu_stats():
    if shutil.which("nvidia-smi") is None: return None
    try:
        # Added: temperature.gpu, fan.speed, clocks.current.graphics, power.draw, power.limit
        cmd = [
            'nvidia-smi', 
            '--query-gpu=index,utilization.gpu,memory.total,memory.used,temperature.gpu,fan.speed,clocks.current.graphics,power.draw,power.limit', 
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0: return None
        gpus = []
        for line in result.stdout.strip().split('\n'):
            # Parse the CSV line safely
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 9: continue
            
            idx, util, mem_total, mem_used, temp, fan, clock, power_draw, power_limit = parts
            
            # Helper for safe float conversion
            def safe_float(val):
                try: return float(val)
                except: return 0.0

            gpus.append({
                "id": idx,
                "usage": safe_float(util),
                "vram_total": safe_float(mem_total),
                "vram_used": safe_float(mem_used),
                "vram_percent": round((safe_float(mem_used) / max(1, safe_float(mem_total))) * 100, 1),
                "temp": safe_float(temp),
                "fan": safe_float(fan),
                "clock": safe_float(clock),
                "power_draw": safe_float(power_draw),
                "power_limit": safe_float(power_limit)
            })
        return gpus
    except Exception as e:
        print(f"GPU Stats Error: {e}")
        return None

# --- Routes ---
@app.get("/")
async def read_root():
    return {"status": "active", "service": "dpgui-backend", "version": "0.5.1", "active_job": job_manager.active_job_id}

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

# --- Job Routes ---
@app.get("/jobs", response_model=List[Job])
async def list_jobs():
    return job_manager.list_jobs()

@app.post("/jobs", response_model=Job)
async def create_job(cmd_data: TrainingCommand):
    return job_manager.create_job(cmd_data.config, cmd_data.command)

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    job_manager.delete_job(job_id)
    return {"status": "deleted"}

@app.post("/jobs/{job_id}/queue")
async def queue_job(job_id: str):
    """Déplace un job du pool vers la queue."""
    job_manager.queue_job(job_id)
    return {"status": "queued"}

@app.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """Arrête ou sort de la queue."""
    await job_manager.stop_job(job_id)
    return {"status": "stopped"}

@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    """Récupère le contenu du fichier de log."""
    content = job_manager.read_log_file(job_id)
    return {"content": content}

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    try:
        generator = await job_manager.get_log_stream()
        if not generator:
             while True:
                await asyncio.sleep(1)
                try: await websocket.send_text("PING") 
                except: break
        else:
            async for line in generator:
                await websocket.send_text(line)
            await websocket.close()
    except: pass