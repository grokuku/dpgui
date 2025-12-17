import asyncio
import json
import os
import time
import uuid
import glob
import re
import socket
from typing import List, Optional, Dict
from schemas import Job, JobStatus, TrainingConfig
from process_manager import ProcessManager

JOBS_DIR = "jobs"
LOGS_DIR = "logs"

class JobManager:
    def __init__(self):
        self.process_manager = ProcessManager()
        self.jobs: Dict[str, Job] = {}
        self.active_job_id: Optional[str] = None
        self._ensure_dirs()
        self._load_jobs()
        asyncio.create_task(self._worker_loop())

    def _ensure_dirs(self):
        os.makedirs(JOBS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)

    def _load_jobs(self):
        for filepath in glob.glob(os.path.join(JOBS_DIR, "*.json")):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    job = Job(**data)
                    # Si crash pendant run, on marque comme Failed
                    if job.status == JobStatus.RUNNING:
                        job.status = JobStatus.FAILED
                        job.error = "Server restart detected during execution"
                        self._save_job(job)
            except Exception as e:
                print(f"Error loading job {filepath}: {e}")

    def _save_job(self, job: Job):
        filepath = os.path.join(JOBS_DIR, f"{job.id}.json")
        with open(filepath, "w") as f:
            f.write(job.model_dump_json(indent=2))

    def create_job(self, config: TrainingConfig, command: str) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            config=config,
            command=command,
            status=JobStatus.STOPPED, 
            log_path=os.path.join(LOGS_DIR, f"{job_id}.log")
        )
        self.jobs[job_id] = job
        self._save_job(job)
        return job

    def list_jobs(self) -> List[Job]:
        return sorted(self.jobs.values(), key=lambda x: x.created_at, reverse=True)

    def queue_job(self, job_id: str):
        """Passe un job du Pool vers la Queue (Pending)"""
        job = self.jobs.get(job_id)
        if job and job.status in [JobStatus.STOPPED, JobStatus.FAILED, JobStatus.COMPLETED]:
            job.status = JobStatus.PENDING
            job.error = None 
            self._save_job(job)

    async def stop_job(self, job_id: str):
        """Arrête un job en cours ou le sort de la file d'attente"""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        if job.status == JobStatus.RUNNING and self.process_manager.is_running:
            await self.process_manager.stop()
        elif job.status == JobStatus.PENDING:
            job.status = JobStatus.STOPPED
            self._save_job(job)

    def delete_job(self, job_id: str):
        if job_id in self.jobs:
            try: os.remove(os.path.join(JOBS_DIR, f"{job_id}.json"))
            except: pass
            try: os.remove(os.path.join(LOGS_DIR, f"{job_id}.log"))
            except: pass
            del self.jobs[job_id]

    def read_log_file(self, job_id: str) -> str:
        """Lit le contenu du fichier de log"""
        job = self.jobs.get(job_id)
        if not job or not job.log_path or not os.path.exists(job.log_path):
            return "No log file available."
        
        try:
            with open(job.log_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading log file: {e}"

    def _fix_vendor_symlinks(self):
        """
        Répare ou crée les liens symboliques pour ComfyUI et HunyuanVideo dans vendor/libs.
        Cela permet d'importer 'comfy' et 'hyvideo' sans importer les dossiers conflictuels.
        """
        libs_dir = os.path.abspath("vendor/libs")
        if not os.path.exists(libs_dir):
            os.makedirs(libs_dir)

        # Liste des liens à créer : (nom_lien, chemin_source)
        links_config = [
            ("comfy", os.path.abspath("vendor/ComfyUI/comfy")),
            ("hyvideo", os.path.abspath("vendor/HunyuanVideo/hyvideo")),
            ("node_helpers.py", os.path.abspath("vendor/ComfyUI/node_helpers.py"))
        ]

        for link_name, source_path in links_config:
            link_path = os.path.join(libs_dir, link_name)
            
            # 1. Vérification et Nettoyage si nécessaire
            if os.path.lexists(link_path):
                try:
                    if not os.path.exists(link_path) or os.path.realpath(link_path) != source_path:
                        os.remove(link_path)
                    else:
                        continue 
                except OSError:
                    pass 
            
            # 2. Création
            if os.path.exists(source_path):
                try:
                    os.symlink(source_path, link_path)
                    print(f"[JobManager] Symlink created: {link_path} -> {source_path}")
                except Exception as e:
                    print(f"[JobManager] Failed to create symlink for {link_name}: {e}")

    def _patch_compatibility(self):
        """
        Applique des correctifs à la volée pour Python 3.11+.
        """
        # Patch pour utils/cache.py : sqlite3 autocommit (Python 3.12+)
        # Ce patch est conservé car indispensable et sans effet de bord sur ZeRO
        cache_file = os.path.abspath("vendor/diffusion-pipe/utils/cache.py")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # On cherche l'argument incompatible avec Python 3.11
                if ", autocommit=False" in content:
                    print("[JobManager] Patching utils/cache.py for Python 3.11 compatibility...")
                    new_content = content.replace(", autocommit=False", "")
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
            except Exception as e:
                print(f"[JobManager] Warning: Failed to patch cache.py: {e}")
        
        # NOTE: Le patch de train.py pour ZeRO a été SUPPRIMÉ ici.
        # On utilise désormais le fichier train.py propre (native pipeline parallelism).

    # --- PORT DIAGNOSTIC TOOLS ---
    def _is_port_free(self, port: int) -> bool:
        """Checks if a port is free by trying to bind to it."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return True
        except OSError:
            return False

    def _get_free_port(self) -> int:
        """Asks the OS for a free ephemeral port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    # -----------------------------

    async def _worker_loop(self):
        print("[JobManager] Worker loop started.")
        while True:
            # 1. Check Active
            if self.process_manager.is_running:
                await asyncio.sleep(1)
                continue
            
            # 2. Cleanup finished job
            if self.active_job_id:
                completed_job = self.jobs.get(self.active_job_id)
                if completed_job and completed_job.status == JobStatus.RUNNING:
                    completed_job.status = JobStatus.COMPLETED 
                    completed_job.ended_at = time.time()
                    self._save_job(completed_job)
                self.active_job_id = None

            # 3. Pick Next PENDING
            pending_jobs = sorted(
                [j for j in self.jobs.values() if j.status == JobStatus.PENDING],
                key=lambda x: x.created_at
            )
            
            if pending_jobs:
                next_job = pending_jobs[0]
                print(f"[JobManager] Starting job {next_job.id}...")
                
                next_job.status = JobStatus.RUNNING
                next_job.started_at = time.time()
                self._save_job(next_job)
                self.active_job_id = next_job.id
                
                os.makedirs(os.path.dirname(next_job.log_path), exist_ok=True)
                
                try:
                    work_dir = os.path.abspath("vendor/diffusion-pipe")
                    
                    # --- REPARATION ENVIRONNEMENT & CODE ---
                    self._fix_vendor_symlinks()
                    self._patch_compatibility() 
                    
                    # --- CONFIGURATION ENVIRONNEMENT ---
                    libs_path = os.path.abspath("vendor/libs")
                    
                    env_updates = {}
                    current_pythonpath = os.environ.get("PYTHONPATH", "")
                    
                    new_pythonpath = f"{libs_path}{os.pathsep}{current_pythonpath}"
                    env_updates["PYTHONPATH"] = new_pythonpath

                    # --- PORT MANAGEMENT ---
                    # 1. Extract intended port
                    port_match = re.search(r"--master_port=(\d+)", next_job.command)
                    target_port = int(port_match.group(1)) if port_match else 29500
                    
                    # 2. Check availability
                    is_free = self._is_port_free(target_port)
                    print(f"[JobManager] Diagnostic: Port {target_port} is {'FREE' if is_free else 'OCCUPIED'}.")
                    
                    # 3. Resolve conflict
                    final_port = target_port
                    if not is_free:
                        print(f"[JobManager] Port {target_port} is busy. Finding a new one...")
                        final_port = self._get_free_port()
                        # Update command string to match new port
                        if port_match:
                            next_job.command = next_job.command.replace(f"--master_port={target_port}", f"--master_port={final_port}")
                        else:
                            # Insert if missing (unlikely given config_gen, but safe)
                            next_job.command += f" --master_port={final_port}"
                        print(f"[JobManager] Switched to port {final_port}.")
                    
                    # 4. Enforce Env Vars
                    env_updates["MASTER_PORT"] = str(final_port)
                    env_updates["MASTER_ADDR"] = "127.0.0.1"
                    # -------------------------------------
                    
                    # --- LANCEMENT ---
                    await self.process_manager.start(
                        command=next_job.command, 
                        log_file_path=next_job.log_path,
                        cwd=work_dir,
                        env=env_updates
                    )
                except Exception as e:
                    print(f"FAILED TO START JOB: {e}")
                    next_job.status = JobStatus.FAILED
                    next_job.error = str(e)
                    self._save_job(next_job)
                    self.active_job_id = None
            
            await asyncio.sleep(2)

    async def get_log_stream(self):
        return None