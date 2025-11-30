import asyncio
import json
import os
import time
import uuid
import glob
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
                    self.jobs[job.id] = job
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
            status=JobStatus.STOPPED, # PAR DEFAUT : Dans le Pool (pas la queue)
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
            job.error = None # Reset error on retry
            self._save_job(job)

    async def stop_job(self, job_id: str):
        """Arrête un job en cours ou le sort de la file d'attente"""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        if job.status == JobStatus.RUNNING and self.process_manager.is_running:
            await self.process_manager.stop()
            # Le worker mettra à jour le statut
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
                
                # S'assurer que le dossier logs existe
                os.makedirs(os.path.dirname(next_job.log_path), exist_ok=True)
                
                # On redirige stdout/stderr vers le fichier log
                # Pour le live stream, le ProcessManager devra lire ce fichier (TODO future improvement)
                # Pour l'instant on garde le ProcessManager tel quel (pipe) et on écrit manuellement dans le fichier ?
                # NON : ProcessManager capture stdout. On va écrire ce stdout dans le fichier à la volée.
                # Pour simplifier cette phase, on laisse ProcessManager gérer le flux live, 
                # et on ne peut pas vraiment écrire dans le fichier facilement sans modifier ProcessManager.
                # FIX RAPIDE: On va modifier ProcessManager pour supporter le 'tee' (écriture fichier) ?
                # Trop risqué pour maintenant. 
                # ALTERNATIVE : On redirige la sortie de la commande shell vers un fichier ET stdout.
                
                cmd_with_log = f"{next_job.command} 2>&1 | tee {next_job.log_path}"
                
                try:
                    work_dir = os.path.abspath("vendor/diffusion-pipe")
                    await self.process_manager.start(cmd_with_log, cwd=work_dir)
                except Exception as e:
                    next_job.status = JobStatus.FAILED
                    next_job.error = str(e)
                    self._save_job(next_job)
                    self.active_job_id = None
            
            await asyncio.sleep(2)

    async def get_log_stream(self):
        return self.process_manager.get_output_generator()