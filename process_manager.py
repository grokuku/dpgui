import asyncio
import os
import signal
from typing import Optional, Set
from collections import deque

class ProcessManager:
    def __init__(self):
        self.process = None
        self.is_running = False
        self._log_task = None
        # Tampon circulaire pour garder les derniers logs en mémoire (1000 lignes)
        self._recent_logs = deque(maxlen=1000)
        self._websockets = set()
        self._log_file_handle = None

    async def start(self, command: str, log_file_path: str, cwd: Optional[str] = None, env: Optional[dict] = None):
        """
        Lance la commande et démarre la capture des logs en arrière-plan.
        """
        if self.is_running and self.process:
            if self.process.returncode is None:
                raise Exception("A training process is already running.")

        # Préparation de l'environnement : Force le non-buffering Python
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        run_env["PYTHONUNBUFFERED"] = "1"

        # Ouverture du fichier de log
        # MODIF: mode "w" pour écraser (vider) les anciens logs au démarrage
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self._log_file_handle = open(log_file_path, "w", encoding="utf-8")
        
        # Reset logs mémoire pour le nouveau job
        self._recent_logs.clear()

        # Lancement du processus
        self.process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, 
            preexec_fn=os.setsid,
            cwd=cwd,
            env=run_env
        )
        self.is_running = True
        
        # Démarrage de la tâche de consommation des logs
        self._log_task = asyncio.create_task(self._consume_logs())
        
        return self.process.pid

    async def stop(self):
        """
        Arrête le processus proprement.
        """
        if self.process and self.is_running:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            
            self.is_running = False
            
            if self._log_task:
                try:
                    await self._log_task
                except asyncio.CancelledError:
                    pass
                
            self._close_log_file()
            self.process = None

    def _close_log_file(self):
        if self._log_file_handle:
            try:
                self._log_file_handle.close()
            except:
                pass
            self._log_file_handle = None

    async def _consume_logs(self):
        """
        Lit le flux stdout du processus en continu, écrit dans le fichier
        et diffuse aux WebSockets connectés.
        """
        if not self.process:
            return

        try:
            while True:
                line_bytes = await self.process.stdout.readline()
                if not line_bytes:
                    break
                
                line = line_bytes.decode('utf-8', errors='replace').rstrip()
                
                # 1. Écriture Fichier
                if self._log_file_handle:
                    self._log_file_handle.write(line + "\n")
                    self._log_file_handle.flush()
                
                # 2. Mémoire
                self._recent_logs.append(line)
                
                # 3. Diffusion WebSocket
                for ws in list(self._websockets):
                    try:
                        await ws.send_text(line)
                    except:
                        self._websockets.discard(ws)
                        
        except Exception as e:
            print(f"[ProcessManager] Error consuming logs: {e}")
        finally:
            if self.process and self.process.returncode is not None:
                self.is_running = False
            self._close_log_file()

    async def register_websocket(self, websocket):
        """Enregistre un nouveau client WebSocket et lui envoie l'historique récent."""
        self._websockets.add(websocket)
        for line in self._recent_logs:
            try:
                await websocket.send_text(line)
            except:
                break

    async def get_output_generator(self):
        pass