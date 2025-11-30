import asyncio
import os
import signal
from typing import Optional

class ProcessManager:
    def __init__(self):
        self.process = None
        self.is_running = False

    async def start(self, command: str, cwd: Optional[str] = None, env: Optional[dict] = None):
        """
        Lance la commande en arrière-plan.
        Redirige stderr vers stdout pour n'avoir qu'un seul flux à lire.
        Accepts optional working directory (cwd) and environment variables (env).
        """
        if self.is_running and self.process:
             # Vérifie si le processus est vraiment en vie
            if self.process.returncode is None:
                raise Exception("A training process is already running.")

        # On utilise le shell pour pouvoir passer la commande complexe entière
        # On redirige stderr vers stdout (2>&1) pour simplifier la capture des logs
        # On utilise os.setsid pour créer un groupe de processus (facilite le kill)
        self.process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, 
            preexec_fn=os.setsid,
            cwd=cwd,
            env=env # Injection de l'environnement modifié
        )
        self.is_running = True
        return self.process.pid

    async def stop(self):
        """
        Arrête le processus en cours.
        """
        if self.process and self.is_running:
            try:
                # Tuer le groupe de processus (pgid) pour être sûr de tuer les sous-processus deepspeed
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass # Déjà mort
            
            self.is_running = False
            self.process = None

    async def get_output_generator(self):
        """
        Générateur asynchrone qui renvoie les lignes de logs une par une.
        """
        if not self.process:
            return

        while True:
            # Lit une ligne du flux stdout
            line = await self.process.stdout.readline()
            
            if line:
                yield line.decode('utf-8', errors='replace').rstrip()
            else:
                # Si pas de ligne et que le process est fini, on arrête
                if self.process.returncode is not None:
                    self.is_running = False
                    break
                # Sinon on attend un peu pour ne pas spammer la boucle
                await asyncio.sleep(0.1)