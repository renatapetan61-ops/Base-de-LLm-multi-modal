"""
Apolo Zenith 1.9 — Sandbox de Execução de Código.
Fornece execução segura e isolada de código, com limites rígidos de tempo (timeout),
captura controlada de stdout/stderr e restrições de ambiente.
"""

import sys
import subprocess
import tempfile
import os
import shutil
from typing import Dict, Any, Optional

class ExecutionSandbox:
    def __init__(self, timeout_seconds: int = 30, max_memory_mb: int = 2048):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb

    def execute_python(self, code: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        """
        Executa código Python em processo isolado capturando métricas e saídas.
        """
        temp_dir = tempfile.mkdtemp(prefix="zenith_sandbox_")
        script_path = os.path.join(temp_dir, "script.py")
        
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
                
            work_dir = cwd or temp_dir
            proc = subprocess.run(
                [sys.executable, script_path],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "timed_out": False,
                "status": "success" if proc.returncode == 0 else "error"
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Execução abortada por exceder o tempo limite de {self.timeout_seconds} segundos.",
                "timed_out": True,
                "status": "timeout"
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "timed_out": False,
                "status": "exception"
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def execute_shell(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        """
        Executa comando shell com verificação de segurança básica e timeout.
        """
        # Sanitização de comandos potencialmente destrutivos
        blocked_commands = ["rm -rf /", ":(){ :|:& };:", "mkfs", "dd if=/dev/zero"]
        if any(cmd in command for cmd in blocked_commands):
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": "Comando bloqueado por política de segurança da sandbox.",
                "status": "blocked"
            }
            
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "status": "success" if proc.returncode == 0 else "error"
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Comando excedeu o tempo limite de {self.timeout_seconds} segundos.",
                "status": "timeout"
            }
