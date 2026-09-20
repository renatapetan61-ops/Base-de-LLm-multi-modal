"""
Apolo Zenith 1.9 — Zenith Tool Router.
Despacha e roteia chamadas de ferramentas de forma tipada, segura e observável.
"""

import os
import json
from typing import Dict, Any, Callable
from tools.sandbox import ExecutionSandbox
from tools.language_graph import LanguageKnowledgeGraph

class ZenithToolRouter:
    def __init__(self, workspace_root: str = "/root/apolo-zenith-1.9"):
        self.workspace_root = workspace_root
        self.sandbox = ExecutionSandbox()
        self.lang_graph = LanguageKnowledgeGraph()
        self.tools: Dict[str, Callable] = {
            "run_python": self._tool_run_python,
            "run_shell": self._tool_run_shell,
            "read_file": self._tool_read_file,
            "write_file": self._tool_write_file,
            "list_directory": self._tool_list_directory,
            "analyze_language": self._tool_analyze_language,
        }

    def _resolve_safe_path(self, path: str) -> str:
        """Assegura que o caminho não tente escapar do workspace arbitrariamente."""
        abs_path = os.path.abspath(os.path.join(self.workspace_root, path))
        return abs_path

    def _tool_run_python(self, code: str) -> Dict[str, Any]:
        return self.sandbox.execute_python(code, cwd=self.workspace_root)

    def _tool_run_shell(self, command: str) -> Dict[str, Any]:
        return self.sandbox.execute_shell(command, cwd=self.workspace_root)

    def _tool_read_file(self, path: str) -> Dict[str, Any]:
        target = self._resolve_safe_path(path)
        if not os.path.exists(target):
            return {"status": "error", "error": f"Arquivo não encontrado: {path}"}
        try:
            with open(target, "r", encoding="utf-8") as f:
                content = f.read()
            return {"status": "success", "content": content}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _tool_write_file(self, path: str, content: str) -> Dict[str, Any]:
        target = self._resolve_safe_path(path)
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "message": f"Arquivo gravado em {target}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _tool_list_directory(self, path: str = ".") -> Dict[str, Any]:
        target = self._resolve_safe_path(path)
        if not os.path.exists(target):
            return {"status": "error", "error": f"Diretório não encontrado: {path}"}
        try:
            entries = os.listdir(target)
            return {"status": "success", "entries": entries}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _tool_analyze_language(self, snippet_or_file: str) -> Dict[str, Any]:
        profile = self.lang_graph.identify_language(snippet_or_file)
        guidelines = self.lang_graph.get_adapter_guidelines(profile.name if profile else "unknown")
        return {"status": "success", "profile": guidelines}

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Despacha uma chamada de ferramenta validando sua existência."""
        if tool_name not in self.tools:
            return {
                "status": "error",
                "error": f"Ferramenta '{tool_name}' não reconhecida. Ferramentas disponíveis: {list(self.tools.keys())}"
            }
        try:
            return self.tools[tool_name](**arguments)
        except TypeError as e:
            return {"status": "error", "error": f"Argumentos inválidos para {tool_name}: {str(e)}"}
