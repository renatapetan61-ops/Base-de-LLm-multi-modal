"""
Testes dos Sistemas Agênticos, Ferramentas e Memória do Apolo Zenith 1.9
"""

import unittest
from tools.sandbox import ExecutionSandbox
from tools.tool_router import ZenithToolRouter
from agents.orchestrator import ZenithOrchestrator
from memory.hierarchical_memory import ZenithMemorySystem

class TestAgentsAndTools(unittest.TestCase):
    def setUp(self):
        self.sandbox = ExecutionSandbox(timeout_seconds=5)
        self.tool_router = ZenithToolRouter()
        self.memory = ZenithMemorySystem()

    def test_sandbox_python_execution(self):
        res = self.sandbox.execute_python("print(sum([10, 20, 30]))")
        self.assertEqual(res["exit_code"], 0)
        self.assertIn("60", res["stdout"])

    def test_tool_router_dispatch(self):
        res = self.tool_router.dispatch("run_python", {"code": "print('Tool Router OK')"})
        self.assertEqual(res["status"], "success")
        self.assertIn("Tool Router OK", res["stdout"])

    def test_hierarchical_memory(self):
        self.memory.working.set("active_task", "Testar sistema de memória")
        self.assertEqual(self.memory.working.get("active_task"), "Testar sistema de memória")
        
        self.memory.codebase.add_document("doc1", "def calcular_raiz(x): return x ** 0.5")
        self.memory.codebase.add_document("doc2", "class ServidorWeb: pass")
        
        search_res = self.memory.codebase.search("calcular", top_k=1)
        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res[0]["id"], "doc1")

    def test_orchestrator_workflow(self):
        orchestrator = ZenithOrchestrator(self.tool_router)
        result = orchestrator.execute_workflow("Criar uma API em Python para processamento de imagens")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["steps_executed"], 4)
        self.assertTrue(result["agent_results"]["security"]["secure"])

if __name__ == "__main__":
    unittest.main()
