"""
Apolo Zenith 1.9 — Zenith Multi-Agent Orchestrator.
Coordena a delegação de tarefas entre os agentes especializados,
mantendo o plano de execução, observações de ferramentas e auto-reflexão.
"""

from typing import Dict, List, Any, Optional
from tools.tool_router import ZenithToolRouter
from agents.reasoning_engine import ZenithReasoningEngine
from agents.specialized_agents import (
    ZenithArchitectAgent,
    ZenithSeniorDeveloperAgent,
    ZenithCodeReviewerAgent,
    ZenithDebuggerAgent,
    ZenithSecurityEngineerAgent
)

class ZenithOrchestrator:
    def __init__(self, tool_router: Optional[ZenithToolRouter] = None):
        self.tool_router = tool_router or ZenithToolRouter()
        self.reasoning_engine = ZenithReasoningEngine()
        
        self.agents = {
            "architect": ZenithArchitectAgent(self.tool_router),
            "senior_dev": ZenithSeniorDeveloperAgent(self.tool_router),
            "reviewer": ZenithCodeReviewerAgent(self.tool_router),
            "debugger": ZenithDebuggerAgent(self.tool_router),
            "security": ZenithSecurityEngineerAgent(self.tool_router),
        }

    def execute_workflow(self, user_prompt: str) -> Dict[str, Any]:
        """
        Executa um fluxo completo agêntico com etapas coordenadas:
        Decomposição -> Arquitetura -> Desenvolvimento -> Revisão de Segurança -> Validação.
        """
        plan = self.reasoning_engine.plan_task(user_prompt)
        
        # 1. Arquitetura
        arch_res = self.agents["architect"].run(user_prompt, {})
        self.reasoning_engine.record_step("Planning", "Blueprint arquitetural formulado", action="architect.run", verified=True)
        
        # 2. Desenvolvimento
        dev_res = self.agents["senior_dev"].run(user_prompt, {"architecture": arch_res})
        self.reasoning_engine.record_step("Execution", "Código implementado pelo dev", action="senior_dev.run", verified=True)
        
        # 3. Revisão de Segurança
        sec_res = self.agents["security"].run(user_prompt, {})
        self.reasoning_engine.record_step("Security Check", "Análise de segurança concluída", action="security.run", verified=sec_res["secure"])
        
        # 4. Revisão de Código e QA
        review_res = self.agents["reviewer"].run(user_prompt, {})
        self.reasoning_engine.record_step("Verification", "Revisão e QA aprovados", action="reviewer.run", verified=True)
        
        return {
            "status": "success",
            "prompt": user_prompt,
            "plan": plan,
            "agent_results": {
                "architect": arch_res,
                "developer": dev_res,
                "security": sec_res,
                "reviewer": review_res
            },
            "steps_executed": len(self.reasoning_engine.history)
        }
