"""
Apolo Zenith 1.9 — Zenith Reasoning Engine.
Implementa o ciclo de raciocínio de alta fidelidade:
Problem Understanding -> Task Decomposition -> Planning -> Tool Selection ->
Execution -> Observation -> Verification -> Error Recovery -> Final Validation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

@dataclass
class ReasoningStep:
    step_number: int
    phase: str
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    verified: bool = False

class ZenithReasoningEngine:
    def __init__(self):
        self.history: List[ReasoningStep] = []

    def plan_task(self, user_goal: str) -> List[str]:
        """Decompõe o objetivo em subtarefas atômicas e verificáveis."""
        plan = [
            f"1. Analisar requisitos e restrições técnicas para: {user_goal}",
            "2. Inspecionar ambiente, dependências e código existente",
            "3. Formular arquitetura e plano de modificação incremental",
            "4. Executar implementação em sandbox isolado",
            "5. Executar suíte de validação: Build, Test, Lint, Type Check e Segurança",
            "6. Se houver falha, coletar traceback e aplicar autocorreção",
            "7. Emitir relatório final com evidências verificadas"
        ]
        return plan

    def record_step(self, phase: str, thought: str, action: Optional[str] = None, observation: Optional[str] = None, verified: bool = False) -> ReasoningStep:
        step = ReasoningStep(
            step_number=len(self.history) + 1,
            phase=phase,
            thought=thought,
            action=action,
            observation=observation,
            verified=verified
        )
        self.history.append(step)
        return step

    def verify_result(self, test_output: Dict[str, Any]) -> bool:
        """Verifica se o resultado obtido cumpre os critérios sem assumir sucesso às cegas."""
        exit_code = test_output.get("exit_code", -1)
        stderr = test_output.get("stderr", "")
        return exit_code == 0 and ("Error" not in stderr and "Exception" not in stderr)
