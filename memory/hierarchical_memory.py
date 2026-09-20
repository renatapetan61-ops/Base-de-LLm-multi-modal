"""
Apolo Zenith 1.9 — Zenith Hierarchical Memory Subsystem.
Gerencia memória de trabalho, histórico conversacional, índice semântico de projeto
e memória de longo prazo através de busca vetorial leve e armazenamento estruturado.
"""

from typing import Dict, List, Any, Optional
import math
import torch
import torch.nn.functional as F

class WorkingMemory:
    """Armazena o estado volátil da tarefa atual e variáveis intermediárias."""
    def __init__(self):
        self.state: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self.state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def clear(self):
        self.state.clear()

class ConversationMemory:
    """Armazena os turnos de interação entre usuário e sistema."""
    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_turns:
            self.messages.pop(0)

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.messages)

class CodebaseVectorMemory:
    """Indexador semântico de snippets e arquivos de código do projeto."""
    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[torch.Tensor] = []

    def _pseudo_embed(self, text: str) -> torch.Tensor:
        """Gera embedding semântico via hashing de termos (Bag-of-Words / Hashing Trick)."""
        vec = torch.zeros(self.embed_dim)
        import re
        tokens = re.findall(r"\w+", text.lower())
        for token in tokens:
            h = abs(hash(token)) % self.embed_dim
            vec[h] += 1.0
        norm = torch.norm(vec, p=2)
        if norm > 0:
            vec = vec / norm
        return vec

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        emb = self._pseudo_embed(content)
        self.documents.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata or {}
        })
        self.embeddings.append(emb)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []
            
        q_emb = self._pseudo_embed(query)
        stacked = torch.stack(self.embeddings) # (N, D)
        scores = torch.mv(stacked, q_emb)      # (N,)
        
        topk_vals, topk_indices = torch.topk(scores, min(top_k, len(self.documents)))
        
        results = []
        for score, idx in zip(topk_vals.tolist(), topk_indices.tolist()):
            doc = dict(self.documents[idx])
            doc["score"] = score
            results.append(doc)
            
        return results

class ZenithMemorySystem:
    def __init__(self):
        self.working = WorkingMemory()
        self.conversation = ConversationMemory()
        self.codebase = CodebaseVectorMemory()
