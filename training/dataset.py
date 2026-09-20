"""
Apolo Zenith 1.9 — Dataset e Carregamento de Dados.
Fornece loaders determinísticos para validação rápida e pipelines de dados de texto e código.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict

class ZenithTokenizedDataset(Dataset):
    def __init__(self, data: List[List[int]], max_seq_len: int = 128, pad_token_id: int = 0):
        self.data = data
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens = tokens + [self.pad_token_id] * (self.max_seq_len - len(tokens))
            
        tensor = torch.tensor(tokens, dtype=torch.long)
        return {"input_ids": tensor, "labels": tensor.clone()}
