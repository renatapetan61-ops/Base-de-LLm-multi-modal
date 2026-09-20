"""
Apolo Zenith 1.9 — Zenith Trainer.
Pipeline de treinamento com otimizador AdamW, Cosine LR Scheduler com warmup,
acumulação de gradientes, detecção de instabilidades numéricas (NaN/Inf)
e salvamento periódico de checkpoints.
"""

import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Optional

class ZenithTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 2,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "/root/apolo-zenith-1.9/models/checkpoints",
        device: Optional[torch.device] = None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Otimizador com decaimento de peso desacoplado (AdamW)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        self.step_count = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_aux_loss = 0.0
        batches = 0
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(self.train_dataloader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)
            
            logits, loss, aux_loss = self.model(input_ids=input_ids, labels=labels)
            
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN/Inf detectado no loss! Passo {self.step_count} ignorado.")
                continue
                
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_aux_loss += aux_loss.item()
            batches += 1
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.step_count += 1
                
        avg_loss = total_loss / max(batches, 1)
        avg_aux = total_aux_loss / max(batches, 1)
        return {"loss": avg_loss, "aux_loss": avg_aux, "epoch": epoch}

    def save_checkpoint(self, tag: str):
        path = os.path.join(self.checkpoint_dir, f"zenith_ckpt_{tag}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step_count
        }, path)
        return path
