"""
Apolo Zenith 1.9 — Script de Treinamento no Dispositivo Móvel.
Executa o treinamento real com otimizador AdamW, acumulação de gradientes,
detecção de instabilidades e salvamento de checkpoint.
"""

import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/apolo-zenith-1.9")

from models.config import ZenithConfig
from models.backbone.transformer import ZenithLMHeadModel
from training.trainer import ZenithTrainer
from training.dataset import ZenithTokenizedDataset

def main():
    print("Iniciando pipeline de treinamento do protótipo no dispositivo móvel...")
    cfg = ZenithConfig.from_yaml("/root/apolo-zenith-1.9/configs/apolo_zenith_1_9.yaml")
    
    # Modelo instanciado com o perfil ativo
    model = ZenithLMHeadModel(cfg.model)
    
    # Criação de dados de validação de pré-treino
    mock_data = [
        torch.randint(0, cfg.model.vocab_size, (64,)).tolist()
        for _ in range(20)
    ]
    dataset = ZenithTokenizedDataset(mock_data, max_seq_len=64)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    trainer = ZenithTrainer(
        model=model,
        train_dataloader=dataloader,
        learning_rate=1e-4,
        gradient_accumulation_steps=2,
        checkpoint_dir="/root/apolo-zenith-1.9/models/checkpoints"
    )
    
    print("Executando época 1 de treinamento...")
    metrics = trainer.train_epoch(epoch=1)
    print(f"Época 1 concluída: Loss médio = {metrics['loss']:.4f}, Perda Auxiliar MoE = {metrics['aux_loss']:.6f}")
    
    ckpt_path = trainer.save_checkpoint("mobile_epoch_1")
    print(f"Checkpoint salvo com sucesso em: {ckpt_path}")

if __name__ == "__main__":
    main()
