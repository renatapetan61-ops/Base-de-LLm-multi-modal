# Apolo Zenith 1.9 — Plano de Escalabilidade para 199 Bilhões de Parâmetros

## 1. Requisitos Computacionais e de Memória

### 1.1 Análise de Parâmetros e VRAM
- **Tamanho do Modelo**: ~199B parâmetros totais, ~32B parâmetros ativos por token.
- **Armazenamento de Pesos em FP16 / BF16**:
  $$\text{Memória dos Pesos} = 199 \times 10^9 \times 2 \text{ bytes} \approx 398 \text{ GB}$$
- **Estados do Otimizador (AdamW em FP32 para precisão mista)**:
  - Momentum ($m_t$): 796 GB
  - Variância ($v_t$): 796 GB
  - Cópia mestra dos pesos em FP32: 796 GB
  - **Memória Total do Otimizador**: $\approx 2.38 \text{ TB}$
- **Gradientes (FP16/BF16)**: ~398 GB
- **Capacidade Total Mínima para Treinamento Completo**: $\approx 3.2 \text{ TB}$ de memória agregada de GPUs/aceleradores.

---

## 2. Estratégia de Paralelismo Distribuído (3D + Expert Parallelism)

Para treinar o modelo de 199B parâmetros, o código-base do Apolo Zenith 1.9 foi projetado para particionamento ortogonal:

1. **Expert Parallelism (EP = 8 a 16)**:
   - Os 64 experts por camada são divididos entre os nós de computação (por exemplo, 4 a 8 experts por GPU).
   - O tráfego de dados é coordenado através de uma operação coletiva All-to-All (`torch.distributed.all_to_all_single`).
2. **Tensor Parallelism (TP = 8)**:
   - Divisão intra-nó das matrizes de atenção ($W_Q, W_K, W_V, W_O$) e das projeções de gating do router através do barramento NVLink (largura de banda $\ge 900 \text{ GB/s}$).
3. **Pipeline Parallelism (PP = 8)**:
   - As 64 camadas do Transformer são distribuídas em 8 estágios de pipeline com agendamento 1F1B (One Forward, One Backward).
4. **Data Parallelism & ZeRO-3 / FSDP (DP = 4 a 8)**:
   - Sharding dos estados do otimizador e dos gradientes entre réplicas de dados para saturação do throughput.
5. **Configuração de Hardware Recomendada**:
   - Cluster de 64 a 128 nós equipados com aceleradores modernos (ex: 8x H100 / B200 por nó com rede InfiniBand 400-800 Gbps).

---

## 3. Caminho de Migração do Protótipo Móvel para a Escala Completa

O Apolo Zenith 1.9 utiliza arquitetura de código único (`single codebase`):
- O mesmo arquivo de configuração `configs/apolo_zenith_1_9.yaml` chaveia entre `mobile_proto_125m` e `scale_199b`.
- Todas as assinaturas das camadas PyTorch, de atenção GQA com RoPE, da perda auxiliar MoE, dos encoders multimodais e do orquestrador são idênticas.
- Ao migrar para o cluster distribuído, basta executar:
  ```bash
  torchrun --nproc_per_node=8 --nnodes=16 scripts/train_distributed.py --profile scale_199b
  ```
