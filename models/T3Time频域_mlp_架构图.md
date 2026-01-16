# T3Time_FreTS_Gated_Qwen 网络架构图

## 整体架构流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          输入层 (Input Layer)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  input_data: [B, L, N]          embeddings: [B, d_llm, N, 1]               │
│  (batch, seq_len, num_nodes)    (prompt embeddings)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RevIN 归一化 (Normalization)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  x_norm = Normalize(x, 'norm')                                               │
│  [B, L, N] → [B, N, L]                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌───────────────────────────────┐  ┌──────────────────────────────────────────┐
│    时域分支 (Time Domain)      │  │    频域分支 (Frequency Domain)           │
├───────────────────────────────┤  ├──────────────────────────────────────────┤
│  [B, N, L]                     │  │  [B, N, L]                               │
│       │                         │  │       │                                 │
│       ▼                         │  │       ▼                                 │
│  Linear(L → C)                 │  │  Reshape → [B*N, L, 1]                  │
│  length_to_feature             │  │       │                                 │
│       │                         │  │       ▼                                 │
│       ▼                         │  │  Linear(1 → C)                         │
│  [B, N, C]                     │  │  fre_projection                         │
│       │                         │  │       │                                 │
│       ▼                         │  │       ▼                                 │
│  GatedTransformerEncoder       │  │  FreTS Component                        │
│  (e_layer layers)              │  │  - FFT → 频域                           │
│  ts_encoder                    │  │  - 可学习 MLP (r, i, rb, ib)            │
│       │                         │  │  - SoftShrink (稀疏化)                 │
│       ▼                         │  │  - IFFT → 时域                          │
│  time_encoded                  │  │       │                                 │
│  [B, N, C]                     │  │       ▼                                 │
└───────────────────────────────┘  │  AttentionPooling                         │
                                   │  fre_pool                                 │
                                   │       │                                 │
                                   │       ▼                                 │
                                   │  [B, N, C]                               │
                                   │       │                                 │
                                   │       ▼                                 │
                                   │  GatedTransformerEncoder                 │
                                   │  fre_encoder                             │
                                   │       │                                 │
                                   │       ▼                                 │
                                   │  fre_encoded                             │
                                   │  [B, N, C]                               │
                                   └──────────────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Gate 融合机制 (Fusion Gate)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  horizon_info = pred_len / 100.0                                            │
│  gate_input = Concat([time_encoded, fre_encoded, horizon_info])              │
│  gate = Sigmoid(MLP(gate_input))                                             │
│  fused_features = time_encoded + gate * fre_encoded                          │
│  [B, N, C] → [B, C, N]                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌───────────────────────────────┐  ┌──────────────────────────────────────────┐
│  Prompt 编码 (Prompt Encoder) │  │  融合特征 (Fused Features)               │
├───────────────────────────────┤  ├──────────────────────────────────────────┤
│  embeddings: [B, N, d_llm]    │  │  fused_features: [B, C, N]              │
│       │                        │  │                                          │
│       ▼                        │  │                                          │
│  GatedTransformerEncoder      │  │                                          │
│  (e_layer layers)              │  │                                          │
│  prompt_encoder                │  │                                          │
│       │                        │  │                                          │
│       ▼                        │  │                                          │
│  [B, N, d_llm] → [B, d_llm, N] │  │                                          │
│  prompt_feat                   │  │                                          │
└───────────────────────────────┘  └──────────────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              Cross-Modal Alignment (CMA)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  4个 CMA Heads (并行):                                                       │
│  cma_outputs[i] = CrossModal(fused_features, prompt_feat, prompt_feat)      │
│       │                                                                      │
│       ▼                                                                      │
│  AdaptiveDynamicHeadsCMA                                                     │
│  (自适应动态头融合)                                                           │
│       │                                                                      │
│       ▼                                                                      │
│  fused_cma: [B, C, N]                                                        │
│       │                                                                      │
│       ▼                                                                      │
│  Residual Connection:                                                         │
│  cross_out = α * fused_cma + (1-α) * fused_features                         │
│  [B, C, N] → [B, N, C]                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     解码器 (Decoder)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  TransformerDecoder                                                          │
│  (d_layer layers)                                                            │
│       │                                                                      │
│       ▼                                                                      │
│  Linear(C → pred_len)                                                        │
│  c_to_length                                                                 │
│       │                                                                      │
│       ▼                                                                      │
│  dec_out: [B, N, pred_len]                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RevIN 反归一化 (Denormalization)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  output = Normalize(dec_out, 'denorm')                                       │
│  [B, N, pred_len]                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │   输出 (Output)│
                            │  [B, N, pred_len]│
                            └───────────────┘
```

## 详细组件说明

### 1. RevIN 归一化层
- **功能**: 可逆实例归一化
- **输入**: [B, L, N]
- **输出**: [B, N, L] (归一化后)

### 2. 时域分支 (Time Domain Branch)
```
[B, N, L] 
  → Linear(L → C)          [length_to_feature]
  → [B, N, C]
  → GatedTransformerEncoder × e_layer
  → time_encoded [B, N, C]
```

### 3. 频域分支 (Frequency Domain Branch)
```
[B, N, L]
  → Reshape [B*N, L, 1]
  → Linear(1 → C)         [fre_projection]
  → FreTS Component:
     - FFT (时域 → 频域)
     - 可学习 MLP (复数域操作)
     - SoftShrink (稀疏化, threshold=0.009)
     - IFFT (频域 → 时域)
  → AttentionPooling       [fre_pool]
  → [B, N, C]
  → GatedTransformerEncoder [fre_encoder]
  → fre_encoded [B, N, C]
```

### 4. FreTS Component 详细结构
```
输入: [B*N, L, C]
  ↓
FFT: [B*N, L, C] (复数)
  ↓
可学习 MLP:
  - o_real = ReLU(real × r - imag × i + rb)
  - o_imag = ReLU(imag × r + real × i + ib)
  ↓
SoftShrink (稀疏化, λ=0.009)
  ↓
IFFT: [B*N, L, C]
  ↓
输出: [B*N, L, C]
```

### 5. Gate 融合机制
```
输入:
  - time_encoded: [B, N, C]
  - fre_encoded: [B, N, C]
  - horizon_info: [B, N, 1] (pred_len / 100.0)

处理:
  gate_input = Concat([time, fre, horizon])  [B, N, 2C+1]
  gate = Sigmoid(MLP(gate_input))            [B, N, C]
  fused = time + gate * fre                  [B, N, C]
  → [B, C, N]
```

### 6. Cross-Modal Alignment (CMA)
```
输入:
  - fused_features: [B, C, N]
  - prompt_feat: [B, d_llm, N]

处理:
  4个 CMA Heads (并行):
    cma_outputs[i] = CrossModal(fused, prompt, prompt)
  
  AdaptiveDynamicHeadsCMA:
    fused_cma = AdaptiveFusion(cma_outputs)
  
  Residual:
    cross_out = α * fused_cma + (1-α) * fused_features
  → [B, N, C]
```

### 7. 解码器
```
输入: cross_out [B, N, C]
  ↓
TransformerDecoder (d_layer layers)
  ↓
Linear(C → pred_len) [c_to_length]
  ↓
输出: [B, N, pred_len]
```

## 关键参数

- **B**: Batch size
- **L**: Sequence length (seq_len)
- **N**: Number of nodes/variables (num_nodes)
- **C**: Channel dimension (channel)
- **d_llm**: LLM embedding dimension (1024)
- **pred_len**: Prediction length
- **e_layer**: Encoder layers
- **d_layer**: Decoder layers
- **head**: Attention heads

## 数据流维度变化

```
input_data:        [B, L, N]
  ↓ RevIN norm
x_norm:            [B, N, L]
  ↓ Split
  ├─→ 时域分支
  │     ↓
  │  time_encoded: [B, N, C]
  │
  └─→ 频域分支
        ↓
     fre_encoded:  [B, N, C]
  ↓ Gate Fusion
fused_features:    [B, C, N]
  ↓ CMA
cross_out:         [B, N, C]
  ↓ Decoder
dec_out:           [B, N, pred_len]
  ↓ RevIN denorm
output:            [B, N, pred_len]
```
