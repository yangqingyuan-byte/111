# Level=3 小波包分解详细说明

## 重要澄清

**Level=3 不会产生"低频、中频、高频"三个频段**，而是产生 **8个频段**（2^3 = 8）。

小波包分解是**二叉树结构**，每次分解都是将信号分成**低频（cA）和高频（cD）**两部分，而不是分成三部分。

## Level=3 的完整分解过程

### 分解树结构

```
                        原始信号 [B, N, L]
                                │
                ┌───────────────┴───────────────┐
                │                               │
        Level 1: cA (低频)              Level 1: cD (高频)
        [B, N, L/2]                    [B, N, L/2]
                │                               │
        ┌───────┴───────┐               ┌───────┴───────┐
        │               │               │               │
Level 2: cA-cA      Level 2: cA-cD  Level 2: cD-cA  Level 2: cD-cD
[B, N, L/4]        [B, N, L/4]     [B, N, L/4]     [B, N, L/4]
        │               │               │               │
    ┌───┴───┐       ┌───┴───┐       ┌───┴───┐       ┌───┴───┐
    │       │       │       │       │       │       │       │
Level 3: 8个频段节点（每个 [B, N, L/8]）
```

### 8个频段的详细划分

| 频段编号 | 路径 | 频率范围 | 说明 |
|---------|------|---------|------|
| **频段 0** | cA-cA-cA | 最低频 | 经过3次低频提取 |
| **频段 1** | cA-cA-cD | 低中频 | 2次低频 + 1次高频 |
| **频段 2** | cA-cD-cA | 中低频 | 1次低频 + 1次高频 + 1次低频 |
| **频段 3** | cA-cD-cD | 中频 | 1次低频 + 2次高频 |
| **频段 4** | cD-cA-cA | 中高频 | 1次高频 + 2次低频 |
| **频段 5** | cD-cA-cD | 高频 | 1次高频 + 1次低频 + 1次高频 |
| **频段 6** | cD-cD-cA | 高高频 | 2次高频 + 1次低频 |
| **频段 7** | cD-cD-cD | 最高频 | 经过3次高频提取 |

### 频率范围示意

```
频率范围（从低到高）:
┌─────────────────────────────────────────────────────────┐
│ 频段0 │ 频段1 │ 频段2 │ 频段3 │ 频段4 │ 频段5 │ 频段6 │ 频段7 │
│最低频 │低中频 │中低频 │ 中频  │中高频 │ 高频  │高高频 │最高频 │
└─────────────────────────────────────────────────────────┘
```

## 代码实现

### 分解过程（代码第61-72行）

```python
def forward(self, x):
    nodes = [x]  # 初始：1个节点
    
    # Level 1: 1 → 2 个节点
    for _ in range(self.level):  # level=3，循环3次
        next_nodes = []
        for node in nodes:
            cA, cD = self._dwt_step(node)  # 每个节点分成2个
            next_nodes.append(cA)  # 低频
            next_nodes.append(cD)  # 高频
        nodes = next_nodes
    
    # Level 3 最终：8个节点
    return nodes  # List of 8 tensors, each [B, N, L/8]
```

### 频段处理（代码第189-210行）

```python
def wp_domain_processing(self, x):
    wp_nodes = self.wp_transform(x)  # 返回8个频段节点
    
    wp_features_list = []
    for i, node in enumerate(wp_nodes):  # i = 0, 1, 2, ..., 7
        # 每个频段独立处理
        node_reshaped = node.unsqueeze(-1).reshape(B * N, L/8, 1)
        tokens = self.wp_proj_layers[i](node_reshaped)  # 独立投影层
        tokens = tokens + self.freq_pos_embed[:, i, :]  # 频段位置编码
        encoded = self.wp_encoder(tokens)
        wp_features_list.append(encoded)
    
    # 融合8个频段
    pooled = self.wp_pool(wp_features_list)  # [B*N, C]
    return pooled.reshape(B, N, self.channel)
```

## 频段融合机制

### WaveletPacketAttentionPooling（代码第74-96行）

对于 Level=3 的8个频段：

```python
# 1. 对每个频段内部进行注意力池化
pooled_list = []
for i in range(8):  # i = 0, 1, 2, ..., 7
    feat = wp_features[i]  # [B*N, L/8, C]
    attn_weights = F.softmax(self.attention(feat), dim=1)
    pooled = (feat * attn_weights).sum(dim=1)  # [B*N, C]
    
    # 2. 应用可学习的节点权重
    pooled_list.append(pooled * self.node_weights[i])

# 3. 融合8个频段（平均）
final_pooled = torch.stack(pooled_list, dim=1).mean(dim=1)  # [B*N, C]
```

### 融合公式

```
最终特征 = Mean([
    w0 × pooled[0],  # 最低频
    w1 × pooled[1],  # 低中频
    w2 × pooled[2],  # 中低频
    w3 × pooled[3],  # 中频
    w4 × pooled[4],  # 中高频
    w5 × pooled[5],  # 高频
    w6 × pooled[6],  # 高高频
    w7 × pooled[7]   # 最高频
])
```

其中 `w0, w1, ..., w7` 是可学习的节点权重。

## 数据维度变化

```
输入:              [B, N, L]
    ↓ WPD (level=3)
Level 1:           [B, N, L/2] × 2 (cA, cD)
    ↓
Level 2:           [B, N, L/4] × 4
    ↓
Level 3:           [B, N, L/8] × 8 个频段
    ↓ 投影
    ↓              [B*N, L/8, C] × 8
    ↓ 编码+池化
    ↓              [B*N, C] × 8
    ↓ 融合
输出:              [B, N, C]
```

## 不同 Level 的对比

| Level | 频段数量 | 每个频段长度 | 频率分辨率 | 时间分辨率 |
|-------|---------|------------|----------|-----------|
| **1** | 2 | L/2 | 粗 | 细 |
| **2** | 4 | L/4 | 中 | 中 |
| **3** | 8 | L/8 | 细 | 粗 |

## 为什么不是"低频、中频、高频"三个？

小波包分解使用**二进制树结构**：
- 每次分解：**1个节点 → 2个子节点**（低频 cA + 高频 cD）
- Level=3：**1 → 2 → 4 → 8** 个节点

如果要得到"低频、中频、高频"三个频段，需要使用**三进制分解**或其他方法，但这不是小波包分解的标准做法。

## 实际应用建议

### Level=2 vs Level=3

- **Level=2 (4个频段)**: 
  - ✅ 计算效率高
  - ✅ 适合大多数时序预测任务
  - ✅ 平衡了频率和时间分辨率

- **Level=3 (8个频段)**:
  - ✅ 频率分辨率更精细
  - ⚠️ 计算量增加（8个频段需要8个投影层）
  - ⚠️ 每个频段的时间分辨率降低（L/8）
  - 💡 适合需要精细频率分析的任务

### 选择建议

- **一般任务**: Level=2 (4个频段)
- **精细频率分析**: Level=3 (8个频段)
- **简单快速**: Level=1 (2个频段)
