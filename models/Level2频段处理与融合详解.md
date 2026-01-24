# Level=2 时 4 个频段的处理与融合详解

## 完整处理流程

### 流程图

```
原始信号 [B, N, L]
    ↓
小波包分解 (Level=2)
    ↓
4个频段: [B, N, L/4] × 4
    ↓
【对每个频段独立处理】
    ├─ 频段 0 (cA-cA, 最低频)
    │   ├─ Reshape: [B*N, L/4, 1]
    │   ├─ 独立投影: Linear(1 → C) → [B*N, L/4, C]
    │   ├─ 位置编码: + freq_pos_embed[0] → [B*N, L/4, C]
    │   ├─ 编码器: GatedTransformerEncoder → [B*N, L/4, C]
    │   └─ 注意力池化: AttentionPooling → [B*N, C]
    │
    ├─ 频段 1 (cA-cD, 低中频)
    │   └─ (相同流程)
    │
    ├─ 频段 2 (cD-cA, 中高频)
    │   └─ (相同流程)
    │
    └─ 频段 3 (cD-cD, 最高频)
        └─ (相同流程)
    ↓
【融合阶段】
    ├─ 应用节点权重: pooled[i] × node_weights[i]
    └─ 平均融合: mean([pooled[0], pooled[1], pooled[2], pooled[3]])
    ↓
最终输出: [B, N, C]
```

## 详细步骤说明

### 步骤 1: 小波包分解

**代码位置**: `wp_domain_processing()` 第192行

```python
wp_nodes = self.wp_transform(x)  # List of 4 tensors
```

**结果**:
- 频段 0: `[B, N, 30]` (cA-cA, 最低频)
- 频段 1: `[B, N, 30]` (cA-cD, 低中频)
- 频段 2: `[B, N, 30]` (cD-cA, 中高频)
- 频段 3: `[B, N, 30]` (cD-cD, 最高频)

### 步骤 2: 独立投影（每个频段独立处理）

**代码位置**: `wp_domain_processing()` 第195-206行

```python
wp_features_list = []
for i, node in enumerate(wp_nodes):  # i = 0, 1, 2, 3
    # 2.1 Reshape
    L_part = node.shape[-1]  # 30
    node_reshaped = node.unsqueeze(-1).reshape(B * N, L_part, 1)
    # [B, N, 30] → [B*N, 30, 1]
    
    # 2.2 独立投影（每个频段使用不同的投影层）
    proj = self.wp_proj_layers[i]  # 频段 i 的独立投影层
    tokens = proj(node_reshaped)  # [B*N, 30, 1] → [B*N, 30, C]
    
    # 2.3 频段位置编码
    tokens = tokens + self.freq_pos_embed[:, i, :].unsqueeze(1)
    # 添加频段特定的位置编码，区分不同频段
    
    # 2.4 编码器处理
    encoded = self.wp_encoder(tokens)  # [B*N, 30, C]
    wp_features_list.append(encoded)
```

**关键点**:
- **独立投影层**: 每个频段使用 `wp_proj_layers[i]`，允许不同频段学习不同的特征表示
- **位置编码**: `freq_pos_embed[:, i, :]` 为每个频段添加特定的位置信息
- **共享编码器**: 所有频段使用同一个 `wp_encoder`，但输入不同

**数据维度变化**:
```
频段 i: [B, N, 30]
    ↓ reshape
    [B*N, 30, 1]
    ↓ 独立投影 (Linear: 1 → C)
    [B*N, 30, C]
    ↓ + 位置编码
    [B*N, 30, C]
    ↓ 编码器
    [B*N, 30, C]
```

### 步骤 3: 注意力池化（每个频段内部）

**代码位置**: `WaveletPacketAttentionPooling.forward()` 第84-96行

#### 3.1 对每个频段进行注意力池化

```python
pooled_list = []
for i, feat in enumerate(wp_features):  # feat: [B*N, L_part, C]
    # 3.1.1 计算注意力权重
    attn_logits = self.attention(feat)  # [B*N, 30, C] → [B*N, 30, 1]
    attn_weights = F.softmax(attn_logits, dim=1)  # [B*N, 30, 1]
    
    # 3.1.2 加权池化（对时序维度）
    pooled = (feat * attn_weights).sum(dim=1)  # [B*N, 30, C] → [B*N, C]
    
    # 3.1.3 应用节点权重
    pooled_list.append(pooled * self.node_weights[i])
```

**注意力机制详解**:

1. **注意力网络**:
   ```python
   self.attention = nn.Sequential(
       nn.Linear(C, C // 2),  # [B*N, 30, C] → [B*N, 30, C//2]
       nn.ReLU(),
       nn.Linear(C // 2, 1)   # [B*N, 30, C//2] → [B*N, 30, 1]
   )
   ```

2. **注意力权重计算**:
   - 对每个时间步计算一个标量分数
   - Softmax 归一化，使得权重和为 1
   - 权重大的时间步对最终特征贡献更大

3. **加权池化**:
   ```
   输入: feat = [B*N, 30, C]
   权重: attn_weights = [B*N, 30, 1]
   输出: pooled = Σ(feat[t] × attn_weights[t])  for t in [0, 29]
        = [B*N, C]
   ```

**数学公式**:
$$pooled_i = \sum_{t=0}^{L_{part}-1} feat_i[t] \cdot \alpha_i[t]$$

其中 $\alpha_i[t] = \text{softmax}(\text{MLP}(feat_i[t]))$ 是注意力权重。

#### 3.2 4 个频段的池化结果

```
频段 0: pooled[0] = [B*N, C]  (最低频的池化特征)
频段 1: pooled[1] = [B*N, C]  (低中频的池化特征)
频段 2: pooled[2] = [B*N, C]  (中高频的池化特征)
频段 3: pooled[3] = [B*N, C]  (最高频的池化特征)
```

### 步骤 4: 节点权重和融合

**代码位置**: `WaveletPacketAttentionPooling.forward()` 第92-95行

#### 4.1 应用可学习的节点权重

```python
# 对每个频段的池化结果应用权重
pooled_list = []
for i in range(4):
    weighted = pooled[i] * self.node_weights[i]
    pooled_list.append(weighted)
```

**节点权重的作用**:
- `node_weights`: 可学习参数，形状 `[4]`
- 自动学习不同频段的重要性
- 例如：如果低频更重要，`node_weights[0]` 会更大

**示例**:
```python
node_weights = [1.2, 0.9, 0.8, 0.6]  # 学习到的权重
# 表示：最低频最重要，最高频相对不重要
```

#### 4.2 多频段融合

```python
# 堆叠所有频段的池化结果
stacked = torch.stack(pooled_list, dim=1)  # [B*N, 4, C]

# 平均融合
final_pooled = stacked.mean(dim=1)  # [B*N, C]
```

**融合公式**:
$$final = \frac{1}{4} \sum_{i=0}^{3} w_i \cdot pooled_i$$

其中 $w_i = \text{node\_weights}[i]$ 是可学习的节点权重。

**数据维度变化**:
```
4个频段: [B*N, C] × 4
    ↓ stack
    [B*N, 4, C]
    ↓ mean(dim=1)
    [B*N, C]
    ↓ reshape
    [B, N, C]
```

## 完整代码流程

### wp_domain_processing() 完整流程

```python
def wp_domain_processing(self, x):
    B, N, L = x.shape  # [B, N, 96]
    
    # 1. 小波包分解
    wp_nodes = self.wp_transform(x)  # List of 4: [B, N, 30] × 4
    
    wp_features_list = []
    # 2. 对每个频段独立处理
    for i, node in enumerate(wp_nodes):  # i = 0, 1, 2, 3
        # 2.1 Reshape
        L_part = node.shape[-1]  # 30
        node_reshaped = node.unsqueeze(-1).reshape(B * N, L_part, 1)
        
        # 2.2 独立投影
        tokens = self.wp_proj_layers[i](node_reshaped)  # [B*N, 30, C]
        
        # 2.3 位置编码
        tokens = tokens + self.freq_pos_embed[:, i, :].unsqueeze(1)
        
        # 2.4 编码器
        encoded = self.wp_encoder(tokens)  # [B*N, 30, C]
        wp_features_list.append(encoded)
    
    # 3. 注意力池化和融合
    pooled = self.wp_pool(wp_features_list)  # [B*N, C]
    
    # 4. Reshape 回原始维度
    return pooled.reshape(B, N, self.channel)  # [B, N, C]
```

### WaveletPacketAttentionPooling.forward() 完整流程

```python
def forward(self, wp_features):
    # wp_features: List of 4, each [B*N, 30, C]
    
    pooled_list = []
    # 对每个频段进行注意力池化
    for i, feat in enumerate(wp_features):  # i = 0, 1, 2, 3
        # 3.1 计算注意力权重
        attn_logits = self.attention(feat)  # [B*N, 30, 1]
        attn_weights = F.softmax(attn_logits, dim=1)  # [B*N, 30, 1]
        
        # 3.2 加权池化
        pooled = (feat * attn_weights).sum(dim=1)  # [B*N, C]
        
        # 3.3 应用节点权重
        pooled_list.append(pooled * self.node_weights[i])
    
    # 4. 融合所有频段
    final_pooled = torch.stack(pooled_list, dim=1).mean(dim=1)  # [B*N, C]
    return final_pooled
```

## 数据维度完整变化

```
原始输入: [B, N, L] = [2, 7, 96]
    ↓ 小波包分解 (Level=2)
4个频段: [B, N, L/4] × 4 = [2, 7, 30] × 4
    ↓ Reshape
    [B*N, L/4, 1] × 4 = [14, 30, 1] × 4
    ↓ 独立投影 (Linear: 1 → C)
    [B*N, L/4, C] × 4 = [14, 30, 32] × 4
    ↓ + 位置编码
    [B*N, L/4, C] × 4 = [14, 30, 32] × 4
    ↓ 编码器
    [B*N, L/4, C] × 4 = [14, 30, 32] × 4
    ↓ 注意力池化 (对每个频段)
    [B*N, C] × 4 = [14, 32] × 4
    ↓ 应用节点权重
    [B*N, C] × 4 = [14, 32] × 4
    ↓ Stack + Mean
    [B*N, C] = [14, 32]
    ↓ Reshape
最终输出: [B, N, C] = [2, 7, 32]
```

## 关键设计点

### 1. 独立投影层

**为什么每个频段独立投影？**
- 不同频段可能包含不同类型的信息
- 允许模型学习频段特定的特征表示
- 提高模型的表达能力

### 2. 频段位置编码

**为什么需要位置编码？**
- 区分不同频段（0, 1, 2, 3）
- 帮助模型理解频段的频率位置关系
- 增强模型对频段差异的感知

### 3. 注意力池化

**为什么使用注意力池化？**
- 不同时间步的重要性不同
- 自动关注重要的时间点
- 比简单的平均池化更灵活

### 4. 可学习节点权重

**为什么需要节点权重？**
- 不同频段对任务的重要性不同
- 自动学习最优的频段权重
- 例如：低频可能对趋势预测更重要

### 5. 平均融合

**为什么使用平均融合？**
- 简单有效
- 结合所有频段的信息
- 与节点权重配合，实现加权平均

## 总结

Level=2 时 4 个频段的处理流程：

1. **分解**: 原始信号 → 4 个频段
2. **独立处理**: 每个频段独立投影、编码
3. **注意力池化**: 每个频段内部进行注意力加权池化
4. **权重应用**: 应用可学习的节点权重
5. **融合**: 平均融合所有频段的特征

**关键特点**:
- ✅ 每个频段独立学习特征
- ✅ 注意力机制关注重要时间点
- ✅ 可学习权重自适应频段重要性
- ✅ 最终融合所有频段信息
