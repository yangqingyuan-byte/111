"""
T3Time 原模型 - 完整独立版本
此文件包含了所有必要的依赖代码，可以独立运行，无需项目内的其他文件。
适用于迁移到其他项目（如硕士论文项目）使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional
from torch import Tensor
import math


# ============================================================================
# 辅助函数和工具类
# ============================================================================

def pv(msg, verbose=False):
    """打印函数，用于调试输出"""
    if verbose:
        print(msg)


class Transpose(nn.Module):
    """转置操作模块"""
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: 
            return x.transpose(*self.dims).contiguous()
        else: 
            return x.transpose(*self.dims)


def get_activation_fn(activation):
    """获取激活函数"""
    if callable(activation): 
        return activation()
    elif activation.lower() == "relu": 
        return nn.ReLU()
    elif activation.lower() == "gelu": 
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


def PositionalEncoding(q_len, d_model, normalize=True):
    """位置编码函数"""
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    """2D坐标位置编码"""
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: 
            break
        elif cpe.mean() > eps: 
            x += .001
        else: 
            x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    """1D坐标位置编码"""
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    """位置编码生成函数"""
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': 
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': 
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': 
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': 
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': 
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: 
        raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


# ============================================================================
# Normalize 类 (RevIN 归一化)
# ============================================================================

class Normalize(nn.Module):
    """RevIN (Reversible Instance Normalization) 归一化模块"""
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ============================================================================
# CrossModal 相关类 (跨模态对齐)
# ============================================================================

class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) 
    with optional residual attention from previous layer (Realformer: Transformer likes residual 
    attention by He et al, 2020) and locality self attention (Vision Transformer for Small-Size 
    Datasets by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, 
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: 
            attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: 
            return output, attn_weights, attn_scores
        else: 
            return output, attn_weights


class _MultiheadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, 
                 attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, 
                                                    res_attention=self.res_attention, lsa=lsa)

        # Project output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: 
            K = Q
        if V is None: 
            V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, 
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: 
            return output, attn_weights, attn_scores
        else: 
            return output, attn_weights


class TSTEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='LayerNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", 
                 res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, 
                                            proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.LayerNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.LayerNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, 
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            q = self.norm_attn(q)
            k = self.norm_attn(k)
            v = self.norm_attn(v)
        ## Multi-Head attention
        if self.res_attention:
            q2, attn, scores = self.self_attn(q, k, v, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            q2, attn = self.self_attn(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        q = q + self.dropout_attn(q2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            q = self.norm_attn(q)

        # Feed-forward sublayer
        if self.pre_norm:
            q = self.norm_ffn(q)
        ## Position-wise Feed-Forward
        q2 = self.ff(q)
        ## Add & Norm
        q = q + self.dropout_ffn(q2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            q = self.norm_ffn(q)

        if self.res_attention:
            return q, scores
        else:
            return q


class CrossModal(nn.Module):
    """跨模态对齐模块"""
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='LayerNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
    
        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, q:Tensor, k:Tensor, v:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        q  [bs * nvars x (text_num) x d_model]
        k  [bs * nvars x (text_num) x d_model]
        v  [bs * nvars x (text_num) x d_model]
        '''
        scores = None
        if self.res_attention:
            for mod in self.layers: 
                output, scores = mod(q, k, v, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: 
                output = mod(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


# ============================================================================
# T3Time 模型主类
# ============================================================================

class AdaptiveDynamicHeadsCMA(nn.Module):
    """
    Small network to compute per-head weights from the concatenated heads
    Input shape: [B, N, H*C]      Output shape: [B, N, H]
    """
    def __init__(self, num_heads, num_nodes, channel, device):
        super().__init__()
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.channel = channel
        self.device = device
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_heads * channel, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_heads)
        ).to(device)

    def forward(self, cma_outputs):

        B, C, N = cma_outputs[0].shape
        H = self.num_heads

        combined = torch.cat(cma_outputs, dim=1)                        # [B, H*C, N]
        combined_permute = combined.permute(0, 2, 1)                    # [B, N, H*C]

        gates = self.gate_mlp(combined_permute)                         # raw scores: [B, N, H]
        gates = F.softmax(gates, dim=-1)                                # [B, N, H]

        combined_heads = combined.view(B, H, C, N).permute(0, 1, 3, 2)  # [B, H, N, C]
        gates = gates.permute(0, 2, 1).unsqueeze(-1)                    # [B, H, N, 1]
        
        weighted_heads = combined_heads * gates                         # [B, H, C, N] * [B, H, N, 1] → broadcasting
        weighted_heads = weighted_heads.permute(0, 1, 3, 2)             # back to [B, H, C, N]
        
        fused = weighted_heads.sum(dim=1)                               # [B, C, N]

        return fused


class RichHorizonGate(nn.Module):
    """
    Each channel has its own gate that depends both 
    on the global context (pooled) and on the forecast horizon.
    """
    def __init__(self, embed_dim):

        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, embedding: torch.Tensor, horizon: int) -> torch.Tensor:

        B, C, N = embedding.size()
        pooled_embed = embedding.mean(dim=2)                                # [B, C]
        horizon_tensor = torch.full((B, 1), float(horizon) / 1000.0, device=embedding.device)

        gating_input = torch.cat([pooled_embed, horizon_tensor], dim=1)     # [B, C+1]
        gate = self.gate_mlp(gating_input).unsqueeze(-1)                    # [B, C, 1]
        return gate
    

class FrequencyAttentionPooling(nn.Module):
    """
    Learnable, attention-weighted pooling over frequency bins
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.freq_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, freq_enc_out):

        attn_logits  = self.freq_attention(freq_enc_out)           # [B*N, Lf, 1]
        attn_weights = F.softmax(attn_logits, dim=1)               # normalize over Lf

        pooled_freq  = (freq_enc_out * attn_weights).sum(dim=1)    # [B*N, C]
        return pooled_freq


class TriModal(nn.Module):
    """T3Time 三模态时间序列预测模型"""
    def __init__(
        self,
        device = "cuda:7",
        channel = 32,
        num_nodes = 7,
        seq_len = 96,
        pred_len = 96,
        dropout_n = 0.1,
        d_llm = 768,
        e_layer = 1,
        d_layer = 1,
        d_ff=32,
        head =8
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n= dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head

        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.num_cma_heads = 4 

        # Time Series Encoder
        self.ts_encoder_layer = nn.TransformerEncoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, 
                                                           norm_first = True,dropout = self.dropout_n).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Prompt Encoder
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_llm, nhead = self.head, batch_first=True, 
                                                            norm_first = True,dropout = self.dropout_n).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers = self.e_layer).to(self.device)
        
        # Spectral Encoder
        self.Lf = seq_len // 2 + 1   
        self.freq_token_proj = nn.Linear(1, self.channel).to(self.device)
        self.freq_attn_layer = nn.TransformerEncoderLayer(d_model=self.channel, nhead=self.head, batch_first=True,
                                                            norm_first=True, dropout=self.dropout_n).to(self.device)
        self.freq_encoder = nn.TransformerEncoder(self.freq_attn_layer, num_layers=1).to(self.device)

        self.freq_pool = FrequencyAttentionPooling(self.channel).to(self.device)    # Dynamic frequency‐domain pooling
        self.rich_horizon_gate = RichHorizonGate(self.channel).to(self.device)

        # multi head CMA
        self.cma_heads = nn.ModuleList([
            CrossModal(d_model= self.num_nodes, n_heads= 1, d_ff=self.d_ff, norm='LayerNorm', attn_dropout=self.dropout_n, 
                                dropout=self.dropout_n, pre_norm=True, activation="gelu", res_attention=True, n_layers=1, 
                                store_attn=False).to(self.device)  # single head internally
            for _ in range(self.num_cma_heads)
        ])

        # Aggregate multi heads
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(num_heads=self.num_cma_heads, num_nodes=self.num_nodes, channel=self.channel, device=self.device)
   
        # Residual connection 
        self.residual_alpha = nn.Parameter(torch.ones(self.channel) * 0.5).to(self.device)  

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, norm_first = True, dropout = self.dropout_n).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = self.d_layer).to(self.device)

        # Projection
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        """返回模型参数总数"""
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def frequency_domain_processing(self, input_data):

        freq_complex    = torch.fft.rfft(input_data, dim=-1)    # [B, N, Lf]
        freq_mag        = torch.abs(freq_complex)
        B, N, Lf        = freq_mag.shape
        
        freq_tokens = freq_mag.unsqueeze(-1)                    # [B, N, Lf, 1]
        freq_tokens = freq_tokens.reshape(B*N, Lf, 1)           # [B*N, Lf, 1]
        freq_tokens = self.freq_token_proj(freq_tokens)         # [B*N, Lf, C]

        freq_enc_out = self.freq_encoder(freq_tokens)           # [B*N, Lf, C]
        freq_enc_out = self.freq_pool(freq_enc_out)             # [B*N, C]

        freq_enc_out = freq_enc_out.reshape(B, N, self.channel) # [B, N, C] 
        return freq_enc_out
    

    def forward(self, input_data, input_data_mark, embeddings):
        print("\n" + "="*80)
        print("T3Time Forward Pass - 维度追踪")
        print("="*80)
        
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()
        print(f"[输入] input_data: {input_data.shape}, input_data_mark: {input_data_mark.shape}, embeddings: {embeddings.shape}")
        
        embeddings = embeddings.squeeze(-1)                                 # [B, E, N]
        print(f"[Embeddings处理] embeddings.squeeze(-1): {embeddings.shape}")
        embeddings = embeddings.permute(0,2,1)                              # [B, N, E]
        print(f"[Embeddings处理] embeddings.permute(0,2,1): {embeddings.shape}")

        #------ RevIN
        input_data = self.normalize_layers(input_data, 'norm')
        print(f"[RevIN归一化] input_data after norm: {input_data.shape}")
        input_data = input_data.permute(0,2,1)                              # [B, N, L]
        print(f"[RevIN归一化] input_data.permute(0,2,1): {input_data.shape}")

        #------ Frequency Encoding
        freq_enc_out = self.frequency_domain_processing(input_data)         # [B, N, C]
        print(f"[频域编码] freq_enc_out: {freq_enc_out.shape}")
        input_data = self.length_to_feature(input_data)                     # [B, N, C]
        print(f"[时域嵌入] input_data after length_to_feature: {input_data.shape}")

        #------ Time Series Encoding
        enc_out = self.ts_encoder(input_data)                               # [B, N, C]
        print(f"[时域编码] enc_out after ts_encoder: {enc_out.shape}")
        enc_out = enc_out.permute(0,2,1)                                    # [B, C, N]
        print(f"[时域编码] enc_out.permute(0,2,1): {enc_out.shape}")
  
        #------ Rich Horizon Gate
        gate = self.rich_horizon_gate(enc_out, self.pred_len)               # [B, C, 1]
        print(f"[RichHorizonGate] gate: {gate.shape}, freq_enc_out.permute(0,2,1): {freq_enc_out.permute(0,2,1).shape}")
        enc_out = gate * freq_enc_out.permute(0,2,1) + (1 - gate) * enc_out # [B, C, N]
        print(f"[RichHorizonGate] enc_out after fusion: {enc_out.shape}")
        
        #------ Prompt encoding 
        embeddings = self.prompt_encoder(embeddings)                        # [B, N, E]
        print(f"[Prompt编码] embeddings after prompt_encoder: {embeddings.shape}")
        embeddings = embeddings.permute(0,2,1)                              # [B, E, N]
        print(f"[Prompt编码] embeddings.permute(0,2,1): {embeddings.shape}")

        #------ Aggregating Multiple CMA Heads
        cma_outputs = []
        print(f"[CMA对齐] enc_out: {enc_out.shape}, embeddings: {embeddings.shape}")
        for idx, cma_head in enumerate(self.cma_heads):
            head_out = cma_head(enc_out, embeddings, embeddings)            # [B,C,N]
            cma_outputs.append(head_out)
            print(f"[CMA对齐] CMA Head {idx+1} output: {head_out.shape}")

        fused = self.adaptive_dynamic_heads_cma(cma_outputs)                # [B, C, N]
        print(f"[CMA融合] fused after adaptive_dynamic_heads_cma: {fused.shape}")

        #------ Residual Fusion 
        B, C, N = fused.shape
        alpha = self.residual_alpha.view(1, C, 1)
        print(f"[残差融合] fused: {fused.shape}, enc_out: {enc_out.shape}, alpha: {alpha.shape}")
        cross_out = alpha * fused + (1 - alpha) * enc_out                   # [B, C, N]
        print(f"[残差融合] cross_out: {cross_out.shape}")
        cross_out = cross_out.permute(0, 2, 1)                              # [B, N, C]
        print(f"[残差融合] cross_out.permute(0,2,1): {cross_out.shape}")

        #------ Decoder
        dec_out = self.decoder(cross_out, cross_out)                        # [B, N, C]
        print(f"[解码器] dec_out after decoder: {dec_out.shape}")

        #------ Projection
        dec_out = self.c_to_length(dec_out)                                 # [B, N, L]
        print(f"[投影层] dec_out after c_to_length: {dec_out.shape}")
        dec_out = dec_out.permute(0,2,1)                                    # [B, L, N]
        print(f"[投影层] dec_out.permute(0,2,1): {dec_out.shape}")

        #------ Denorm
        dec_out = self.normalize_layers(dec_out, 'denorm')
        print(f"[RevIN反归一化] dec_out after denorm: {dec_out.shape}")
        print("="*80 + "\n")

        return dec_out


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例：创建模型实例
    model = TriModal(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        channel=32,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        d_llm=768,
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8
    )
    
    print(f"模型参数总数: {model.param_num():,}")
    print(f"可训练参数数: {model.count_trainable_params():,}")
    
    # 示例：前向传播
    batch_size = 2
    input_data = torch.randn(batch_size, 96, 7)  # [B, L, N]
    input_data_mark = torch.randn(batch_size, 96, 4)  # [B, L, mark_dim]
    embeddings = torch.randn(batch_size, 768, 7, 1)  # [B, E, N, 1]
    
    output = model(input_data, input_data_mark, embeddings)
    print(f"输出形状: {output.shape}")  # 应该是 [B, pred_len, N]
