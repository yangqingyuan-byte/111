# T3Time: Tri-Modal Time Series Forecasting via Adaptive Multi-Head Alignment and Residual Fusion

Abdul Monaf Chowdhury<sup>1</sup>, Rabeya Akter<sup>1</sup>, Safaeid Hossain Arib<sup>1</sup>

$^{1}$ Robotics & Mechatronics Engineering, University of Dhaka

{monafabdul15, rabeyaakter231023, safaeid48}  $@$  gmail.com

# Abstract

Multivariate time series forecasting (MTsF) seeks to model temporal dynamics among variables to predict future trends. Transformer-based models and large language models (LLMs) have shown promise due to their ability to capture long-range dependencies and patterns. However, current methods often rely on rigid inductive biases, ignore inter-variable interactions, or apply static fusion strategies that limit adaptability across forecast horizons. These limitations create bottlenecks in capturing nuanced, horizon-specific relationships in time-series data. To solve this problem, we propose T3Time, a novel trimodal framework consisting of time, spectral, and prompt branches, where the dedicated frequency encoding branch captures the periodic structures along with a gating mechanism that learns prioritization between temporal and spectral features based on the prediction horizon. We also proposed a mechanism which adaptively aggregates multiple cross-modal alignment heads by dynamically weighting the importance of each head based on the features. Extensive experiments on benchmark datasets demonstrate that our model consistently outperforms state-of-the-art baselines, achieving an average reduction of  $3.28\%$  in MSE and  $2.29\%$  in MAE. Furthermore, it shows strong generalization in few-shot learning settings: with  $5\%$  training data, we see a reduction in MSE and MAE by  $4.13\%$  and  $1.91\%$ , respectively; and with  $10\%$  data, by  $3.62\%$  and  $1.98\%$  on average.

Code - https://github.com/monaf-chowdhury/T3Time/

# Introduction

Multivariate time-series forecasting (MTsF) lies at the heart of modern decision-making, powering everything from energy-load balancing (Liu et al. 2023a) and urban traffic management (Liu et al. 2024c) to high-frequency trading (Xu et al. 2021) and weather forecasting (Schneider and Dickinson 1974). While the objective appears straightforward—predicting future values based on past observations—the underlying challenge is profoundly complex. Effective models must simultaneously capture short-term temporal fluctuations, long-range dependencies, and intricate inter-variable dynamics, all while maintaining computational efficiency and robustness in data-sparse regimes.

Recent advances in deep learning have led to the development of numerous models for time series forecasting (Miao et al. 2024; Challu et al. 2023). However, early approaches were constrained by limited parameters and poor representations (Liu et al. 2022a), hindering generalization. Following the development of transformer-based architectures (Vaswani et al. 2017), frameworks like Informer (Zhou et al. 2021), Autoformer (Wu et al. 2021), and Fedformer (Zhou et al. 2022) addressed the quadratic complexity of standard self-attention and introduced trend-seasonal decomposition mechanisms.

Subsequent efforts have advanced in two complementary directions for better representation learning in the subspace of time-series. The first focuses on token restructuring, exemplified by PatchTST (Nie et al. 2023), which segments time series into sub-series patches and processes each channel independently. The second explores alternative representation spaces; for instance, Freeformer (Yue et al. 2025) operates in the frequency domain, demonstrating that spectral tokens can encode global periodic patterns more compactly than conventional time-domain attention. However, single-modality encoder architectures remain limited in their capacity to fully capture the intricate and multi-scale structure of temporal dynamics.

To make contextual time-series representations even robust, pre-trained LLM based (Radford et al. 2018) prompting techniques have been utilized in some frameworks. Prompts are either used to encapsulate the time-series information (Jia et al. 2024; Huang et al. 2024) or to provide further context to the time series (Jin et al. 2023; Xue and Salim 2023; Xue, Voutharoja, and Salim 2022). Although time and prompt-based dual modal embeddings have accomplished promising performance, they have been plagued with embedding overlapping issues (Chang et al. 2024; Jia et al. 2024), which weakened representation integrity as seen here by Fig. 1(a).

Recently, TimeCMA (Liu et al. 2025) has presented cross-modal alignment to integrate temporal-language dual modalities to represent embeddings as disentangled yet robust. Regardless, they miss out on aligning spectral modality to further represent global periodicity. By and large, existing MTSF frameworks exhibit three fundamental limitations. First, existing models often adopt a modality-isolated architecture, emphasizing a single modality while disregarding

![](images/71ace932f1c03410821dd54088ce5cdd6114633fb68c79070d4301070d9f512f.jpg)



Figure 1: Comparison of bimodal vs. tri-modal framework for time series forecasting. (a) Bimodal models use static fusion of time and prompt features, lacking frequency awareness and horizon adaptivity. (b) T3Time introduces tri-modal encoding with horizon-aware gating and adaptive multi-head cross-modal alignment for robust, horizontal-sensitive representations.


the other, which leads to fragmented representations. Second, if they utilize multi-modalities, they suffer from limited alignment capacity that constrain the model's ability to capture rich, fine-grained interactions between multi modalities. Third, these models exhibit horizon rigidity by applying static processing across forecast lengths, ultimately hindering their ability to adapt modality emphasis based on the temporal scope of the prediction.

To address these challenges, we propose T3Time - illustrated in Fig. 1(b), a tri-modal framework for MTSF that integrates temporal, spectral, and prompt-based semantic representations. Each modality captures complementary structures in the data. To enable effective multimodal integration, T3Time employs an adaptive multi-head cross-modal alignment module that dynamically weighs modality-specific embeddings based on their relevance to the forecasting task. Additionally, a horizon-aware gating mechanism modulates the influence of temporal and spectral features by prediction horizon, while a channel-wise residual fusion preserves variable-specific priors and enhances representational granularity. These components collectively enable T3Time to learn horizon-aware, variable-sensitive representations for robust generalization and improved forecasting. Our contributions are summarized as follows:

- We propose T3Time, a novel tri-modal forecasting framework that unifies temporal, spectral, and prompt-based semantic representations via an adaptive multi-head cross-modal alignment mechanism. This design enables dynamic, content-aware fusion of heterogeneous modalities for more expressive and context-aware representation.

- We introduce horizon aware gating module and channelwise residual fusion mechanism to elevate temporal adaptability and fine-grained feature representation.

- T3Time consistently outperforms state-of-the-art baselines and demonstrates strong generalization across benchmark datasets.

# Related Work

Forecasting in Time Domain. Time series forecasting has witnessed significant advances driven by deep learning

(Lai et al. 2018; Franceschi, Dieuleveut, and Jaggi 2019; Jin et al. 2022), with Transformer-based architectures (Wen et al. 2022) emerging as powerful tools due to their capacity to model long-range dependencies. Early transformer-based methods (Zhou et al. 2021; Wu et al. 2021; Liu et al. 2022b) introduce various attention mechanisms to reduce the quadratic complexity of standard self-attention and improve the efficiency of forecasting. However, most of them rely on fixed inductive biases (e.g., decomposition, sparse priors) and can not adapt to varying forecasting horizons. iTransformer (Liu et al. 2023b) models each time series as an independent token, enabling flexible cross-variable attention. Regardless, its reliance on a single-stream encoder and static fusion limits its ability to capture complex, modality-specific dynamics in multivariate forecasting.

Forecasting in Frequency Domain. Frequency-domain methods (Cao et al. 2020; Woo et al. 2022; Sun and Boning 2022; Yi et al. 2023; Chen et al. 2023) offer another perspective by modeling temporal periodicity and long-term structure. Autoformer (Wu et al. 2021) integrates series decomposition with autocorrelation in the frequency domain, while FEDformer (Zhou et al. 2022) further enhances this approach employing a mixture-of-expert structure and Fourier-based attention. Although effective, these methods typically employ fixed or global spectral representations, lacking mechanisms for adapting frequency-domain importance based on forecast length or contextual features.

Cross-modal Alignment. Recent work has explored adapting large language models (LLMs) (Lu et al. 2021) for time series forecasting, either by replacing standard tokenizers with learned embeddings for time series inputs (Zhou et al. 2023; Liu et al. 2024a) or by formatting time series as textual prompts (Xue and Salim 2023; Jin et al. 2023; Pan et al. 2024). Although these methods leverage the representational power of LLMs, they often face modality mismatches. Moreover, fusion between language and numerical representations is typically static, limiting adaptability across tasks. Several studies have proposed the transfer of knowledge from pre-trained models using self-supervised learning (Zhang et al. 2022; Deldari et al. 2022; Zhang et al. 2024), multimodal reprogramming (Chen 2024), or instruction-based fine-tuning (Yin et al. 2024). TimeCMA (Liu et al. 2025) represents a recent effort in this direction, introducing a cross-modal alignment to integrate time series and prompt embeddings for multivariate forecasting. However, its use of a single head alignment mechanism can be limiting, as it constrains the model's ability to capture diverse and fine-grained interactions between semantic and temporal signals. To this end, we introduce trimodal encodings to better represent time series representations and ensure robust performance.

# Methodology

Our three-stage framework, as illustrated in Figure 2 contains Tri-Modal Encoding, Adaptive Multi-Head CrossModal Alignment (CMA), and channel-wise residual connection. Tri-Modal Encoding comprises three branches: the frequency encoding branch, the time series encoding branch

![](images/6bfb8e41bb5c76f0ea696cbd9703adebfd92eb33969b540596597214cecba321.jpg)



Figure 2: Overview of our framework. The model comprises tri-modal encoding (time, frequency, prompt), a horizon-aware gating module for dynamic temporal-spectral fusion, adaptive multi-head cross-modal alignment with per-head importance weighting, and a channel-wise residual connection for fine-grained representation mixing prior to decoding.


and the LLM encoding branch, designed to extract three different representations from the input time series. We also designed a representation-rich horizon-gating to fuse the frequency and time encoder branches dynamically based on the forecast horizon. Adaptive Multi-head CMA, inspired by (Vaswani et al. 2017), aligns the fused temporal-spectral features with the prompt embeddings using multiple instances of a Cross-Modal Attention module. Each attention head independently computes cross-attention between the fused temporal-spectral embeddings and prompt embeddings. Adaptive head fusion mechanism combines all the attention heads based on their importance and generates an expressive aligned representation. Finally, Channelwise Residual Connection creates a fine-grained residual fusion between original temporal-spectral embeddings and aligned cross-modal representations, which are ultimately passed through the decoder for forecasting.

# Tri-Modal Encoding

Frequency Encoding Branch: To capture periodic and frequency-aware patterns from the time series, we designed a dedicated Frequency Encoding Branch, which transforms raw temporal inputs into the frequency domain using the real-valued fast Fourier transform and processes the resulting spectral features through a Transformer-based encoder.

Given the normalized input  $\mathbf{X}_t\in \mathbb{R}^{B\times N\times L}$ , where  $B$  is the batch size,  $N$  is the number of variables (or nodes), and  $L$  is the sequence length, we apply the real-valued Fourier

transform along the temporal dimension to obtain complex-valued spectra:

$$
\widehat {\mathbf {X}} _ {t} = \mathcal {F} _ {r} \left(\mathbf {X} _ {t}\right) \in \mathbb {C} ^ {B \times N \times L _ {f}}, \quad L _ {f} = \left\lfloor \frac {L}{2} \right\rfloor + 1 \tag {1}
$$

Only the magnitude spectrum,  $\mathbf{F}$ , is retained as input to the frequency encoder. Each frequency bin is treated as a token. The tensor  $\mathbf{F}$  is reshaped to  $(BN)\times L_{f}$  and projected to the embedding dimension  $C$  using a learnable projection matrix  $\mathbf{W}_f\in \mathbb{R}^{C\times 1}$ :

$$
\mathbf {Z} _ {f} = \phi (\mathbf {F} \cdot \mathbf {W} _ {f} ^ {\top}) \in \mathbb {R} ^ {B N \times L _ {f} \times C} \tag {2}
$$

where  $\phi (\cdot)$  denotes an element-wise ReLU nonlinearity.

To model dependencies across frequency components, the projected frequency tokens are passed through a single-layer Transformer encoder. Let,  $\mathcal{T}$  denote the self-attention block with pre-normalization. The encoded representation is given by:

$$
\widetilde {\mathbf {Z}} _ {f} = \mathcal {T} \left(\mathbf {Z} _ {f}\right) \in \mathbb {R} ^ {B N \times L _ {f} \times C} \tag {3}
$$

To aggregate the encoded spectral features, we compute a learnable attention-weighted pooling over the frequency bins to summarize the frequency information for each feature. Specifically, we first project the encoded frequency tokens  $\widetilde{\mathbf{Z}}_f\in \mathbb{R}^{BN\times L_f\times C}$  through a two-layer perceptron consisting of weights  $\mathbf{W}_1\in \mathbb{R}^{C\times d}$  and  $\mathbf{W}_2\in \mathbb{R}^{d\times 1}$ , with an intermediate nonlinearity  $\phi (\cdot)$ . The resulting scalar logits are

normalized via a softmax function across the frequency dimension  $L_{f}$ , yielding attention weights  $\alpha \in \mathbb{R}^{BN\times L_f\times 1}$ :

$$
\boldsymbol {\alpha} = \frac {\exp \left(\left[ \phi \left(\widetilde {\mathbf {Z}} _ {f} \mathbf {W} _ {1}\right) \right] \mathbf {W} _ {2}\right)}{\sum_ {j = 1} ^ {L _ {f}} \exp \left(\left[ \phi \left(\widetilde {\mathbf {Z}} _ {f} \mathbf {W} _ {1}\right) \mathbf {W} _ {2} \right] _ {j}\right)} \in \mathbb {R} ^ {B N \times L _ {f} \times 1} \tag {4}
$$

The final pooled embedding is computed as a weighted sum:

$$
\mathbf {F} _ {\text {p o o l e d}} = \sum_ {l = 1} ^ {L _ {f}} \alpha_ {l} \cdot \widetilde {\mathbf {Z}} _ {f,: l} \in \mathbb {R} ^ {B N \times C} \tag {5}
$$

Eventually, the pooled frequency features are reshaped back to match the node dimension,  $\tilde{\mathbf{F}}\in \mathbb{R}^{B\times N\times C}$ . This frequency-aware representation  $\tilde{\mathbf{F}}\in \mathbb{R}^{B\times N\times C}$  captures periodic patterns and serves as one of the three modality-specific encodings for subsequent fusion.

Time Series Encoding Branch: To model temporal dependencies and evolving patterns in raw time-domain signals, we construct a dedicated Time Series Encoding Branch. This branch transforms normalized time series into contextualized representations using a shared projection followed by a Transformer-based encoder.

Given the normalized input  $\mathbf{X}_t\in \mathbb{R}^{B\times N\times L}$ , we project the temporal dimension into a latent embedding space of dimension  $C$  using a shared learnable projection matrix  $\mathbf{W}_t\in \mathbb{R}^{L\times C}$ :

$$
\mathbf {Z} _ {t} = \mathbf {X} _ {t} \mathbf {W} _ {t} \in \mathbb {R} ^ {B \times N \times C} \tag {6}
$$

This projection treats each node's input sequence as a single vector, linearly embedding the temporal axis into  $C$  features per node. To model interactions and dependencies across temporal patterns within each node, we apply a Transformer encoder with pre-normalization. Let  $T_{t}$  denote the time-domain Transformer encoder:

$$
\widetilde {\mathbf {Z}} _ {t} = \mathcal {T} _ {t} \left(\mathbf {Z} _ {t}\right) \in \mathbb {R} ^ {B \times N \times C} \tag {7}
$$

$\widetilde{\mathbf{Z}}_t$  provides positionally-aware, temporally contextualized embeddings for each node and serves as time modality-specific representation in our tri-modal framework.

LLM Encoding Branch: To inject external priors and semantic structure into the forecasting model, we introduce a dedicated LLM Encoding Branch, which leverages a pre-trained frozen GPT-2 (Radford et al. 2019) to encode prompt-based descriptions of input time series segments.

Given the normalized input series  $\mathbf{X}_t\in \mathbb{R}^{B\times N\times L}$  and their associated temporal markers  $\mathbf{M}_t\in \mathbb{R}^{B\times L\times D}$  (e.g., date, hour), we generate natural language prompts describing the sequence statistics. Each prompt is constructed for every feature. Prompt generation strategy is further discussed in Appendix B-1.

Each prompt is tokenized using a GPT-2 tokenizer and fed into a pre-trained GPT-2 model. Let  $\mathcal{L}(\cdot)$  denote the language model. For a given tokenized prompt  $\mathbf{P}_{i,j}$  associated with the  $j$ -th node in the  $i$ -th sample, we obtain:

$$
\mathbf {E} _ {i, j} = \mathcal {L} \left(\mathbf {P} _ {i, j}\right) \in \mathbb {R} ^ {T \times d _ {\mathrm {L L M}}} \tag {8}
$$

where  $T$  is the number of tokens in the prompt and  $d_{\mathrm{LLM}}$  is the embedding dimension of the language model (e.g., 768 for GPT-2). Since different prompts may produce variable-length token sequences, we uniformly pad the output with copies of the final token embedding to ensure consistent shape across all samples and features.

After batching, we extract the final token embedding per prompt as a compact summary:

$$
\mathbf {Z} _ {\text {L L M}} [ i, j ] = \mathbf {E} _ {i, j} [ - 1 ] \in \mathbb {R} ^ {d _ {\text {L L M}}} \tag {9}
$$

The final LLM-derived representation is a tensor  $\mathbf{Z}_{\mathrm{LLM}} \in \mathbb{R}^{B \times N \times d_{\mathrm{LLM}}}$ , which is projected and passed through a Transformer encoder before being used in downstream cross-modal fusion.

We introduce a Horizon-Aware Gating Module to adaptively balance contributions from different modalities based on the forecast horizon. The core intuition is that short-term forecasts may benefit more from time-localized representations, whereas long-range forecasts can better leverage global periodic patterns captured in the frequency domain.

Time encoding,  $\widetilde{\mathbf{Z}}_t\in \mathbb{R}^{B\times N\times C}$ , is first pooled over the feature dimension to obtain a global summary per sample. We normalize the prediction length by a constant factor and concatenate it with the pooled time encoding in order to provide the forecast length as a continuous conditioning signal,  $\mathbf{g}_{\mathrm{in}}\in \mathbb{R}^{B\times (C + 1)}$ . The concatenated vector,  $\mathbf{g}_{\mathrm{in}}$ , is processed through a lightweight two-layer MLP followed by a sigmoid nonlinearity to produce channel-wise gating weights:

$$
\mathbf {g} = \sigma \left(\mathbf {W} _ {4} \cdot \phi \left(\mathbf {W} _ {3} \cdot \mathbf {g} _ {\mathrm {i n}} ^ {\top}\right)\right) ^ {\top} \in \mathbb {R} ^ {B \times C} \tag {10}
$$

where  $\mathbf{W}_3\in \mathbb{R}^{(C + 1)\times d}$ ,  $\mathbf{W}_4\in \mathbb{R}^{d\times C}$ ,  $\sigma (\cdot)$  denotes the element-wise sigmoid nonlinearity.

The output of this module is a horizon-aware convex combination of the frequency and time representations:

$$
\mathbf {Z} _ {\mathrm {g}} = \mathbf {g} \odot \tilde {\mathbf {F}} + (1 - \mathbf {g}) \odot \tilde {\mathbf {Z}} _ {t} \in \mathbb {R} ^ {B \times C \times N} \tag {11}
$$

where  $\odot$  denotes element-wise multiplication broadcast over the feature dimension. This gating mechanism enables the model to adaptively shift focus between temporally localized and spectrally global features, as a function of both the input content and the desired forecast horizon.

# Adaptive Dynamic Head Cross-Modal Alignment

To integrate heterogeneous contextual representations from the time-spectral and semantic domains, we follow the cross-modal alignment strategy introduced in (Liu et al. 2025), wherein the time series encoder output is aligned with the prompt encoder output through cross-attention. However, instead of relying on a single head cross-modal alignment as in the original approach, we extend this paradigm by introducing adaptive dynamic head fusion, where multiple CMA heads are independently learned and their outputs are dynamically fused based on data-dependent gating scores.

Each CMA head  $h$  maps the fused representation from the time and frequency branches,  $\mathbf{Z}_{\mathrm{g}} \in \mathbb{R}^{B \times C \times N}$ , and prompt embeddings,  $\mathbf{Z}_{\mathrm{LLM}} \in \mathbb{R}^{B \times E \times N}$ , into a head-specific

aligned output  $\mathbf{H}^{(h)}\in \mathbb{R}^{B\times C\times N}$  via an independent crossattention mechanism with queries from  $\mathbf{Z}_{\mathrm{g}}$  and keys/values from  $\mathbf{Z}_{\mathrm{LLM}}$ . Rather than aggregating these heads through static averaging or fixed linear projections, head-wise outputs are aggregated via a feature-aware, head-adaptive fusion to dynamically weight the importance of each head per feature. Necessarily, the outputs are first concatenated along the channel dimension and transposed to shape  $\mathbb{R}^{B\times N\times HC}$ :

$$
\mathbf {U} = \left[ \mathbf {H} ^ {(1)}; \dots ; \mathbf {H} ^ {(H)} \right] ^ {\top} \in \mathbb {R} ^ {B \times N \times H C} \tag {12}
$$

Each node's fused embedding vector  $\mathbf{U}_{b,n} \in \mathbb{R}^{HC}$  (for batch index  $b$  and node index  $n$ ) is passed through a two-layer gating network to compute importance scores  $\pi_{b,n} \in \mathbb{R}^H$  over the CMA heads:

$$
\mathbf {e} _ {b, n} = \mathbf {W} _ {6} \phi \left(\mathrm {L N} \left(\mathbf {W} _ {5} \mathbf {U} _ {b, n} ^ {\top}\right)\right) \in \mathbb {R} ^ {H}, \tag {13}
$$

$$
\pi_ {b, n} ^ {(h)} = \frac {\exp \left(e _ {b , n} ^ {(h)}\right)}{\sum_ {j = 1} ^ {H} \exp \left(e _ {b , n} ^ {(j)}\right)}, \quad \sum_ {h = 1} ^ {H} \pi_ {b, n} ^ {(h)} = 1 \tag {14}
$$

where  $\mathbf{W}_5\in \mathbb{R}^{128\times HC}$  and  $\mathbf{W}_6\in \mathbb{R}^{H\times 128}$  are learnable matrices,  $\phi (\cdot)$  is a pointwise ReLU nonlinearity, and  $\mathrm{LN}(\cdot)$  denotes layer normalization.

Let  $\mathbf{H}_{b,:,n}^{(h)}\in \mathbb{R}^C$  be the  $h$ -th head output for sample  $b$  and node  $n$ . The gated fusion is obtained via a convex combination over the head dimension, weighted by the attention scores  $\pi_{b,n}^{(h)}$ :

$$
\boldsymbol {\Lambda} _ {b,: n} = \sum_ {h = 1} ^ {H} \pi_ {b, n} ^ {(h)} \cdot \mathbf {H} _ {b,: n} ^ {(h)} \in \mathbb {R} ^ {C} \tag {15}
$$

yielding the final cross-modally aligned representation  $\Lambda \in \mathbb{R}^{B\times C\times N}$  across all  $B$  samples and  $N$  features.

# Channel-wise Residual Connection

Before decoding, we apply a channel-wise residual fusion to reconcile horizon-aware spectral-temporal features with cross-modal alignment outputs, allowing each latent feature to balance its dependence on intrinsic patterns and external priors.

Here,  $\mathbf{Z}_{\mathrm{g}} \in \mathbb{R}^{B \times C \times N}$  denotes the horizon-gated fusion of time-domain and frequency-domain encodings, and  $\Lambda \in \mathbb{R}^{B \times C \times N}$  denotes the output of the adaptive multi-head cross-modal alignment. We learn a set of trainable channel-wise residual coefficients  $\gamma \in \mathbb{R}^{C}$  that modulate the importance of the two streams during the fusion process.

The fused representation  $\Theta \in \mathbb{R}^{B\times C\times N}$  is computed as a convex combination along the channel axis:

$$
\boldsymbol {\Theta} _ {b, c, n} = \gamma_ {c} \odot \boldsymbol {\Lambda} _ {b, c, n} + (1 - \gamma_ {c}) \odot \mathbf {Z} _ {\mathrm {g}, b, c, n} \tag {16}
$$

where  $\odot$  denotes element-wise multiplication,  $b$ ,  $c$ , and  $n$  index the batch, channel, and node dimensions respectively, and  $\gamma_{c} \in [0,1]$  is a learnable scalar specific to channel  $c$ .

This  $\Theta \in \mathbb{R}^{B\times C\times N}$  formulation allows each latent dimension to adaptively balance cross-modal information

against temporal and spectral evidence, enabling fine-grained control over representational mixing. The fused representation  $\Theta$  is subsequently passed into the decoder for final forecasting.

Decoder: Fused representation  $\Theta \in \mathbb{R}^{B\times C\times N}$ , which encapsulates spectro-temporal priors and semantically grounded alignment, is transposed to match the input format of the transformer decoder. The decoder module, composed of  $D$  stacked multi-head cross-attention blocks with pre-normalization, integrates global dependencies across nodes and channels.

Let  $\mathcal{D}(\cdot)$  denote the Transformer decoder. The contextualized output  $\mathbf{Z}_d\in \mathbb{R}^{B\times N\times C}$  is given by:

$$
\mathbf {Z} _ {d} = \mathcal {D} \left(\boldsymbol {\Theta} ^ {\top}, \boldsymbol {\Theta} ^ {\top}\right), \quad \boldsymbol {\Theta} ^ {\top} \in \mathbb {R} ^ {B \times N \times C} \tag {17}
$$

Following decoding, the representation  $\mathbf{Z}_d$  is linearly projected along the channel axis to produce the final forecast sequence of length  $L_{p}$ :

$$
\widehat {\mathbf {Y}} = \mathbf {Z} _ {d} \mathbf {W} _ {p} ^ {\top} + \mathbf {b} _ {p}, \quad \mathbf {W} _ {p} \in \mathbb {R} ^ {L _ {p} \times C}, \quad \widehat {\mathbf {Y}} \in \mathbb {R} ^ {B \times N \times L _ {p}} \tag {18}
$$

Forecast tensor  $\widehat{\mathbf{Y}}$  is transposed to match the original input format  $\mathbf{Y} \in \mathbb{R}^{B \times L_p \times N}$ , representing the final output.

# Experiments

T3Time demonstrates consistent and superior performance across a wide range of benchmarks, particularly excelling in long-horizon forecasting and low-data regimes. To ensure fair comparison, we adhere strictly to the experimental protocol established by Liu et al. (2025) across all baseline models unless explicitly stated otherwise. Our evaluation includes a comprehensive set of strong baselines: prompt-based LLMs such as TimeCMA (Liu et al. 2025), Time-LLM (Jin et al. 2023), and UniTime (Liu et al. 2024b); Transformer-based architectures including iTransformer (Liu et al. 2023b), OFA (GPT4TS) (Zhou et al. 2023), PatchTST (Nie et al. 2023), and Fedformer (Zhou et al. 2022); the linear model DLinear (Zeng et al. 2023); and the CNN-based framework TimesNet (Wu et al. 2022). Details regarding device, datasets, and evaluation metrics are provided in Appendix A.

# Long-Term Forecasting

Setup. We evaluate T3Time on eight widely-used multivariate time series benchmarks: ETTh1, ETTh2, ETTm1, ETTm2, ECL, Weather, ILI, and Exchange. Following the standardized protocol from Liu et al. (2025), we set the input sequence length to 96 and vary the forecasting horizon across \{96, 192, 336, 720\}, except for ILI where we follow the \{24, 36, 48, 60\} prediction steps. Mean squared error (MSE) and mean average error (MAE) are used as evaluation metrics for all our experiments.

Results. Table 9 summarizes the forecasting performance across all datasets. On average, T3Time consistently achieves state-of-the-art (SOTA) results in 14 out of 16 baselines. The model yields the lowest error across most horizons, outperforming recent competitive approaches such as

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Ours</td><td colspan="2">TimeCMA</td><td colspan="2">TimeLLM</td><td colspan="2">UniTime</td><td colspan="2">TimesNet</td><td colspan="2">DLinear</td><td colspan="2">iTransformer</td><td colspan="2">PatchTST</td><td colspan="2">OFA</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTm1</td><td>0.372</td><td>0.393</td><td>0.380</td><td>0.392</td><td>0.410</td><td>0.409</td><td>0.385</td><td>0.399</td><td>0.400</td><td>0.406</td><td>0.403</td><td>0.407</td><td>0.407</td><td>0.410</td><td>0.392</td><td>0.402</td><td>0.396</td><td>0.401</td></tr><tr><td>ETTm2</td><td>0.279</td><td>0.322</td><td>0.275</td><td>0.323</td><td>0.296</td><td>0.340</td><td>0.293</td><td>0.334</td><td>0.291</td><td>0.333</td><td>0.350</td><td>0.401</td><td>0.288</td><td>0.332</td><td>0.285</td><td>0.328</td><td>0.294</td><td>0.339</td></tr><tr><td>ETTh1</td><td>0.418</td><td>0.430</td><td>0.423</td><td>0.431</td><td>0.448</td><td>0.443</td><td>0.442</td><td>0.448</td><td>0.458</td><td>0.450</td><td>0.456</td><td>0.452</td><td>0.454</td><td>0.447</td><td>0.463</td><td>0.449</td><td>0.457</td><td>0.450</td></tr><tr><td>ETTh2</td><td>0.348</td><td>0.390</td><td>0.372</td><td>0.397</td><td>0.381</td><td>0.404</td><td>0.378</td><td>0.403</td><td>0.414</td><td>0.427</td><td>0.559</td><td>0.515</td><td>0.383</td><td>0.407</td><td>0.395</td><td>0.414</td><td>0.389</td><td>0.414</td></tr><tr><td>ECL</td><td>0.170</td><td>0.266</td><td>0.174</td><td>0.269</td><td>0.195</td><td>0.288</td><td>0.216</td><td>0.306</td><td>0.192</td><td>0.295</td><td>0.212</td><td>0.300</td><td>0.178</td><td>0.270</td><td>0.207</td><td>0.289</td><td>0.217</td><td>0.308</td></tr><tr><td>Weather</td><td>0.244</td><td>0.275</td><td>0.250</td><td>0.276</td><td>0.275</td><td>0.291</td><td>0.253</td><td>0.276</td><td>0.259</td><td>0.287</td><td>0.265</td><td>0.317</td><td>0.258</td><td>0.278</td><td>0.257</td><td>0.280</td><td>0.279</td><td>0.297</td></tr><tr><td>ILI</td><td>1.705</td><td>0.835</td><td>1.922</td><td>0.921</td><td>2.432</td><td>1.012</td><td>2.108</td><td>0.929</td><td>2.139</td><td>0.931</td><td>2.616</td><td>1.090</td><td>2.444</td><td>1.203</td><td>2.388</td><td>1.011</td><td>2.623</td><td>1.060</td></tr><tr><td>Exchange</td><td>0.353</td><td>0.401</td><td>0.395</td><td>0.429</td><td>0.372</td><td>0.416</td><td>0.364</td><td>0.404</td><td>0.416</td><td>0.443</td><td>0.354</td><td>0.414</td><td>0.360</td><td>0.403</td><td>0.390</td><td>0.429</td><td>0.519</td><td>0.500</td></tr><tr><td>1st Count</td><td>7</td><td>7</td><td>1</td><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>


Table 1: Multivariate forecasting results. All results are averaged from four different forecasting horizons:  $\mathbf{H} \in \{96, 192, 336, 720\}$  for the input sequence length 96. Bold: the best, underline: the second best. Full results are in Appendix C.


<table><tr><td rowspan="2">Dataset</td><td colspan="2">Ours</td><td colspan="2">TimeCMA</td><td colspan="2">TimeLLM</td><td colspan="2">GPT4TS</td><td colspan="2">TimesNet</td><td colspan="2">DLinear</td><td colspan="2">PatchTST</td><td colspan="2">Fedformer</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTm1</td><td>0.376</td><td>0.398</td><td>0.387</td><td>0.410</td><td>0.404</td><td>0.427</td><td>0.464</td><td>0.441</td><td>0.677</td><td>0.537</td><td>0.411</td><td>0.429</td><td>0.501</td><td>0.466</td><td>0.722</td><td>0.605</td></tr><tr><td>ETTm2</td><td>0.266</td><td>0.327</td><td>0.312</td><td>0.358</td><td>0.277</td><td>0.323</td><td>0.293</td><td>0.335</td><td>0.320</td><td>0.353</td><td>0.316</td><td>0.368</td><td>0.296</td><td>0.343</td><td>0.463</td><td>0.488</td></tr><tr><td>ETTh1</td><td>0.449</td><td>0.454</td><td>0.480</td><td>0.479</td><td>0.556</td><td>0.522</td><td>0.590</td><td>0.525</td><td>0.869</td><td>0.628</td><td>0.691</td><td>0.600</td><td>0.633</td><td>0.542</td><td>0.639</td><td>0.561</td></tr><tr><td>ETTh2</td><td>0.357</td><td>0.388</td><td>0.398</td><td>0.433</td><td>0.370</td><td>0.394</td><td>0.397</td><td>0.421</td><td>0.479</td><td>0.465</td><td>0.605</td><td>0.538</td><td>0.415</td><td>0.431</td><td>0.466</td><td>0.475</td></tr><tr><td>Weather</td><td>0.226</td><td>0.268</td><td>0.229</td><td>0.272</td><td>0.234</td><td>0.273</td><td>0.238</td><td>0.275</td><td>0.279</td><td>0.301</td><td>0.241</td><td>0.283</td><td>0.242</td><td>0.279</td><td>0.284</td><td>0.324</td></tr><tr><td>1st Count</td><td>5</td><td>4</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>


Table 2: Few-shot learning on  ${10}\%$  training data. All results are averaged from four different forecasting horizons:  $\mathbf{H} \in$ $\{ {96},{192},{336},{720}\}$  for the input sequence length 512. Bold: the best,underline: the second best. Full results are in Appendix D.


<table><tr><td rowspan="2">Dataset</td><td colspan="2">Ours</td><td colspan="2">TimeCMA</td><td colspan="2">TimeLLM</td><td colspan="2">GPT4TS</td><td colspan="2">TimesNet</td><td colspan="2">DLinear</td><td colspan="2">PatchTST</td><td colspan="2">Fedformer</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTm1</td><td>0.384</td><td>0.405</td><td>0.396</td><td>0.416</td><td>0.425</td><td>0.434</td><td>0.472</td><td>0.450</td><td>0.717</td><td>0.561</td><td>0.400</td><td>0.417</td><td>0.526</td><td>0.476</td><td>0.730</td><td>0.592</td></tr><tr><td>ETTm2</td><td>0.267</td><td>0.330</td><td>0.329</td><td>0.367</td><td>0.274</td><td>0.323</td><td>0.308</td><td>0.346</td><td>0.344</td><td>0.372</td><td>0.399</td><td>0.426</td><td>0.314</td><td>0.352</td><td>0.381</td><td>0.404</td></tr><tr><td>ETTh1</td><td>0.442</td><td>0.451</td><td>0.472</td><td>0.470</td><td>0.627</td><td>0.543</td><td>0.681</td><td>0.560</td><td>0.925</td><td>0.647</td><td>0.750</td><td>0.611</td><td>0.694</td><td>0.569</td><td>0.658</td><td>0.562</td></tr><tr><td>ETTh2</td><td>0.357</td><td>0.403</td><td>0.395</td><td>0.430</td><td>0.382</td><td>0.418</td><td>0.400</td><td>0.433</td><td>0.439</td><td>0.448</td><td>0.694</td><td>0.577</td><td>0.827</td><td>0.615</td><td>0.463</td><td>0.454</td></tr><tr><td>Weather</td><td>0.226</td><td>0.269</td><td>0.231</td><td>0.273</td><td>0.260</td><td>0.309</td><td>0.263</td><td>0.301</td><td>0.298</td><td>0.318</td><td>0.263</td><td>0.308</td><td>0.269</td><td>0.303</td><td>0.309</td><td>0.353</td></tr><tr><td>1st Count</td><td>5</td><td>5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 3: Few-shot learning on  $5\%$  training data. All results are averaged from four different forecasting horizons:  $\mathbf{H} \in \{96,192,336,720\}$  for the input sequence length 512. Bold: the best, underline: the second best. Full results are in Appendix D.

TimeLLM, (Jin et al. 2023) and iTransformer, (Liu et al. 2023b) by  $11.28\%$  in MSE and  $6.20\%$  in MAE, and  $8.86\%$  in MSE and  $6.10\%$  in MAE, respectively. When compared to the strongest prompt-based model, TimeCMA, (Liu et al. 2025), T3Time reduces the MSE by up to  $4.36\%$  on average, while demonstrating more stable MAE performance across long horizons. Overall, T3Time achieves an average MSE reduction of  $3.28\%$  and an MAE reduction of  $2.29\%$  compared to the SOTA baselines. A comprehensive overview of our full results is provided in Appendix C.

# Few-Shot Forecasting

**Setup.** Over the years the foundation models have shown exceptional performance in generalization tasks by few-shot learning or zero-shot learning settings (Brown et al. 2020;

Achiam et al. 2023). To evaluate the generalization performance of T3Time under few-shot conditions, we follow the setups from Jin et al. (2023) for a fair comparison. Specifically, we adopt a scenario where the training data consists of  $10\%$  and  $5\%$  of the available time steps, with input sequence length set to 512. We evaluate T3Time on the same benchmarks used in the long-term forecasting experiments.

Results. Tables 2 and 3 summarize the results for  $10\%$  and  $5\%$  few-shot learning, respectively. In both settings, T3Time outperforms almost all SOTA baselines. Specifically, for the recent SOTA models such as TimeCMA, TimeLLM, and GPT4TS (OFA) in the  $10\%$  few-shot task, average MSE is reduced by  $7.13\%$ ,  $7.42\%$ , and  $13.44\%$ , respectively. T3Time reduces average MSE and MAE by  $3.62\%$  and  $1.98\%$  for  $10\%$  few-shot forecasting tasks in all

<table><tr><td rowspan="2">Design</td><td colspan="2">ETTm1</td><td colspan="2">ETTm2</td><td colspan="2">ETTh1</td><td colspan="2">ETTh2</td><td colspan="2">Weather</td><td colspan="2">ILI</td><td colspan="2">Exchange Rate</td><td rowspan="2">1st Count</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>Our Model</td><td>0.372</td><td>0.393</td><td>0.279</td><td>0.322</td><td>0.418</td><td>0.430</td><td>0.348</td><td>0.390</td><td>0.244</td><td>0.275</td><td>1.705</td><td>0.835</td><td>0.353</td><td>0.401</td><td>14</td></tr><tr><td>w/o Frequency Module</td><td>0.381</td><td>0.394</td><td>0.279</td><td>0.324</td><td>0.433</td><td>0.433</td><td>0.364</td><td>0.398</td><td>0.250</td><td>0.280</td><td>1.786</td><td>0.884</td><td>0.374</td><td>0.410</td><td>1</td></tr><tr><td>w/o Multihead CMA</td><td>0.374</td><td>0.395</td><td>0.283</td><td>0.325</td><td>0.421</td><td>0.432</td><td>0.355</td><td>0.392</td><td>0.249</td><td>0.277</td><td>1.813</td><td>0.865</td><td>0.378</td><td>0.413</td><td>0</td></tr><tr><td>w/o Residual Connection</td><td>0.404</td><td>0.413</td><td>0.288</td><td>0.329</td><td>0.433</td><td>0.443</td><td>0.384</td><td>0.411</td><td>0.249</td><td>0.280</td><td>2.176</td><td>0.969</td><td>0.396</td><td>0.427</td><td>0</td></tr><tr><td>w/o Gating Mechanism</td><td>0.373</td><td>0.396</td><td>0.280</td><td>0.322</td><td>0.425</td><td>0.432</td><td>0.363</td><td>0.398</td><td>0.249</td><td>0.277</td><td>1.724</td><td>0.837</td><td>0.373</td><td>0.412</td><td>1</td></tr></table>

Table 4: Results of design choices related to the model. All results are averaged from four different forecasting horizons:  $\mathbf{H} \in \{96, 192, 336, 720\}$  for the input sequence length 96. **Bold:** the best. Full results are in Appendix E.

SOTA scores. Similar trend is visible in the  $5\%$  few-shot forecasting task, as T3Time improves upon SOTA scores by  $4.13\%$  and  $1.91\%$  w.r.t. MSE and MAE. When compared with models like TimesNet, DLinear, and PatchTST, T3Time consistently outperforms these approaches, surpassing  $30\%$  in MSE reduction. Additionally, relative to recent SOTA methods like TimeCMA, TimeLLM, and GPT4TS, T3Time demonstrates an average MSE improvement exceeding  $10\%$ . A comprehensive overview of our full results for few-shot forecasting is provided in Appendix D.

# Design Variant

Setup. To evaluate the robustness and contribution of various design choices in T3Time, we experiment with several ablations, including removing frequency features, multihead mechanisms, residual connections, and gating. Specifically, we evaluate the following design variants: (1) Without Frequency, where frequency-domain features are excluded; (2) Without Multi-Head, where the adaptive multi-head cross-modal alignment is disabled; (3) Without Residual, which eliminates the channel-wise residual fusion mechanism; and (4) Without Gating, where the horizon-aware gating mechanism is omitted. We maintain the same setup as the long-term forecasting.

Results. Table 4 presents the performance comparison across different design variants. The full model achieves average SOTA performance in 14 out of 14 cases for both MSE and MAE. Notably, removing frequency-domain features results in a  $3.22\%$  MSE and  $1.85\%$  MAE drop compared to the full model. This emphasizes the crucial role of frequency-domain information in capturing both temporal and spectral patterns. However, the residual connection's contribution is the most significant as excluding it leads to the largest performance drop, with an average MSE increase of  $8.36\%$  and MAE of  $5.25\%$ . Excluding the gating mechanism and multihead CMA results in a more modest decline, with an average MSE decrease of approximately  $2\%$ . Full results for the design variants are provided in Appendix E.

# t-SNE Visualization

To better understand the effectiveness of the learned embeddings, we employ T-SNE visualization. Figure 3 presents the T-SNE visualization of the embeddings across four key modalities: time series embeddings, frequency embeddings, prompt embeddings, and forecasted embeddings. Panel 3(a) illustrates the time series embeddings, which demonstrate

![](images/db4da5c4f634daed1b81bbc2a201f1949cca1d2a8493ccaeed97d402ee1ba626.jpg)



Figure 3: The combined T-SNE visualization of time series, frequency, prompt, and forecasted embeddings across six datasets. A detailed breakdown of each modality and dataset is provided in Appendix F.


clear clustering of the datasets. Separation between the datasets indicates that the model is effectively capturing their temporal characteristics. Panel 3(b) visualizes the frequency embeddings, where we observe a distinct clustering pattern. This suggests that the model captures meaningful frequency-domain information, which is crucial for distinguishing underlying periodicities in the time series. Panel 3(c) shows the prompt embeddings, which exhibit a relatively more dispersed structure compared to the time series and frequency embeddings, reflecting the diverse nature of the prompts. Finally, panel 3(d) displays the forecasted embeddings, where the clustering is similar to that of the time series embeddings, suggesting that the model is learning representations that align with both historical and forecasted data. These visualizations illustrate T3Time's ability to integrate temporal, spectral, and prompt information in learning robust multimodal representation.

# Conclusion

In this work, we present T3Time, a novel framework for multivariate time series forecasting which integrates three distinct modalities—time series, frequency, and prompt embeddings. Along with the tri-modal encoding framework, T3Time adapts the fusion of temporal-spectral features using a dynamic horizon-aware gating mechanism. The introduction of adaptive multi-head cross-modal alignment en

ables more flexible and contextual interaction between the time-spectral and prompt representations, while the channelwise residual connection ensures efficient fusion of these representations later in the framework. T3Time consistently outperforms SOTA models, achieving notable reductions in both MSE and MAE across numerous baselines. Additionally, T3Time demonstrates strong generalization in few-shot learning scenarios, with significant performance gains. Future works can explore large-scale pertaining and better representation methods to enrich modalities for stronger time series forecasting.

# References



Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.





Asuncion, A.; Newman, D.; et al. 2007. UCI machine learning repository.





Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877-1901.





Cao, D.; Wang, Y.; Duan, J.; Zhang, C.; Zhu, X.; Huang, C.; Tong, Y.; Xu, B.; Bai, J.; Tong, J.; et al. 2020. Spectral temporal graph neural network for multivariate time-series forecasting. Advances in neural information processing systems, 33: 17766-17778.





Challu, C.; Olivares, K. G.; Oreshkin, B. N.; Ramirez, F. G.; Canseco, M. M.; and Dubrawski, A. 2023. Nhits: Neural hierarchical interpolation for time series forecasting. In Proceedings of the AAAI conference on artificial intelligence, volume 37, 6989-6997.





Chang, C.; Chan, C.-T.; Wang, W.-Y.; Peng, W.-C.; and Chen, T.-F. 2024. Timedrl: Disentangled representation learning for multivariate time-series. In 2024 IEEE 40th International Conference on Data Engineering (ICDE), 625-638. IEEE.





Chen, M.; Xu, Z.; Zeng, A.; and Xu, Q. 2023. FrAug: Frequency domain augmentation for time series forecasting. arXiv preprint arXiv:2302.09292.





Chen, P.-Y. 2024. Model reprogramming: Resource-efficient cross-domain machine learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 22584-22591.





Deldari, S.; Xue, H.; Saeed, A.; He, J.; Smith, D. V.; and Salim, F. D. 2022. Beyond just vision: A review on self-supervised representation learning on multimodal and temporal data. arXiv preprint arXiv:2206.02353.





Franceschi, J.-Y.; Dieuleveut, A.; and Jaggi, M. 2019. Unsupervised scalable representation learning for multivariate time series. Advances in neural information processing systems, 32.





Huang, Q.; Zhou, Z.; Yang, K.; Lin, G.; Yi, Z.; and Wang, Y. 2024. Leret: Language-empowered retentive network





for time series forecasting. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24.





Jia, F.; Wang, K.; Zheng, Y.; Cao, D.; and Liu, Y. 2024. Gpt4mts: Prompt-based large language model for multimodal time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 23343-23351.





Jin, G.; Liu, C.; Xi, Z.; Sha, H.; Liu, Y.; and Huang, J. 2022. Adaptive dual-view wavenet for urban spatial-temporal event prediction. Information Sciences, 588: 315-330.





Jin, M.; Wang, S.; Ma, L.; Chu, Z.; Zhang, J. Y.; Shi, X.; Chen, P.-Y.; Liang, Y.; Li, Y.-F.; Pan, S.; et al. 2023. Time-lmm: Time series forecasting by reprogramming large language models. arXiv preprint arXiv:2310.01728.





Lai, G.; Chang, W.-C.; Yang, Y.; and Liu, H. 2018. Modeling long-and short-term temporal patterns with deep neural networks. In The 41st international ACM SIGIR conference on research & development in information retrieval, 95-104.





Liu, C.; Xiao, Z.; Wang, D.; Wang, L.; Jiang, H.; Chen, H.; and Yu, J. 2022a. Exploiting Spatiotemporal Correlations of Arrive-Stay-Leave Behaviors for Private Car Flow Prediction. IEEE Transactions on Network Science and Engineering, 9(2): 834-847.





Liu, C.; Xu, Q.; Miao, H.; Yang, S.; Zhang, L.; Long, C.; Li, Z.; and Zhao, R. 2025. Timecma: Towards lvm-empowered multivariate time series forecasting via cross-modality alignment. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 18780–18788.





Liu, C.; Yang, S.; Xu, Q.; Li, Z.; Long, C.; Li, Z.; and Zhao, R. 2024a. Spatial-temporal large language model for traffic prediction. In 2024 25th IEEE International Conference on Mobile Data Management (MDM), 31-40. IEEE.





Liu, H.; Ma, Z.; Yang, L.; Zhou, T.; Xia, R.; Wang, Y.; Wen, Q.; and Sun, L. 2023a. Sadi: A self-adaptive decomposed interpretable framework for electric load forecasting under extreme events. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1-5. IEEE.





Liu, S.; Yu, H.; Liao, C.; Li, J.; Lin, W.; Liu, A. X.; and Dustdar, S. 2022b. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In # PLACEHOLDER_PARENT_METADATA_VALUE#.





Liu, X.; Hu, J.; Li, Y.; Diao, S.; Liang, Y.; Hooi, B.; and Zimmermann, R. 2024b. Unitime: A language-empowered unified model for cross-domain time series forecasting. In Proceedings of the ACM Web Conference 2024, 4095–4106.





Liu, Y.; Hu, T.; Zhang, H.; Wu, H.; Wang, S.; Ma, L.; and Long, M. 2023b. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. arXiv preprint arXiv:2310.06625.





Liu, Z.; Miao, H.; Zhao, Y.; Liu, C.; Zheng, K.; and Li, H. 2024c. LightTR: A lightweight framework for federated trajectory recovery. In 2024 IEEE 40th International Conference on Data Engineering (ICDE), 4422-4434. IEEE.





Lu, K.; Grover, A.; Abbeel, P.; and Mordatch, I. 2021. Pretrained Transformers as Universal Computation Engines. arXiv preprint arXiv:2103.05247.





Miao, H.; Zhao, Y.; Guo, C.; Yang, B.; Zheng, K.; Huang, F.; Xie, J.; and Jensen, C. S. 2024. A unified replay-based continuous learning framework for spatio-temporal prediction on streaming data. In 2024 IEEE 40th International Conference on Data Engineering (ICDE), 1050-1062. IEEE.





Nie, Y.; H. Nguyen, N.; Sinthong, P.; and Kalagnanam, J. 2023. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. In International Conference on Learning Representations.





Pan, Z.; Jiang, Y.; Garg, S.; Schneider, A.; Nevmyvaka, Y.; and Song, D. 2024.  $s^2$  IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting. In *Forty-first International Conference on Machine Learning*.





Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.; et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.





Radford, A.; Narasimhan, K.; Salimans, T.; Sutskever, I.; et al. 2018. Improving language understanding by generative pre-training.





Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; Sutskever, I.; et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8): 9.





Schneider, S. H.; and Dickinson, R. E. 1974. Climate modeling. *Reviews of Geophysics*, 12(3): 447-493.





Sun, F.-K.; and Boning, D. S. 2022. Fredo: frequency domain-based long-term time series forecasting. arXiv preprint arXiv:2205.12301.





Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L.; and Polosukhin, I. 2017. Attention is all you need. Advances in neural information processing systems, 30.





Wen, Q.; Zhou, T.; Zhang, C.; Chen, W.; Ma, Z.; Yan, J.; and Sun, L. 2022. Transformers in time series: A survey. arXiv preprint arXiv:2202.07125.





Woo, G.; Liu, C.; Sahoo, D.; Kumar, A.; and Hoi, S. 2022. Cost: Contrastive learning of disentangled seasonal-trend representations for time series forecasting. arXiv preprint arXiv:2202.01575.





Wu, H.; Hu, T.; Liu, Y.; Zhou, H.; Wang, J.; and Long, M. 2022. Timesnet: Temporal 2d-variation modeling for general time series analysis. arXiv preprint arXiv:2210.02186.





Wu, H.; Xu, J.; Wang, J.; and Long, M. 2021. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in neural information processing systems, 34: 22419-22430.





Xu, J.; Wu, H.; Wang, J.; and Long, M. 2021. Anomaly transformer: Time series anomaly detection with association discrepancy. arXiv preprint arXiv:2110.02642.





Xue, H.; and Salim, F. D. 2023. Promptcast: A new prompt-based learning paradigm for time series forecasting. IEEE





Transactions on Knowledge and Data Engineering, 36(11): 6851-6864.





Xue, H.; Voutharoja, B. P.; and Salim, F. D. 2022. Leveraging language foundation models for human mobility forecasting. In Proceedings of the 30th international conference on advances in geographic information systems, 1-9.





Yi, K.; Zhang, Q.; Fan, W.; Wang, S.; Wang, P.; He, H.; An, N.; Lian, D.; Cao, L.; and Niu, Z. 2023. Frequency-domain MLPs are more effective learners in time series forecasting. Advances in Neural Information Processing Systems, 36: 76656-76679.





Yin, S.; Fu, C.; Zhao, S.; Li, K.; Sun, X.; Xu, T.; and Chen, E. 2024. A survey on multimodal large language models. National Science Review, 11(12): nwae403.





Yue, W.; Liu, Y.; Ying, X.; Xing, B.; Guo, R.; and Shi, J. 2025. Freeformer: Frequency enhanced transformer for multivariate time series forecasting. arXiv preprint arXiv:2501.13989.





Zeng, A.; Chen, M.; Zhang, L.; and Xu, Q. 2023. Are transformers effective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, 11121-11128.





Zhang, K.; Wen, Q.; Zhang, C.; Cai, R.; Jin, M.; Liu, Y.; Zhang, J. Y.; Liang, Y.; Pang, G.; Song, D.; et al. 2024. Self-supervised learning for time series analysis: Taxonomy, progress, and prospects. IEEE transactions on pattern analysis and machine intelligence, 46(10): 6775-6794.





Zhang, X.; Zhao, Z.; Tsiligkaridis, T.; and Zitnik, M. 2022. Self-supervised contrastive pre-training for time series via time-frequency consistency. Advances in neural information processing systems, 35: 3988-4003.





Zhou, H.; Zhang, S.; Peng, J.; Zhang, S.; Li, J.; Xiong, H.; and Zhang, W. 2021. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence, volume 35, 11106-11115.





Zhou, T.; Ma, Z.; Wen, Q.; Wang, X.; Sun, L.; and Jin, R. 2022. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International conference on machine learning, 27268-27286. PMLR.





Zhou, T.; Niu, P.; Sun, L.; Jin, R.; et al. 2023. One fits all: Power general time series analysis by pretrained lm. Advances in neural information processing systems, 36: 43322-43355.



# Appendix A: Experimental Setup

# A.1: Hardware Details

All experiments, including the ablation studies, were conducted on a workstation equipped with an Intel Core i9-285K processor (24 cores, 3.7 GHz base clock), 96 GB of RAM, and an NVIDIA GeForce RTX 4090 GPU with 24 GB of dedicated VRAM. The system ran on Microsoft Windows 11 Pro (build 26100) with DirectX 12 and CUDA 12 support. The motherboard used was MSI MPG Z690 Carbon WiFi, and the GPU driver version was 32.0.15.6614. All code was executed in a Python 3.11, PyTorch 2.1.2, torchvision 0.8.0, and conda environment with PyTorch (Paszke et al. 2019) and Transformers frameworks. All of the experiments were done using three different seed values, and the average score was reported.

# A.2: Dataset Details

Overall dataset statistics are illustrated in Table 5. We evaluate the performance of time series forecasting on seven widely used benchmark datasets, including ETT (ETTm1, ETTm2, ETTh1, ETTh2) (Zhou et al. 2021), ECL (Asuncion, Newman et al. 2007), Weather (Wu et al. 2021), Exchange (Wu et al. 2022), and ILI (Wu et al. 2022).

ETT. The Electricity Transformer Temperature (ETT) dataset serves as a critical benchmark for evaluating electric power forecasting. It comprises two years of data collected from two separate counties in China. To analyze the impact of temporal granularity, the dataset is divided into four subsets with different sampling frequencies: ETTh1 and ETTh2 are sampled at 1-hour intervals, while ETTm1 and ETTm2 are sampled at 15-minute intervals. Each data point contains six power load-related features along with a target variable, oil temperature.

ECL. The Electricity dataset includes hourly electricity consumption data from 370 clients, providing insights into consumer-level load patterns. Data is collected from 1st January, 2011 with a sampling interval of 15 minutes.

Weather. Weather dataset consists of one year of meteorological measurements recorded every 10 minutes across 21 weather stations of the Max Planck Biogeochemistry Institute in Germany. It includes 21 variables such as air temperature, humidity, and wind speed, etc.

Exchange. Exchange comprises daily exchange rate records from 1990 to 2016 for eight foreign currencies, including those of Australia, the United Kingdom, China, Japan, Canada, Singapore, Switzerland, and New Zealand. The data is sampled at a one-day interval.

ILI. The Influenza-like Illness (ILI) dataset captures the weekly number of reported cases involving severe influenza symptoms with complications.

# A.3: Evaluation Metrics

We evaluate the forecasting performance using two widely adopted metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE). These metrics offer complementary perspectives on prediction accuracy. MSE penalizes larger errors more heavily due to the squaring term, making it particularly sensitive to significant deviations and thus

suitable for capturing overall stability and variance in the predictions. In contrast, MAE measures the average magnitude of errors in a more uniform manner, providing a robust view of typical forecasting accuracy.

Let  $N$  denote the prediction horizon, and  $y_{n}$  and  $\hat{y}_n$  represent the ground truth and predicted values at step  $n$ , respectively, where  $n \in \{1, \dots, N\}$ . The metrics are defined as follows.

$$
\mathrm {M S E} = \frac {1}{N} \sum_ {n = 1} ^ {N} \left(y _ {n} - \hat {y} _ {n}\right) ^ {2} \tag {19}
$$

$$
\mathrm {M A E} = \frac {1}{N} \sum_ {n = 1} ^ {N} \left| y _ {n} - \hat {y} _ {n} \right| \tag {20}
$$

# Appendix B: Implementation Details

# B.1: Prompt Description

To adapt multivariate time series data for language model processing, we design dataset-specific prompt templates that transform structured temporal data into natural language sequences. Each prompt, as illustrated in figure 4, captures a sliding window of observations and encodes four key components: the temporal interval, the sequence of numerical values, the sampling resolution, and a summary statistic representing the trend over the interval. Specifically, [t1] and [t2] indicate the start and end timestamps of the window; [value1, ..., valuen] denotes the ordered sequence of measurements; [f] specifies the data sampling frequency; and [T] encodes a high-level trend metric (e.g., cumulative change) over the window. This textualization enables the integration of time series dynamics into language-based architectures.

# B.2: Hyperparameter Sensitivity

For all experiments, we employed grid search to identify the optimal hyperparameters of our model across different datasets. The complete hyperparameter search space is detailed in Table 6. For each dataset, we selected the set of hyperparameters that yielded the lowest validation loss and reported the corresponding test performance. Optimal hyperparameters are provided in the codebase. The MSE sensitivity of different hyperparameters are also illustrated in Figure 5 for the ETTh2 dataset across varying forecasting horizons,  $H \in \{96,192,336,720\}$ .

# B.3: Model Configurations

The architectural and training configurations adopted for our framework across diverse datasets are summarized in Tab. 7. The input length is uniformly set to 96 time steps across all tasks, providing a consistent temporal context for forecasting. A multi-head attention mechanism with 4 heads is used consistently for all configurations, and dropout rates are tuned individually to mitigate overfitting. Training-related parameters, presented in the five rightmost columns of Tab. 7, include dataset-specific batch sizes and epoch counts tailored to convergence behaviour. The model is trained using the MSE loss across all experiments. The learning rate and weight decay are kept fixed throughout all

<table><tr><td>Dataset</td><td>Dimension</td><td>Series Length</td><td>Dataset Size</td><td>Frequency</td><td>Domain</td></tr><tr><td>ETTm1</td><td>7</td><td>{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>15 mins</td><td>Electricity</td></tr><tr><td>ETTm2</td><td>7</td><td>{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>15 mins</td><td>Electricity</td></tr><tr><td>ETTh1</td><td>7</td><td>{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>15 mins</td><td>Electricity</td></tr><tr><td>ETTh2</td><td>7</td><td>{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>15 mins</td><td>Electricity</td></tr><tr><td>Electricity (ECL)</td><td>321</td><td>{96, 192, 336, 720}</td><td>(18317, 2633, 5261)</td><td>Hourly</td><td>Electricity</td></tr><tr><td>Weather</td><td>21</td><td>{96, 192, 336, 720}</td><td>(36792, 5271, 10540)</td><td>10 mins</td><td>Weather</td></tr><tr><td>Exchange</td><td>8</td><td>{96, 192, 336, 720}</td><td>(5120, 665, 1422)</td><td>Daily</td><td>Exchange rate</td></tr><tr><td>ILI</td><td>7</td><td>{24, 36, 48, 60}</td><td>(617, 74, 170)</td><td>Weekly</td><td>Illness</td></tr></table>


Table 5: Summary of datasets used in our experiments. Each dataset varies in domain, dimensionality, sampling frequency, and series length. Forecasting is performed over multiple horizons, with input sequence length fixed to 96 (except ILI). Dataset size is organized as a train set, a validation set, and a test set. Dimensions describe the number of time series channels.


<table><tr><td>ETTm1: From [t1] to [t2], the values were [value1, ..., valuen] every [15 minutes]. The total trend value was [T].</td></tr><tr><td>ETTm2: From [t1] to [t2], the values were [value1, ..., valuen] every [15 minutes]. The total trend value was [T].</td></tr><tr><td>ETTh1: From [t1] to [t2], the values were [value1, ..., valuen] every [hour]. The total trend value was [T].</td></tr><tr><td>ETTh2: From [t1] to [t2], the values were [value1, ..., valuen] every [hour]. The total trend value was [T].</td></tr><tr><td>ECL: From [t1] to [t2], the values were [value1, ..., valuen] every [hour]. The total trend value was [T].</td></tr><tr><td>Weather: From [t1] to [t2], the values were [value1, ..., valuen] every [10 minutes]. The total trend value was [T].</td></tr><tr><td>Exchange: From [t1] to [t2], the values were [value1, ..., valuen] every [day]. The total trend value was [T].</td></tr><tr><td>ILI: From [t1] to [t2], the values were [value1, ..., valuen] every [week]. The total trend value was [T].</td></tr></table>


Figure 4: Dataset-specific prompt templates used to convert multivariate time series windows into natural language descriptions. Each prompt encapsulates the observation window's time span, value sequence, sampling frequency, and trend summary, thereby aligning structured temporal data with language model input requirements.


<table><tr><td>Hyperparameter</td><td>Search Space</td></tr><tr><td>Encoder Layer (e_layer)</td><td>{1,2,3,4,5,6}</td></tr><tr><td>Decoder Layer (d_layer)</td><td>{1,2,3,4,5,6}</td></tr><tr><td>Channel Dimension (channel)</td><td>{16,32,64,128,256}</td></tr><tr><td>Batch Size (batch_size)</td><td>{8,16,32,64,128,256}</td></tr><tr><td>Dropout Rate (dropout)</td><td>{0.1,0.2,0.3,0.4,0.5,0.6}</td></tr></table>

Table 6: Hyperparameter search space used for model optimization.

the experiments. Individual horizon specific hyperparameters are detailed in the codebase.

![](images/ebbebf64834e7f0d975bda56798f81eae8d8dd4daf1fc514c2cc0c45faf472be.jpg)


![](images/c4c0d228751275b535ebd4b635dfee3c46a253b4f4607fd343a7730000283b9b.jpg)


![](images/8bc52835565e06bf24e93ae517d8010fe50fee0d91fa23a015debc34a43df288.jpg)


![](images/d95ae0a9dea8d40c84815d9b3ceb7dbcbe4cfbb157bafe7b423e91c38559a8ca.jpg)



Figure 5: Sensitivity analysis of key hyperparameters on the ETTh2 dataset. We evaluate the impact of varying channel dimension, dropout rate, batch size, and encoder-decoder depth on forecasting across horizon lengths {96, 192, 336, 720}.


<table><tr><td rowspan="2">Configuration 
Dataset</td><td colspan="6">Model Hyperparameter</td><td colspan="5">Training Process</td></tr><tr><td>Encoder Layer</td><td>Decoder Layer</td><td>Input</td><td>Channel Dim.</td><td>Heads</td><td>Dropout</td><td>Learning Rate</td><td>Weight Decay</td><td>Loss</td><td>Batch Size</td><td>Epochs</td></tr><tr><td>ETTm1</td><td>1</td><td>2</td><td>96</td><td>128</td><td>4</td><td>0.5</td><td>1e-4</td><td>1e-3</td><td>MSE</td><td>64</td><td>150</td></tr><tr><td>ETTm2</td><td>1</td><td>1</td><td>96</td><td>64</td><td>4</td><td>0.6</td><td>1e-4</td><td>1e-3</td><td>MSE</td><td>16</td><td>150</td></tr><tr><td>ETTh1</td><td>1</td><td>1</td><td>96</td><td>256</td><td>4</td><td>0.4</td><td>1e-4</td><td>1e-3</td><td>MSE</td><td>256</td><td>150</td></tr><tr><td>ETTh2</td><td>1</td><td>1</td><td>96</td><td>64</td><td>4</td><td>0.25</td><td>1e-4</td><td>1e-3</td><td>MSE</td><td>256</td><td>150</td></tr><tr><td>ECL</td><td>1</td><td>2</td><td>96</td><td>128</td><td>4</td><td>0.3</td><td>1e-4</td><td>1e-3</td><td>MSE</td><td>128</td><td>50</td></tr><tr><td>Weather</td><td>6</td><td>2</td><td>96</td><td>64</td><td>4</td><td>0.1</td><td>1e-4</td><td>1e-3</td><td>MSE</td><td>32</td><td>150</td></tr><tr><td>ILI</td><td>1</td><td>1</td><td>96</td><td>32</td><td>4</td><td>0.1</td><td>1e-4</td><td>1e-3</td><td>MSE</td><td>16</td><td>100</td></tr></table>

Table 7: An overview of the experimental configurations adopted for our model.

# B.4: Multi-Head CMA

To investigate the impact of the number of heads in the Multi-Head Cross-Modal Attention (CMA) module, we evaluated five different configurations:  $\{1,2,4,8,16\}$ , keep

ing all other architectural components constant. As reported in Table 8, the 4-head configuration consistently delivers the best performance, achieving the lowest Mean Squared Error (MSE) and Mean Absolute Error (MAE) in 21 and 20 out of

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Horizon</td><td colspan="2">Head = 4</td><td colspan="2">Head = 1</td><td colspan="2">Head = 2</td><td colspan="2">Head = 8</td><td colspan="2">Head = 16</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.308</td><td>0.354</td><td>0.311</td><td>0.357</td><td>0.308</td><td>0.353</td><td>0.311</td><td>0.357</td><td>0.313</td><td>0.359</td></tr><tr><td>192</td><td>0.357</td><td>0.381</td><td>0.356</td><td>0.382</td><td>0.356</td><td>0.382</td><td>0.354</td><td>0.381</td><td>0.356</td><td>0.383</td></tr><tr><td>336</td><td>0.382</td><td>0.400</td><td>0.385</td><td>0.402</td><td>0.384</td><td>0.402</td><td>0.383</td><td>0.402</td><td>0.381</td><td>0.402</td></tr><tr><td>720</td><td>0.442</td><td>0.437</td><td>0.444</td><td>0.439</td><td>0.441</td><td>0.436</td><td>0.440</td><td>0.437</td><td>0.449</td><td>0.442</td></tr><tr><td>Avg</td><td>0.372</td><td>0.393</td><td>0.374</td><td>0.395</td><td>0.372</td><td>0.393</td><td>0.372</td><td>0.394</td><td>0.374</td><td>0.396</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.172</td><td>0.254</td><td>0.175</td><td>0.258</td><td>0.175</td><td>0.256</td><td>0.175</td><td>0.256</td><td>0.174</td><td>0.257</td></tr><tr><td>192</td><td>0.237</td><td>0.300</td><td>0.243</td><td>0.301</td><td>0.244</td><td>0.302</td><td>0.244</td><td>0.303</td><td>0.243</td><td>0.302</td></tr><tr><td>336</td><td>0.306</td><td>0.337</td><td>0.308</td><td>0.340</td><td>0.308</td><td>0.343</td><td>0.305</td><td>0.342</td><td>0.305</td><td>0.338</td></tr><tr><td>720</td><td>0.400</td><td>0.398</td><td>0.406</td><td>0.402</td><td>0.402</td><td>0.399</td><td>0.405</td><td>0.401</td><td>0.407</td><td>0.401</td></tr><tr><td>Avg</td><td>0.279</td><td>0.322</td><td>0.283</td><td>0.325</td><td>0.282</td><td>0.325</td><td>0.282</td><td>0.326</td><td>0.282</td><td>0.324</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.371</td><td>0.397</td><td>0.371</td><td>0.399</td><td>0.374</td><td>0.399</td><td>0.371</td><td>0.397</td><td>0.373</td><td>0.398</td></tr><tr><td>192</td><td>0.411</td><td>0.421</td><td>0.411</td><td>0.421</td><td>0.411</td><td>0.421</td><td>0.412</td><td>0.423</td><td>0.412</td><td>0.424</td></tr><tr><td>336</td><td>0.448</td><td>0.441</td><td>0.454</td><td>0.444</td><td>0.451</td><td>0.443</td><td>0.453</td><td>0.445</td><td>0.447</td><td>0.444</td></tr><tr><td>720</td><td>0.441</td><td>0.460</td><td>0.447</td><td>0.463</td><td>0.448</td><td>0.464</td><td>0.444</td><td>0.462</td><td>0.445</td><td>0.464</td></tr><tr><td>Avg</td><td>0.418</td><td>0.430</td><td>0.421</td><td>0.432</td><td>0.421</td><td>0.432</td><td>0.419</td><td>0.432</td><td>0.419</td><td>0.433</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.278</td><td>0.338</td><td>0.287</td><td>0.342</td><td>0.287</td><td>0.342</td><td>0.288</td><td>0.344</td><td>0.285</td><td>0.341</td></tr><tr><td>192</td><td>0.351</td><td>0.389</td><td>0.367</td><td>0.397</td><td>0.367</td><td>0.398</td><td>0.343</td><td>0.385</td><td>0.370</td><td>0.402</td></tr><tr><td>336</td><td>0.358</td><td>0.398</td><td>0.362</td><td>0.400</td><td>0.361</td><td>0.400</td><td>0.355</td><td>0.396</td><td>0.359</td><td>0.402</td></tr><tr><td>720</td><td>0.404</td><td>0.433</td><td>0.402</td><td>0.431</td><td>0.404</td><td>0.436</td><td>0.402</td><td>0.430</td><td>0.400</td><td>0.433</td></tr><tr><td>Avg</td><td>0.348</td><td>0.390</td><td>0.355</td><td>0.392</td><td>0.355</td><td>0.394</td><td>0.347</td><td>0.389</td><td>0.354</td><td>0.394</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.162</td><td>0.210</td><td>0.167</td><td>0.212</td><td>0.166</td><td>0.212</td><td>0.161</td><td>0.208</td><td>0.168</td><td>0.215</td></tr><tr><td>192</td><td>0.211</td><td>0.253</td><td>0.212</td><td>0.253</td><td>0.213</td><td>0.254</td><td>0.215</td><td>0.258</td><td>0.215</td><td>0.260</td></tr><tr><td>336</td><td>0.267</td><td>0.293</td><td>0.276</td><td>0.298</td><td>0.269</td><td>0.295</td><td>0.269</td><td>0.297</td><td>0.272</td><td>0.297</td></tr><tr><td>720</td><td>0.335</td><td>0.346</td><td>0.344</td><td>0.345</td><td>0.339</td><td>0.343</td><td>0.342</td><td>0.349</td><td>0.344</td><td>0.344</td></tr><tr><td>Avg</td><td>0.244</td><td>0.275</td><td>0.249</td><td>0.277</td><td>0.247</td><td>0.276</td><td>0.247</td><td>0.278</td><td>0.250</td><td>0.279</td></tr><tr><td rowspan="5">Exchange</td><td>96</td><td>0.085</td><td>0.205</td><td>0.085</td><td>0.205</td><td>0.085</td><td>0.205</td><td>0.084</td><td>0.204</td><td>0.087</td><td>0.207</td></tr><tr><td>192</td><td>0.172</td><td>0.296</td><td>0.174</td><td>0.298</td><td>0.172</td><td>0.296</td><td>0.170</td><td>0.295</td><td>0.175</td><td>0.298</td></tr><tr><td>336</td><td>0.318</td><td>0.408</td><td>0.321</td><td>0.412</td><td>0.332</td><td>0.420</td><td>0.324</td><td>0.414</td><td>0.323</td><td>0.412</td></tr><tr><td>720</td><td>0.836</td><td>0.696</td><td>0.933</td><td>0.737</td><td>0.842</td><td>0.698</td><td>0.876</td><td>0.712</td><td>0.955</td><td>0.744</td></tr><tr><td>Avg</td><td>0.353</td><td>0.401</td><td>0.378</td><td>0.413</td><td>0.358</td><td>0.405</td><td>0.364</td><td>0.406</td><td>0.385</td><td>0.415</td></tr><tr><td rowspan="5">ILI</td><td>24</td><td>1.583</td><td>0.802</td><td>1.676</td><td>0.807</td><td>1.552</td><td>0.801</td><td>1.567</td><td>0.784</td><td>1.542</td><td>0.791</td></tr><tr><td>36</td><td>1.601</td><td>0.820</td><td>1.510</td><td>0.806</td><td>1.569</td><td>0.802</td><td>1.594</td><td>0.812</td><td>1.734</td><td>0.852</td></tr><tr><td>48</td><td>1.718</td><td>0.815</td><td>2.114</td><td>0.935</td><td>1.774</td><td>0.849</td><td>1.826</td><td>0.873</td><td>1.827</td><td>0.874</td></tr><tr><td>60</td><td>1.920</td><td>0.901</td><td>1.955</td><td>0.912</td><td>2.003</td><td>0.922</td><td>1.973</td><td>0.912</td><td>1.984</td><td>0.911</td></tr><tr><td>Avg</td><td>1.705</td><td>0.835</td><td>1.814</td><td>0.865</td><td>1.725</td><td>0.844</td><td>1.739</td><td>0.845</td><td>1.772</td><td>0.857</td></tr><tr><td colspan="2">1st Count</td><td>20</td><td>24</td><td>3</td><td>2</td><td>4</td><td>7</td><td>11</td><td>10</td><td>5</td><td>1</td></tr></table>

Table 8: Ablation study on the number of heads in the Cross-Modal Attention (CMA) module across benchmark datasets and prediction lengths. Metrics reported are MSE and MAE (lower is better). Input sequence length is set to 96. Bold: the best.

# 35 benchmark cases, respectively.

The 8-head variant shows relatively competitive performance, securing the best score in 10 out of 35 cases for both MSE and MAE. However, increasing the head count beyond this (e.g., to 16) does not lead to further improvements and often degrades performance—likely due to over-fragmentation of attention or redundancy in representation subspaces. On the other hand, the 1-head and 2-head settings

underperform across most datasets and horizons, indicating insufficient capacity for capturing complex cross-modal alignments. Overall, the 4-head CMA setting achieves the lowest average MSE and lowest average MAE across all tasks, validating its effectiveness in balancing representation richness and attention diversity. Consequently, we adopt 4 heads as the default configuration for the CMA module in T3Time.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Horizon</td><td colspan="2">Ours</td><td colspan="2">TimeCMA</td><td colspan="2">TimeLLM</td><td colspan="2">UniTime</td><td colspan="2">TimesNet</td><td colspan="2">DLinear</td><td colspan="2">iTransformer</td><td colspan="2">PatchTST</td><td colspan="2">OFA</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.308</td><td>0.354</td><td>0.312</td><td>0.351</td><td>0.359</td><td>0.381</td><td>0.322</td><td>0.363</td><td>0.338</td><td>0.375</td><td>0.345</td><td>0.372</td><td>0.334</td><td>0.368</td><td>0.344</td><td>0.373</td><td>0.335</td><td>0.369</td></tr><tr><td>192</td><td>0.357</td><td>0.381</td><td>0.361</td><td>0.378</td><td>0.383</td><td>0.393</td><td>0.366</td><td>0.387</td><td>0.374</td><td>0.387</td><td>0.380</td><td>0.389</td><td>0.377</td><td>0.391</td><td>0.367</td><td>0.386</td><td>0.374</td><td>0.385</td></tr><tr><td>336</td><td>0.382</td><td>0.400</td><td>0.392</td><td>0.401</td><td>0.416</td><td>0.414</td><td>0.398</td><td>0.407</td><td>0.410</td><td>0.411</td><td>0.413</td><td>0.413</td><td>0.426</td><td>0.420</td><td>0.392</td><td>0.407</td><td>0.407</td><td>0.406</td></tr><tr><td>720</td><td>0.442</td><td>0.437</td><td>0.453</td><td>0.438</td><td>0.483</td><td>0.449</td><td>0.454</td><td>0.440</td><td>0.478</td><td>0.450</td><td>0.474</td><td>0.453</td><td>0.491</td><td>0.459</td><td>0.464</td><td>0.442</td><td>0.469</td><td>0.442</td></tr><tr><td>Avg</td><td>0.372</td><td>0.393</td><td>0.380</td><td>0.392</td><td>0.410</td><td>0.409</td><td>0.385</td><td>0.399</td><td>0.400</td><td>0.406</td><td>0.403</td><td>0.407</td><td>0.407</td><td>0.410</td><td>0.392</td><td>0.402</td><td>0.396</td><td>0.401</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.172</td><td>0.254</td><td>0.173</td><td>0.258</td><td>0.193</td><td>0.280</td><td>0.183</td><td>0.266</td><td>0.187</td><td>0.267</td><td>0.193</td><td>0.292</td><td>0.180</td><td>0.264</td><td>0.177</td><td>0.260</td><td>0.190</td><td>0.275</td></tr><tr><td>192</td><td>0.237</td><td>0.300</td><td>0.238</td><td>0.301</td><td>0.257</td><td>0.318</td><td>0.251</td><td>0.310</td><td>0.249</td><td>0.309</td><td>0.284</td><td>0.362</td><td>0.250</td><td>0.309</td><td>0.246</td><td>0.305</td><td>0.253</td><td>0.313</td></tr><tr><td>336</td><td>0.306</td><td>0.337</td><td>0.297</td><td>0.338</td><td>0.317</td><td>0.353</td><td>0.183</td><td>0.266</td><td>0.319</td><td>0.351</td><td>0.369</td><td>0.427</td><td>0.311</td><td>0.348</td><td>0.305</td><td>0.343</td><td>0.321</td><td>0.360</td></tr><tr><td>720</td><td>0.400</td><td>0.398</td><td>0.393</td><td>0.394</td><td>0.419</td><td>0.411</td><td>0.420</td><td>0.410</td><td>0.408</td><td>0.403</td><td>0.554</td><td>0.522</td><td>0.412</td><td>0.407</td><td>0.410</td><td>0.405</td><td>0.411</td><td>0.406</td></tr><tr><td>Avg</td><td>0.279</td><td>0.322</td><td>0.275</td><td>0.323</td><td>0.296</td><td>0.340</td><td>0.293</td><td>0.334</td><td>0.291</td><td>0.333</td><td>0.350</td><td>0.401</td><td>0.288</td><td>0.332</td><td>0.285</td><td>0.328</td><td>0.294</td><td>0.339</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.371</td><td>0.397</td><td>0.373</td><td>0.391</td><td>0.398</td><td>0.410</td><td>0.397</td><td>0.418</td><td>0.384</td><td>0.402</td><td>0.386</td><td>0.400</td><td>0.386</td><td>0.405</td><td>0.404</td><td>0.413</td><td>0.398</td><td>0.424</td></tr><tr><td>192</td><td>0.411</td><td>0.421</td><td>0.427</td><td>0.421</td><td>0.451</td><td>0.440</td><td>0.434</td><td>0.439</td><td>0.434</td><td>0.429</td><td>0.437</td><td>0.432</td><td>0.441</td><td>0.436</td><td>0.454</td><td>0.430</td><td>0.449</td><td>0.427</td></tr><tr><td>336</td><td>0.448</td><td>0.441</td><td>0.458</td><td>0.448</td><td>0.473</td><td>0.451</td><td>0.468</td><td>0.457</td><td>0.491</td><td>0.469</td><td>0.481</td><td>0.459</td><td>0.487</td><td>0.458</td><td>0.497</td><td>0.462</td><td>0.492</td><td>0.466</td></tr><tr><td>720</td><td>0.441</td><td>0.460</td><td>0.449</td><td>0.460</td><td>0.469</td><td>0.470</td><td>0.469</td><td>0.477</td><td>0.521</td><td>0.500</td><td>0.519</td><td>0.516</td><td>0.503</td><td>0.491</td><td>0.496</td><td>0.481</td><td>0.487</td><td>0.483</td></tr><tr><td>Avg</td><td>0.418</td><td>0.430</td><td>0.423</td><td>0.431</td><td>0.448</td><td>0.443</td><td>0.442</td><td>0.448</td><td>0.458</td><td>0.450</td><td>0.456</td><td>0.452</td><td>0.454</td><td>0.447</td><td>0.463</td><td>0.449</td><td>0.457</td><td>0.450</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.278</td><td>0.338</td><td>0.286</td><td>0.336</td><td>0.295</td><td>0.345</td><td>0.296</td><td>0.345</td><td>0.340</td><td>0.374</td><td>0.333</td><td>0.387</td><td>0.297</td><td>0.349</td><td>0.312</td><td>0.358</td><td>0.312</td><td>0.360</td></tr><tr><td>192</td><td>0.351</td><td>0.389</td><td>0.363</td><td>0.387</td><td>0.386</td><td>0.399</td><td>0.374</td><td>0.394</td><td>0.402</td><td>0.414</td><td>0.477</td><td>0.476</td><td>0.380</td><td>0.400</td><td>0.397</td><td>0.408</td><td>0.387</td><td>0.405</td></tr><tr><td>336</td><td>0.358</td><td>0.398</td><td>0.406</td><td>0.421</td><td>0.419</td><td>0.429</td><td>0.415</td><td>0.427</td><td>0.452</td><td>0.452</td><td>0.594</td><td>0.541</td><td>0.428</td><td>0.432</td><td>0.435</td><td>0.440</td><td>0.424</td><td>0.437</td></tr><tr><td>720</td><td>0.404</td><td>0.433</td><td>0.417</td><td>0.438</td><td>0.425</td><td>0.442</td><td>0.425</td><td>0.444</td><td>0.462</td><td>0.468</td><td>0.831</td><td>0.657</td><td>0.427</td><td>0.445</td><td>0.436</td><td>0.449</td><td>0.433</td><td>0.453</td></tr><tr><td>Avg</td><td>0.348</td><td>0.390</td><td>0.372</td><td>0.397</td><td>0.381</td><td>0.404</td><td>0.378</td><td>0.403</td><td>0.414</td><td>0.427</td><td>0.559</td><td>0.515</td><td>0.383</td><td>0.407</td><td>0.395</td><td>0.414</td><td>0.389</td><td>0.414</td></tr><tr><td rowspan="5">ECL</td><td>96</td><td>0.138</td><td>0.233</td><td>0.143</td><td>0.238</td><td>0.172</td><td>0.265</td><td>0.196</td><td>0.287</td><td>0.168</td><td>0.272</td><td>0.197</td><td>0.282</td><td>0.148</td><td>0.240</td><td>0.186</td><td>0.269</td><td>0.197</td><td>0.290</td></tr><tr><td>192</td><td>0.155</td><td>0.250</td><td>0.161</td><td>0.259</td><td>0.182</td><td>0.279</td><td>0.199</td><td>0.291</td><td>0.184</td><td>0.289</td><td>0.196</td><td>0.285</td><td>0.162</td><td>0.253</td><td>0.190</td><td>0.273</td><td>0.201</td><td>0.292</td></tr><tr><td>336</td><td>0.168</td><td>0.265</td><td>0.169</td><td>0.261</td><td>0.195</td><td>0.288</td><td>0.214</td><td>0.305</td><td>0.198</td><td>0.300</td><td>0.209</td><td>0.301</td><td>0.178</td><td>0.269</td><td>0.206</td><td>0.290</td><td>0.217</td><td>0.309</td></tr><tr><td>720</td><td>0.218</td><td>0.314</td><td>0.219</td><td>0.315</td><td>0.233</td><td>0.320</td><td>0.254</td><td>0.335</td><td>0.220</td><td>0.320</td><td>0.245</td><td>0.333</td><td>0.225</td><td>0.317</td><td>0.247</td><td>0.322</td><td>0.253</td><td>0.339</td></tr><tr><td>Avg</td><td>0.170</td><td>0.266</td><td>0.174</td><td>0.269</td><td>0.195</td><td>0.288</td><td>0.216</td><td>0.306</td><td>0.192</td><td>0.295</td><td>0.212</td><td>0.300</td><td>0.178</td><td>0.270</td><td>0.207</td><td>0.289</td><td>0.217</td><td>0.308</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.162</td><td>0.210</td><td>0.167</td><td>0.211</td><td>0.198</td><td>0.235</td><td>0.171</td><td>0.214</td><td>0.172</td><td>0.220</td><td>0.196</td><td>0.255</td><td>0.174</td><td>0.214</td><td>0.177</td><td>0.218</td><td>0.203</td><td>0.244</td></tr><tr><td>192</td><td>0.211</td><td>0.253</td><td>0.212</td><td>0.253</td><td>0.240</td><td>0.269</td><td>0.217</td><td>0.254</td><td>0.219</td><td>0.261</td><td>0.237</td><td>0.296</td><td>0.221</td><td>0.254</td><td>0.222</td><td>0.259</td><td>0.247</td><td>0.277</td></tr><tr><td>336</td><td>0.267</td><td>0.293</td><td>0.270</td><td>0.292</td><td>0.295</td><td>0.308</td><td>0.274</td><td>0.293</td><td>0.280</td><td>0.306</td><td>0.283</td><td>0.335</td><td>0.278</td><td>0.296</td><td>0.277</td><td>0.297</td><td>0.297</td><td>0.311</td></tr><tr><td>720</td><td>0.335</td><td>0.346</td><td>0.350</td><td>0.348</td><td>0.368</td><td>0.353</td><td>0.351</td><td>0.343</td><td>0.365</td><td>0.359</td><td>0.345</td><td>0.381</td><td>0.358</td><td>0.349</td><td>0.352</td><td>0.347</td><td>0.368</td><td>0.356</td></tr><tr><td>Avg</td><td>0.244</td><td>0.275</td><td>0.250</td><td>0.276</td><td>0.275</td><td>0.291</td><td>0.253</td><td>0.276</td><td>0.259</td><td>0.287</td><td>0.265</td><td>0.317</td><td>0.258</td><td>0.278</td><td>0.257</td><td>0.280</td><td>0.279</td><td>0.297</td></tr><tr><td rowspan="5">ILI</td><td>24</td><td>1.583</td><td>0.802</td><td>1.996</td><td>0.998</td><td>2.383</td><td>1.004</td><td>2.346</td><td>0.954</td><td>2.317</td><td>0.934</td><td>2.398</td><td>1.040</td><td>2.347</td><td>1.731</td><td>2.335</td><td>0.989</td><td>2.732</td><td>1.100</td></tr><tr><td>36</td><td>1.601</td><td>0.820</td><td>1.906</td><td>0.915</td><td>2.390</td><td>0.993</td><td>1.998</td><td>0.912</td><td>1.972</td><td>0.920</td><td>2.646</td><td>1.088</td><td>2.468</td><td>0.998</td><td>2.561</td><td>1.035</td><td>2.664</td><td>1.063</td></tr><tr><td>48</td><td>1.718</td><td>0.815</td><td>1.867</td><td>0.868</td><td>2.394</td><td>1.003</td><td>1.979</td><td>0.912</td><td>2.238</td><td>0.913</td><td>2.614</td><td>1.086</td><td>2.489</td><td>1.016</td><td>2.465</td><td>1.022</td><td>2.617</td><td>1.041</td></tr><tr><td>60</td><td>1.920</td><td>0.901</td><td>1.920</td><td>0.904</td><td>2.562</td><td>1.049</td><td>2.109</td><td>0.938</td><td>2.027</td><td>0.928</td><td>2.804</td><td>1.146</td><td>2.471</td><td>1.065</td><td>2.189</td><td>0.997</td><td>2.478</td><td>1.035</td></tr><tr><td>Avg</td><td>1.705</td><td>0.835</td><td>1.922</td><td>0.921</td><td>2.432</td><td>1.012</td><td>2.108</td><td>0.929</td><td>2.139</td><td>0.931</td><td>2.616</td><td>1.090</td><td>2.444</td><td>1.203</td><td>2.388</td><td>1.011</td><td>2.623</td><td>1.060</td></tr><tr><td rowspan="5">Exchange</td><td>96</td><td>0.085</td><td>0.205</td><td>0.099</td><td>0.224</td><td>0.087</td><td>0.208</td><td>0.086</td><td>0.209</td><td>0.107</td><td>0.234</td><td>0.088</td><td>0.218</td><td>0.086</td><td>0.206</td><td>0.109</td><td>0.236</td><td>0.148</td><td>0.278</td></tr><tr><td>192</td><td>0.172</td><td>0.296</td><td>0.186</td><td>0.312</td><td>0.173</td><td>0.299</td><td>0.174</td><td>0.299</td><td>0.226</td><td>0.344</td><td>0.176</td><td>0.315</td><td>0.177</td><td>0.299</td><td>0.205</td><td>0.327</td><td>0.271</td><td>0.380</td></tr><tr><td>336</td><td>0.318</td><td>0.408</td><td>0.364</td><td>0.444</td><td>0.375</td><td>0.454</td><td>0.319</td><td>0.408</td><td>0.367</td><td>0.448</td><td>0.313</td><td>0.427</td><td>0.331</td><td>0.417</td><td>0.356</td><td>0.436</td><td>0.460</td><td>0.500</td></tr><tr><td>720</td><td>0.836</td><td>0.696</td><td>0.932</td><td>0.735</td><td>0.853</td><td>0.703</td><td>0.875</td><td>0.701</td><td>0.964</td><td>0.746</td><td>0.839</td><td>0.695</td><td>0.847</td><td>0.691</td><td>0.888</td><td>0.716</td><td>1.195</td><td>0.841</td></tr><tr><td>Avg</td><td>0.353</td><td>0.401</td><td>0.395</td><td>0.429</td><td>0.372</td><td>0.416</td><td>0.364</td><td>0.404</td><td>0.416</td><td>0.443</td><td>0.354</td><td>0.414</td><td>0.360</td><td>0.403</td><td>0.390</td><td>0.429</td><td>0.519</td><td>0.500</td></tr><tr><td colspan="2">Ist Count</td><td>36</td><td>30</td><td>5</td><td>12</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td></td><td></td><td></td></tr></table>


Table 9: Multivariate long-term forecasting results. Input sequence is set to 96 for all but ILI (input sequence 36). The prediction horizon for all benchmark datasets,  $H \in \{96,192,336,720\}$ , and for ILI,  $H \in \{24,36,48,60\}$ . A lower score signals better performance. **Bold:** the best, **underline:** the second best.


# Appendix C: Long Term Forecasting Details

The complete results for the long term forecasting tasks of our model are presented in Table 9. For all benchmark datasets, the input sequence length is fixed at 96, and the prediction sequence lengths are set to  $\{96,192,336,720\}$ . However, for the ILI dataset, the input sequence is set to 36 and the output sequence length is  $\{24,36,48,60\}$ . In the long-term forecasting scenario, our model achieves state-of-the-art (SOTA) performance in 66 out of 80 cases across 8 diverse time series benchmarks.

# Appendix D: Few Shot Forecasting Details

The complete results for the few-shot forecasting tasks of our model are presented in Tables 10 and 11. For all benchmark datasets, the input sequence length is fixed at 512, and the prediction sequence lengths are set to  $\{96,192,336,720\}$ . In the  $10\%$  few-shot scenario, our

model achieves state-of-the-art (SOTA) performance in 43 out of 50 cases across five diverse time series benchmarks. Furthermore, in the  $5\%$  few-shot setting, our model attains SOTA performance in 38 out of 46 cases across the same five benchmarks.

# Appendix E: Design Variant

Table 12 presents a comprehensive ablation study across benchmark datasets and forecasting horizons, evaluating the contributions of key components in our T3Time architecture. We systematically remove four major design modules—Frequency Encoding, Multi-Head CrossModal Alignment, Channel-Wise Residual Connection, and Horizon-Aware Gating—to assess their individual impact on model performance.

Our full model demonstrates the most robust predictive capability, achieving the best performance in 23 out

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Horizon</td><td colspan="2">Ours</td><td colspan="2">TimeCMA</td><td colspan="2">TimeLLM</td><td colspan="2">GPT4TS</td><td colspan="2">TimesNet</td><td colspan="2">DLinear</td><td colspan="2">PatchTST</td><td colspan="2">Fedformer</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.315</td><td>0.359</td><td>0.339</td><td>0.380</td><td>0.346</td><td>0.388</td><td>0.390</td><td>0.404</td><td>0.583</td><td>0.501</td><td>0.352</td><td>0.392</td><td>0.410</td><td>0.419</td><td>0.578</td><td>0.518</td></tr><tr><td>192</td><td>0.350</td><td>0.383</td><td>0.373</td><td>0.404</td><td>0.373</td><td>0.416</td><td>0.429</td><td>0.423</td><td>0.630</td><td>0.528</td><td>0.382</td><td>0.412</td><td>0.437</td><td>0.434</td><td>0.617</td><td>0.546</td></tr><tr><td>336</td><td>0.390</td><td>0.406</td><td>0.390</td><td>0.412</td><td>0.413</td><td>0.426</td><td>0.469</td><td>0.439</td><td>0.725</td><td>0.568</td><td>0.419</td><td>0.434</td><td>0.476</td><td>0.454</td><td>0.998</td><td>0.775</td></tr><tr><td>720</td><td>0.447</td><td>0.442</td><td>0.448</td><td>0.444</td><td>0.485</td><td>0.476</td><td>0.569</td><td>0.498</td><td>0.769</td><td>0.549</td><td>0.490</td><td>0.477</td><td>0.681</td><td>0.556</td><td>0.693</td><td>0.579</td></tr><tr><td>Avg</td><td>0.376</td><td>0.398</td><td>0.387</td><td>0.410</td><td>0.404</td><td>0.427</td><td>0.464</td><td>0.441</td><td>0.677</td><td>0.537</td><td>0.411</td><td>0.429</td><td>0.501</td><td>0.466</td><td>0.722</td><td>0.605</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.176</td><td>0.268</td><td>0.219</td><td>0.303</td><td>0.177</td><td>0.261</td><td>0.188</td><td>0.269</td><td>0.212</td><td>0.285</td><td>0.213</td><td>0.303</td><td>0.191</td><td>0.274</td><td>0.291</td><td>0.399</td></tr><tr><td>192</td><td>0.233</td><td>0.307</td><td>0.272</td><td>0.333</td><td>0.241</td><td>0.314</td><td>0.251</td><td>0.309</td><td>0.270</td><td>0.323</td><td>0.278</td><td>0.345</td><td>0.252</td><td>0.317</td><td>0.307</td><td>0.379</td></tr><tr><td>336</td><td>0.283</td><td>0.341</td><td>0.340</td><td>0.375</td><td>0.274</td><td>0.327</td><td>0.307</td><td>0.346</td><td>0.323</td><td>0.353</td><td>0.338</td><td>0.385</td><td>0.306</td><td>0.353</td><td>0.543</td><td>0.559</td></tr><tr><td>720</td><td>0.373</td><td>0.394</td><td>0.418</td><td>0.422</td><td>0.417</td><td>0.390</td><td>0.426</td><td>0.417</td><td>0.474</td><td>0.449</td><td>0.436</td><td>0.440</td><td>0.433</td><td>0.427</td><td>0.712</td><td>0.614</td></tr><tr><td>Avg</td><td>0.266</td><td>0.327</td><td>0.312</td><td>0.358</td><td>0.277</td><td>0.323</td><td>0.293</td><td>0.335</td><td>0.320</td><td>0.353</td><td>0.316</td><td>0.368</td><td>0.296</td><td>0.343</td><td>0.463</td><td>0.488</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.412</td><td>0.422</td><td>0.453</td><td>0.459</td><td>0.448</td><td>0.460</td><td>0.458</td><td>0.456</td><td>0.861</td><td>0.628</td><td>0.492</td><td>0.495</td><td>0.516</td><td>0.485</td><td>0.512</td><td>0.499</td></tr><tr><td>192</td><td>0.440</td><td>0.450</td><td>0.459</td><td>0.464</td><td>0.484</td><td>0.483</td><td>0.570</td><td>0.516</td><td>0.797</td><td>0.593</td><td>0.565</td><td>0.538</td><td>0.598</td><td>0.524</td><td>0.624</td><td>0.555</td></tr><tr><td>336</td><td>0.463</td><td>0.465</td><td>0.484</td><td>0.484</td><td>0.589</td><td>0.540</td><td>0.608</td><td>0.535</td><td>0.941</td><td>0.648</td><td>0.721</td><td>0.622</td><td>0.657</td><td>0.550</td><td>0.691</td><td>0.574</td></tr><tr><td>720</td><td>0.481</td><td>0.481</td><td>0.526</td><td>0.508</td><td>0.700</td><td>0.604</td><td>0.725</td><td>0.591</td><td>0.877</td><td>0.641</td><td>0.986</td><td>0.743</td><td>0.762</td><td>0.610</td><td>0.728</td><td>0.614</td></tr><tr><td>Avg</td><td>0.449</td><td>0.454</td><td>0.480</td><td>0.479</td><td>0.556</td><td>0.522</td><td>0.590</td><td>0.525</td><td>0.869</td><td>0.628</td><td>0.691</td><td>0.600</td><td>0.633</td><td>0.542</td><td>0.639</td><td>0.561</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.300</td><td>0.311</td><td>0.365</td><td>0.406</td><td>0.275</td><td>0.326</td><td>0.331</td><td>0.374</td><td>0.378</td><td>0.409</td><td>0.357</td><td>0.411</td><td>0.353</td><td>0.389</td><td>0.382</td><td>0.416</td></tr><tr><td>192</td><td>0.364</td><td>0.388</td><td>0.424</td><td>0.443</td><td>0.374</td><td>0.373</td><td>0.402</td><td>0.411</td><td>0.490</td><td>0.467</td><td>0.569</td><td>0.519</td><td>0.403</td><td>0.414</td><td>0.478</td><td>0.474</td></tr><tr><td>336</td><td>0.352</td><td>0.407</td><td>0.385</td><td>0.429</td><td>0.406</td><td>0.429</td><td>0.406</td><td>0.433</td><td>0.537</td><td>0.494</td><td>0.671</td><td>0.572</td><td>0.426</td><td>0.441</td><td>0.504</td><td>0.501</td></tr><tr><td>720</td><td>0.413</td><td>0.446</td><td>0.418</td><td>0.451</td><td>0.427</td><td>0.449</td><td>0.449</td><td>0.464</td><td>0.510</td><td>0.491</td><td>0.824</td><td>0.648</td><td>0.477</td><td>0.480</td><td>0.499</td><td>0.509</td></tr><tr><td>Avg</td><td>0.357</td><td>0.388</td><td>0.398</td><td>0.433</td><td>0.370</td><td>0.394</td><td>0.397</td><td>0.421</td><td>0.479</td><td>0.465</td><td>0.605</td><td>0.538</td><td>0.415</td><td>0.431</td><td>0.466</td><td>0.475</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.154</td><td>0.210</td><td>0.157</td><td>0.214</td><td>0.161</td><td>0.210</td><td>0.163</td><td>0.215</td><td>0.184</td><td>0.230</td><td>0.171</td><td>0.224</td><td>0.165</td><td>0.215</td><td>0.188</td><td>0.253</td></tr><tr><td>192</td><td>0.195</td><td>0.248</td><td>0.198</td><td>0.248</td><td>0.204</td><td>0.248</td><td>0.210</td><td>0.254</td><td>0.245</td><td>0.283</td><td>0.215</td><td>0.263</td><td>0.210</td><td>0.257</td><td>0.250</td><td>0.304</td></tr><tr><td>336</td><td>0.246</td><td>0.286</td><td>0.247</td><td>0.287</td><td>0.261</td><td>0.302</td><td>0.256</td><td>0.292</td><td>0.305</td><td>0.321</td><td>0.258</td><td>0.299</td><td>0.259</td><td>0.297</td><td>0.312</td><td>0.346</td></tr><tr><td>720</td><td>0.309</td><td>0.331</td><td>0.318</td><td>0.337</td><td>0.309</td><td>0.332</td><td>0.321</td><td>0.339</td><td>0.381</td><td>0.371</td><td>0.320</td><td>0.346</td><td>0.332</td><td>0.346</td><td>0.387</td><td>0.393</td></tr><tr><td>Avg</td><td>0.226</td><td>0.268</td><td>0.229</td><td>0.272</td><td>0.234</td><td>0.273</td><td>0.238</td><td>0.275</td><td>0.279</td><td>0.301</td><td>0.241</td><td>0.283</td><td>0.242</td><td>0.279</td><td>0.284</td><td>0.324</td></tr><tr><td colspan="2">1st Count</td><td>23</td><td>20</td><td>1</td><td>2</td><td>3</td><td>7</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td></td></tr></table>


Table 10: Few-shot forecasting results on  $10\%$  training data. Input sequence is set to 512 for all benchmark datasets. The prediction horizon,  $H \in \{96, 192, 336, 720\}$ . A lower score signals better performance. **Bold:** the best, **underline:** the second best.


<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Horizon</td><td colspan="2">Ours</td><td colspan="2">TimeCMA</td><td colspan="2">TimeLLM</td><td colspan="2">GPT4TS</td><td colspan="2">TimesNet</td><td colspan="2">DLinear</td><td colspan="2">PatchTST</td><td colspan="2">Fedformer</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.326</td><td>0.367</td><td>0.340</td><td>0.385</td><td>0.316</td><td>0.377</td><td>0.386</td><td>0.405</td><td>0.606</td><td>0.518</td><td>0.332</td><td>0.374</td><td>0.399</td><td>0.414</td><td>0.628</td><td>0.544</td></tr><tr><td>192</td><td>0.364</td><td>0.396</td><td>0.372</td><td>0.403</td><td>0.450</td><td>0.464</td><td>0.440</td><td>0.438</td><td>0.681</td><td>0.539</td><td>0.358</td><td>0.390</td><td>0.441</td><td>0.436</td><td>0.666</td><td>0.566</td></tr><tr><td>336</td><td>0.392</td><td>0.411</td><td>0.402</td><td>0.418</td><td>0.450</td><td>0.424</td><td>0.485</td><td>0.459</td><td>0.786</td><td>0.597</td><td>0.402</td><td>0.416</td><td>0.499</td><td>0.467</td><td>0.807</td><td>0.628</td></tr><tr><td>720</td><td>0.455</td><td>0.446</td><td>0.472</td><td>0.458</td><td>0.483</td><td>0.471</td><td>0.577</td><td>0.499</td><td>0.796</td><td>0.593</td><td>0.511</td><td>0.489</td><td>0.767</td><td>0.587</td><td>0.822</td><td>0.633</td></tr><tr><td>Avg</td><td>0.384</td><td>0.405</td><td>0.396</td><td>0.416</td><td>0.425</td><td>0.434</td><td>0.472</td><td>0.450</td><td>0.717</td><td>0.561</td><td>0.400</td><td>0.417</td><td>0.526</td><td>0.476</td><td>0.730</td><td>0.592</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.176</td><td>0.268</td><td>0.218</td><td>0.303</td><td>0.174</td><td>0.261</td><td>0.199</td><td>0.280</td><td>0.220</td><td>0.299</td><td>0.236</td><td>0.326</td><td>0.206</td><td>0.288</td><td>0.229</td><td>0.320</td></tr><tr><td>192</td><td>0.231</td><td>0.305</td><td>0.298</td><td>0.347</td><td>0.215</td><td>0.287</td><td>0.256</td><td>0.316</td><td>0.311</td><td>0.361</td><td>0.306</td><td>0.373</td><td>0.264</td><td>0.324</td><td>0.394</td><td>0.361</td></tr><tr><td>336</td><td>0.286</td><td>0.343</td><td>0.386</td><td>0.399</td><td>0.273</td><td>0.330</td><td>0.318</td><td>0.353</td><td>0.338</td><td>0.366</td><td>0.380</td><td>0.423</td><td>0.334</td><td>0.367</td><td>0.378</td><td>0.427</td></tr><tr><td>720</td><td>0.375</td><td>0.405</td><td>0.416</td><td>0.419</td><td>0.433</td><td>0.412</td><td>0.460</td><td>0.436</td><td>0.509</td><td>0.465</td><td>0.674</td><td>0.583</td><td>0.454</td><td>0.432</td><td>0.523</td><td>0.510</td></tr><tr><td>Avg</td><td>0.267</td><td>0.330</td><td>0.329</td><td>0.367</td><td>0.274</td><td>0.323</td><td>0.308</td><td>0.346</td><td>0.344</td><td>0.372</td><td>0.399</td><td>0.426</td><td>0.314</td><td>0.352</td><td>0.381</td><td>0.404</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.417</td><td>0.435</td><td>0.457</td><td>0.460</td><td>0.483</td><td>0.464</td><td>0.543</td><td>0.506</td><td>0.892</td><td>0.625</td><td>0.547</td><td>0.503</td><td>0.557</td><td>0.519</td><td>0.593</td><td>0.529</td></tr><tr><td>192</td><td>0.441</td><td>0.447</td><td>0.460</td><td>0.461</td><td>0.629</td><td>0.540</td><td>0.748</td><td>0.580</td><td>0.940</td><td>0.665</td><td>0.720</td><td>0.604</td><td>0.711</td><td>0.570</td><td>0.652</td><td>0.563</td></tr><tr><td>336</td><td>0.467</td><td>0.470</td><td>0.500</td><td>0.489</td><td>0.768</td><td>0.626</td><td>0.754</td><td>0.595</td><td>0.945</td><td>0.653</td><td>0.984</td><td>0.727</td><td>0.816</td><td>0.619</td><td>0.731</td><td>0.594</td></tr><tr><td>720</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Avg</td><td>0.442</td><td>0.451</td><td>0.472</td><td>0.470</td><td>0.627</td><td>0.543</td><td>0.681</td><td>0.560</td><td>0.925</td><td>0.647</td><td>0.750</td><td>0.611</td><td>0.694</td><td>0.569</td><td>0.658</td><td>0.562</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.306</td><td>0.367</td><td>0.363</td><td>0.409</td><td>0.336</td><td>0.397</td><td>0.376</td><td>0.421</td><td>0.409</td><td>0.420</td><td>0.442</td><td>0.456</td><td>0.401</td><td>0.421</td><td>0.390</td><td>0.424</td></tr><tr><td>192</td><td>0.369</td><td>0.409</td><td>0.404</td><td>0.434</td><td>0.406</td><td>0.425</td><td>0.418</td><td>0.441</td><td>0.483</td><td>0.464</td><td>0.617</td><td>0.542</td><td>0.452</td><td>0.455</td><td>0.457</td><td>0.465</td></tr><tr><td>336</td><td>0.395</td><td>0.432</td><td>0.418</td><td>0.447</td><td>0.405</td><td>0.432</td><td>0.408</td><td>0.439</td><td>0.499</td><td>0.479</td><td>1.424</td><td>0.849</td><td>0.464</td><td>0.469</td><td>0.477</td><td>0.483</td></tr><tr><td>720</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td></td></tr><tr><td>Avg</td><td>0.357</td><td>0.403</td><td>0.395</td><td>0.430</td><td>0.382</td><td>0.418</td><td>0.400</td><td>0.433</td><td>0.439</td><td>0.448</td><td>0.694</td><td>0.577</td><td>0.827</td><td>0.615</td><td>0.463</td><td>0.454</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.154</td><td>0.211</td><td>0.160</td><td>0.216</td><td>0.172</td><td>0.263</td><td>0.175</td><td>0.230</td><td>0.207</td><td>0.253</td><td>0.184</td><td>0.242</td><td>0.171</td><td>0.224</td><td>0.229</td><td>0.309</td></tr><tr><td>192</td><td>0.194</td><td>0.245</td><td>0.199</td><td>0.251</td><td>0.224</td><td>0.271</td><td>0.227</td><td>0.276</td><td>0.272</td><td>0.307</td><td>0.228</td><td>0.283</td><td>0.230</td><td>0.277</td><td>0.265</td><td>0.317</td></tr><tr><td>336</td><td>0.244</td><td>0.286</td><td>0.247</td><td>0.287</td><td>0.282</td><td>0.321</td><td>0.286</td><td>0.322</td><td>0.313</td><td>0.328</td><td>0.279</td><td>0.322</td><td>0.294</td><td>0.326</td><td>0.353</td><td>0.392</td></tr><tr><td>720</td><td>0.310</td><td>0.333</td><td>0.318</td><td>0.336</td><td>0.366</td><td>0.381</td><td>0.366</td><td>0.379</td><td>0.400</td><td>0.385</td><td>0.364</td><td>0.388</td><td>0.384</td><td>0.387</td><td>0.391</td><td>0.394</td></tr><tr><td>Avg</td><td>0.226</td><td>0.269</td><td>0.231</td><td>0.273</td><td>0.260</td><td>0.309</td><td>0.263</td><td>0.301</td><td>0.298</td><td>0.318</td><td>0.263</td><td>0.308</td><td>0.269</td><td>0.303</td><td>0.309</td><td>0.353</td></tr><tr><td colspan="2">1st Count</td><td>19</td><td>19</td><td>0</td><td>1</td><td>4</td><td>4</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td></td></tr></table>

Table 11: Few-shot forecasting results on  $5\%$  training data. Input sequence is set to 512 for all benchmark datasets. The prediction horizon,  $H \in \{96, 192, 336, 720\}$ . ‘-’ indicates that  $5\%$  of the time series data was insufficient to form a viable training set. A lower score signals better performance. **Bold:** the best, **underline:** the second best.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Horizon</td><td colspan="2">Ours</td><td colspan="2">w/o Frequency Module</td><td colspan="2">w/o Multihead CMA</td><td colspan="2">w/o Residual Connection</td><td colspan="2">w/o Gating Mechanism</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.308</td><td>0.354</td><td>0.310</td><td>0.353</td><td>0.311</td><td>0.357</td><td>0.333</td><td>0.372</td><td>0.315</td><td>0.359</td></tr><tr><td>192</td><td>0.357</td><td>0.381</td><td>0.359</td><td>0.380</td><td>0.356</td><td>0.382</td><td>0.393</td><td>0.407</td><td>0.358</td><td>0.381</td></tr><tr><td>336</td><td>0.382</td><td>0.400</td><td>0.392</td><td>0.402</td><td>0.385</td><td>0.402</td><td>0.420</td><td>0.423</td><td>0.383</td><td>0.399</td></tr><tr><td>720</td><td>0.442</td><td>0.437</td><td>0.460</td><td>0.441</td><td>0.444</td><td>0.439</td><td>0.468</td><td>0.449</td><td>0.435</td><td>0.435</td></tr><tr><td>Avg</td><td>0.372</td><td>0.393</td><td>0.381</td><td>0.394</td><td>0.374</td><td>0.395</td><td>0.404</td><td>0.413</td><td>0.373</td><td>0.394</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.172</td><td>0.254</td><td>0.174</td><td>0.257</td><td>0.175</td><td>0.258</td><td>0.179</td><td>0.260</td><td>0.170</td><td>0.252</td></tr><tr><td>192</td><td>0.237</td><td>0.300</td><td>0.241</td><td>0.302</td><td>0.243</td><td>0.301</td><td>0.257</td><td>0.308</td><td>0.245</td><td>0.301</td></tr><tr><td>336</td><td>0.306</td><td>0.337</td><td>0.300</td><td>0.338</td><td>0.308</td><td>0.340</td><td>0.314</td><td>0.347</td><td>0.301</td><td>0.337</td></tr><tr><td>720</td><td>0.400</td><td>0.398</td><td>0.400</td><td>0.398</td><td>0.406</td><td>0.402</td><td>0.402</td><td>0.402</td><td>0.403</td><td>0.399</td></tr><tr><td>Avg</td><td>0.279</td><td>0.322</td><td>0.279</td><td>0.324</td><td>0.283</td><td>0.325</td><td>0.288</td><td>0.329</td><td>0.280</td><td>0.322</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.371</td><td>0.397</td><td>0.373</td><td>0.398</td><td>0.371</td><td>0.399</td><td>0.386</td><td>0.414</td><td>0.370</td><td>0.397</td></tr><tr><td>192</td><td>0.411</td><td>0.421</td><td>0.425</td><td>0.423</td><td>0.411</td><td>0.422</td><td>0.434</td><td>0.438</td><td>0.417</td><td>0.422</td></tr><tr><td>336</td><td>0.448</td><td>0.441</td><td>0.469</td><td>0.443</td><td>0.454</td><td>0.444</td><td>0.467</td><td>0.456</td><td>0.454</td><td>0.441</td></tr><tr><td>720</td><td>0.441</td><td>0.460</td><td>0.465</td><td>0.467</td><td>0.447</td><td>0.463</td><td>0.447</td><td>0.464</td><td>0.461</td><td>0.468</td></tr><tr><td>Avg</td><td>0.418</td><td>0.430</td><td>0.433</td><td>0.433</td><td>0.421</td><td>0.432</td><td>0.433</td><td>0.443</td><td>0.425</td><td>0.432</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.278</td><td>0.338</td><td>0.296</td><td>0.348</td><td>0.287</td><td>0.342</td><td>0.316</td><td>0.363</td><td>0.289</td><td>0.343</td></tr><tr><td>192</td><td>0.351</td><td>0.389</td><td>0.369</td><td>0.399</td><td>0.367</td><td>0.397</td><td>0.395</td><td>0.412</td><td>0.371</td><td>0.404</td></tr><tr><td>336</td><td>0.358</td><td>0.398</td><td>0.385</td><td>0.414</td><td>0.362</td><td>0.400</td><td>0.404</td><td>0.427</td><td>0.372</td><td>0.407</td></tr><tr><td>720</td><td>0.404</td><td>0.433</td><td>0.406</td><td>0.432</td><td>0.402</td><td>0.431</td><td>0.422</td><td>0.443</td><td>0.418</td><td>0.437</td></tr><tr><td>Avg</td><td>0.348</td><td>0.390</td><td>0.364</td><td>0.398</td><td>0.355</td><td>0.392</td><td>0.384</td><td>0.411</td><td>0.363</td><td>0.398</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.162</td><td>0.210</td><td>0.169</td><td>0.213</td><td>0.167</td><td>0.212</td><td>0.167</td><td>0.214</td><td>0.161</td><td>0.208</td></tr><tr><td>192</td><td>0.211</td><td>0.253</td><td>0.214</td><td>0.259</td><td>0.212</td><td>0.253</td><td>0.212</td><td>0.255</td><td>0.211</td><td>0.254</td></tr><tr><td>336</td><td>0.267</td><td>0.293</td><td>0.277</td><td>0.299</td><td>0.276</td><td>0.298</td><td>0.270</td><td>0.296</td><td>0.278</td><td>0.301</td></tr><tr><td>720</td><td>0.335</td><td>0.346</td><td>0.340</td><td>0.349</td><td>0.344</td><td>0.345</td><td>0.348</td><td>0.355</td><td>0.347</td><td>0.345</td></tr><tr><td>Avg</td><td>0.244</td><td>0.275</td><td>0.250</td><td>0.280</td><td>0.249</td><td>0.277</td><td>0.249</td><td>0.280</td><td>0.249</td><td>0.277</td></tr><tr><td rowspan="5">Exchange Rate</td><td>96</td><td>0.085</td><td>0.205</td><td>0.085</td><td>0.204</td><td>0.085</td><td>0.205</td><td>0.099</td><td>0.224</td><td>0.086</td><td>0.207</td></tr><tr><td>192</td><td>0.172</td><td>0.296</td><td>0.171</td><td>0.294</td><td>0.174</td><td>0.298</td><td>0.182</td><td>0.307</td><td>0.173</td><td>0.298</td></tr><tr><td>336</td><td>0.318</td><td>0.408</td><td>0.327</td><td>0.416</td><td>0.321</td><td>0.412</td><td>0.347</td><td>0.433</td><td>0.335</td><td>0.421</td></tr><tr><td>720</td><td>0.836</td><td>0.696</td><td>0.914</td><td>0.726</td><td>0.933</td><td>0.737</td><td>0.955</td><td>0.744</td><td>0.898</td><td>0.723</td></tr><tr><td>Avg</td><td>0.353</td><td>0.401</td><td>0.374</td><td>0.410</td><td>0.378</td><td>0.413</td><td>0.396</td><td>0.427</td><td>0.373</td><td>0.412</td></tr><tr><td rowspan="5">ILI</td><td>24</td><td>1.583</td><td>0.802</td><td>1.618</td><td>0.813</td><td>1.676</td><td>0.807</td><td>2.764</td><td>1.089</td><td>1.603</td><td>0.793</td></tr><tr><td>36</td><td>1.601</td><td>0.820</td><td>1.683</td><td>0.836</td><td>1.510</td><td>0.806</td><td>1.967</td><td>0.911</td><td>1.643</td><td>0.823</td></tr><tr><td>48</td><td>1.718</td><td>0.815</td><td>1.865</td><td>0.948</td><td>2.114</td><td>0.935</td><td>1.955</td><td>0.940</td><td>1.638</td><td>0.812</td></tr><tr><td>60</td><td>1.920</td><td>0.901</td><td>1.977</td><td>0.939</td><td>1.955</td><td>0.912</td><td>2.018</td><td>0.938</td><td>2.013</td><td>0.919</td></tr><tr><td>Avg</td><td>1.705</td><td>0.835</td><td>1.786</td><td>0.884</td><td>1.813</td><td>0.865</td><td>2.176</td><td>0.969</td><td>1.724</td><td>0.837</td></tr><tr><td colspan="2">1st Count</td><td>23</td><td>22</td><td>5</td><td>5</td><td>5</td><td>5</td><td>0</td><td>0</td><td>6</td><td>9</td></tr></table>

Table 12: Full ablation analysis of T3Time across seven datasets and forecast horizons. We evaluate the impact of removing four core components—Frequency Encoding, Multi-Head Cross-Modal Alignment, Channel-Wise Residual Connection, and Horizon-Aware Gating. Metrics reported are MSE and MAE (lower is better). Bold: the best.

of 35 MSE cases and 21 out of 35 MAE cases, significantly outperforming all ablated variants. Among the variants, the model without the Gating Mechanism exhibits relatively competitive results, attaining the best MSE in 6 cases and best MAE in 9 cases. However, removing other modules—particularly the Residual Connection—leads to consistent and substantial performance degradation, indicating

their critical role in maintaining deep temporal and semantic alignment. Averaged over all datasets and prediction lengths, the complete T3Time model achieves the lowest MSE and MAE, underscoring the contribution of each component and validating our architectural design choices.

# Appendix F: t-SNE Visualization

Figure 6 illustrates a comprehensive t-SNE visualization of the learned embeddings across six benchmark datasets (ETTm1, ETTm2, ETTh1, ETTh2, Weather, and ILI). For each dataset, we visualize four key embedding types extracted at different stages of the model: time series embeddings, frequency-domain embeddings, prompt embeddings, and forecasted (output) embeddings.

The time and frequency embeddings generally exhibit more dispersed or fragmented clustering, reflecting the challenge of modeling complex temporal and periodic patterns in isolation. In contrast, the prompt embeddings form denser, well-separated clusters, indicating that the LLM-encoded prompts inject strong semantic structure into the latent space. Notably, the forecasted embeddings display smooth, often spiral or circular manifolds, suggesting that the complete model learns compact and coherent representations that effectively align multiple modalities through cross-modal attention and residual refinement.


ETTm1: Time Series Embeddings


![](images/0c5cf95666bc2e82c308ac3cccc4e411c9ed7dd1de074bcd8408afbf2433ef2c.jpg)



ETTm1: Frequency Embeddings


![](images/879f50461160e438cae2a3c45e1c87c6a0e2205da302312825685573c2ce6d5a.jpg)



ETTm1: Prompt Embeddings


![](images/96dd485c6d02711c360322ca0ceddd4d5f1d583de08a68515ded75f3d7912111.jpg)



ETTm1: Forecasted Embeddings


![](images/5dcd6d47a409928f9f1c72058e2347f176cf1f80d273ac62e04c744918bf700f.jpg)



ETTm2: Time Series Embeddings


![](images/0f70b3bdcc55922a91e10df09d298160938298586d6d763252f9259b0134091f.jpg)



ETTm2: Frequency Embeddings


![](images/d99e75607e05ff9e20c847bcf12b7ab15794dbbffc827e7929683113f6eccfa5.jpg)



ETTm2: Prompt Embeddings


![](images/85a05cc0bdb0ad79c151e99dd28c8412628c4ae71961eeb8385654a795ba1d3f.jpg)



ETTm2: Forecasted Embeddings


![](images/e611a6b2055a9efa11028f025429fb4250c29beaca783d9f75ab0ccac6b5488e.jpg)



ETTh1: Time Series Embeddings


![](images/ab9527c2c85b8d2a36e65806243dcde914c2f9d91d8ade82e00ded689069fcb3.jpg)



ETTh1: Frequency Embeddings


![](images/ea29db0a5460948d3d91807d567132191510fef6042e85e5c867d70a4b6661f0.jpg)



ETTh1: Prompt Embeddings


![](images/404fde97f98c834112296a66afc5086a64bd420e1965a23e674ec23732ea9d27.jpg)



ETTh1: Forecasted Embeddings


![](images/7cac2d586b5c79896e85be07c9ee74a36f41ac132571efa3d615e649f12a2ed9.jpg)



ETTh2: Time Series Embeddings


![](images/48b9f2f97f06a2c9c8771842a4350036fc74c0ab340936bec998624b06390a62.jpg)



ETTh2: Frequency Embeddings


![](images/1a8aafb2136fe80656ff454b0deef64b0076dafbfc086b38434754b9ea2a616a.jpg)



ETTh2: Prompt Embeddings


![](images/26e4ae1371a2e32a8a3bb6aa6682c1175827b99b5c516f2ace9ecc5c31423352.jpg)



ETTh2: Forecasted Embeddings


![](images/d2269ea5dddeeafe0d2d8412ce8c94528ca3b171feb695c4a128c6898d79e3b9.jpg)



Weather: Time Series Embeddings


![](images/69d0113cc991b03206c1cfe600011d127a7dff02055125aaa1c47769a66f2096.jpg)



Weather: Frequency Embeddings


![](images/9b6c562eac7e2b7c1d5e54c0b12c29e7d4e378405695e7c93ca447627b213541.jpg)



Weather: Prompt Embeddings


![](images/dfed76b299c02388a33334fb46905789406b27f233acf50f089cc91fce474cac.jpg)



Weather: Forecasted Embeddings


![](images/51018567d3489f70724093473c74f13616ae918550264de787770dabb28c1b1d.jpg)



ILI: Time Series Embeddings


![](images/504d786c3c1c55d7f8681bd2337fcc5805aed2c9a67a2d085e891b42e671666a.jpg)



ILI: Frequency Embeddings


![](images/01c3687202268e1830812ff3e4833a6cea3b8d6a5c7a2960e86710129493ad12.jpg)



ILI: Prompt Embeddings


![](images/22fef36c99fc2bcf57a31fd75ff9ec55374eb71a2135e666ebac4014163adcd8.jpg)



ILI: Forecasted Embeddings


![](images/af551be693113c70002d4853b36b40c29e4740c29310932c24e1101237a7d624.jpg)



Figure 6: t-SNE visualization of four types of learned embeddings (time series, frequency, prompt, and forecasted) across six datasets. The prompt and forecasted embeddings exhibit clear, coherent clustering, while the time series and frequency embeddings appear more fragmented.
