# Frequency-domain MLPs are More EffectiveLearners in Time Series Forecasting

Kun Yi1, Qi Zhang2, Wei $\mathbf { F a n ^ { 3 } }$ , Shoujin Wang4, Pengyang Wang5, Hui He1Defu Lian6, Ning $\mathbf { A n } ^ { 7 }$ , Longbing $\mathbf { C a o ^ { 8 } }$ , Zhendong Niu1∗

1Beijing Institute of Technology, 2Tongji University, 3University of Oxford

4University of Technology Sydney, 5University of Macau, 6USTC

7HeFei University of Technology, 8Macquarie University

{yikun, hehui617, zniu}@bit.edu.cn, zhangqi_cs $@$ tongji.edu.cn, weifan.oxford $@$ gmail.com

pywang@um.edu.mo, liandefu@ustc.edu.cn, ning.g.an@acm.org, longbing.cao@mq.edu.au

# Abstract

Time series forecasting has played the key role in different industrial, includingfinance, traffic, energy, and healthcare domains. While existing literatures have de-signed many sophisticated architectures based on RNNs, GNNs, or Transformers,another kind of approaches based on multi-layer perceptrons (MLPs) are pro-posed with simple structure, low complexity, and superior performance. However,most MLP-based forecasting methods suffer from the point-wise mappings andinformation bottleneck, which largely hinders the forecasting performance. Toovercome this problem, we explore a novel direction of applying MLPs in thefrequency domain for time series forecasting. We investigate the learned patterns offrequency-domain MLPs and discover their two inherent characteristic benefitingforecasting, (i) global view: frequency spectrum makes MLPs own a complete viewfor signals and learn global dependencies more easily, and (ii) energy compaction:frequency-domain MLPs concentrate on smaller key part of frequency componentswith compact signal energy. Then, we propose FreTS, a simple yet effective archi-tecture built upon Frequency-domain MLPs for Time Series forecasting. FreTSmainly involves two stages, (i) Domain Conversion, that transforms time-domainsignals into complex numbers of frequency domain; (ii) Frequency Learning, thatperforms our redesigned MLPs for the learning of real and imaginary part of fre-quency components. The above stages operated on both inter-series and intra-seriesscales further contribute to channel-wise and time-wise dependency learning. Ex-tensive experiments on 13 real-world benchmarks (including 7 benchmarks forshort-term forecasting and 6 benchmarks for long-term forecasting) demonstrateour consistent superiority over state-of-the-art methods. Code is available at thisrepository: https://github.com/aikunyi/FreTS.

# 1 Introduction

Time series forecasting has been a critical role in a variety of real-world industries, such as climatecondition estimation [1, 2, 3], traffic state prediction [4, 5, 6], economic analysis [7, 8], etc. In theearly stage, many traditional statistical forecasting methods have been proposed, such as exponentialsmoothing [9] and auto-regressive moving averages (ARMA) [10]. Recently, the emerging devel-opment of deep learning has fostered many deep forecasting models, including Recurrent NeuralNetwork-based methods (e.g., DeepAR [11], LSTNet [12]), Convolution Neural Network-based meth-ods (e.g., TCN [13], SCINet [14]), Transformer-based methods (e.g., Informer [15], Autoformer [16]),and Graph Neural Network-based methods (e.g., MTGNN [17], StemGNN [18], AGCRN [19]), etc.

![](images/e68f72a2acb9950f8e4825587a8dbbf4616a5714ec02cba33c5c475100934a64.jpg)



(a) Left: time domain. Right: frequency domain.


![](images/0f6338e5b3f9ba6753bfe1e3ec14e82d256021bee18b02f7942ecaffaa302577.jpg)



(b) Left: time domain. Right: frequency domain.



Figure 1: Visualizations of the learned patterns of MLPs in the time domain and the frequencydomain (see Appendix B.4). (a) global view: the patterns learned in the frequency domain exhibitsmore obvious global periodic patterns than the time domain; (b) energy compaction: learning in thefrequency domain can identify clearer diagonal dependencies and key patterns than the time domain.


While these deep models have achieved promising forecasting performance in certain scenarios,their sophisticated network architectures would usually bring up expensive computation burdenin training or inference stage. Besides, the robustness of these models could be easily influencedwith a large amount of parameters, especially when the available training data is limited [15, 20].Therefore, the methods based on multi-layer perceptrons (MLPs) have been recently introduced withsimple structure, low complexity, and superior forecasting performance, such as N-BEATS [21],LightTS [22], DLinear [23], etc. However, these MLP-based methods rely on point-wise mappingsto capture temporal mappings, which cannot handle global dependencies of time series. Moreover,they would suffer from the information bottleneck with regard to the volatile and redundant localmomenta of time series, which largely hinders their performance for time series forecasting.

To overcome the above problems, we explore a novel direction of applying MLPs in the frequencydomain for time series forecasting. We investigate the learned patterns of frequency-domain MLPsin forecasting and have discovered their two key advantages: (i) global view: operating on spectralcomponents acquired from series transformation, frequency-domain MLPs can capture a morecomplete view of signals, making it easier to learn global spatial/temporal dependencies. (ii) energycompaction: frequency-domain MLPs concentrate on the smaller key part of frequency componentswith the compact signal energy, and thus can facilitate preserving clearer patterns while filteringout influence of noises. Experimentally, we have observed that frequency-domain MLPs capturemuch more obvious global periodic patterns than the time-domain MLPs from Figure 1(a), whichhighlights their ability to recognize global signals. Also, from Figure 1(b), we easily note a muchmore clear diagonal dependency in the learned weights of frequency-domain MLPs, compared withthe more scattered dependency learned by time-domain MLPs. This illustrates the great potentialof frequency-domain MLPs to identify most important features and key patterns while handlingcomplicated and noisy information.

To fully utilize these advantages, we propose FreTS, a simple yet effective architecture of Frequency-domain MLPs for Time Series forecasting. The core idea of FreTS is to learn the time seriesforecasting mappings in the frequency domain. Specifically, FreTS mainly involves two stages: (i)Domain Conversion: the original time-domain series signals are first transformed into frequency-domain spectrum on top of Discrete Fourier Transform (DFT) [24], where the spectrum is composed ofseveral complex numbers as frequency components, including the real coefficients and the imaginarycoefficients. (ii) Frequency Learning: given the real/imaginary coefficients, we redesign the frequency-domain MLPs originally for the complex numbers by separately considering the real mappings andimaginary mappings. The respective real/imaginary parts of output learned by two distinct MLPs arethen stacked in order to recover from frequency components to the final forecasting. Also, FreTSperforms above two stages on both inter-series and intra-series scales, which further contributes to thechannel-wise and time-wise dependencies in the frequency domain for better forecasting performance.We conduct extensive experiments on 13 benchmarks under different settings, covering 7 benchmarksfor short-term forecasting and 6 benchmarks for long-term forecasting, which demonstrate ourconsistent superiority compared with state-of-the-art methods.

# 2 Related Work

Forecasting in the Time Domain Traditionally, statistical methods have been proposed for forecast-ing in the time domain, including (ARMA) [10], VAR [25], and ARIMA [26]. Recently, deep learning

based methods have been widely used in time series forecasting due to their capability of extractingnonlinear and complex correlations [27, 28]. These methods have learned the dependencies in thetime domain with RNNs (e.g., deepAR [11], LSTNet [12]) and CNNs (e.g., TCN [13], SCINet [14]).In addition, GNN-based models have been proposed with good forecasting performance because oftheir good abilities to model series-wise dependencies among variables in the time domain, such asTAMP-S2GCNets [5], AGCRN [19], MTGNN [17], and GraphWaveNet [29]. Besides, Transformer-based forecasting methods have been introduced due to their attention mechanisms for long-rangedependency modeling ability in the time domain, such as Reformer [20] and Informer [15].

Forecasting in the Frequency Domain Several recent time series forecasting methods have ex-tracted knowledge of the frequency domain for forecasting [30]. Specifically, SFM [31] decomposesthe hidden state of LSTM into frequencies by Discrete Fourier Transform (DFT). StemGNN [18]performs graph convolutions based on Graph Fourier Transform (GFT) and computes series corre-lations based on Discrete Fourier Transform. Autoformer [16] replaces self-attention by proposingthe auto-correlation mechanism implemented with Fast Fourier Transforms (FFT). FEDformer [32]proposes a DFT-based frequency enhanced attention, which obtains the attentive weights by thespectrums of queries and keys, and calculates the weighted sum in the frequency domain. CoST [33]uses DFT to map the intermediate features to frequency domain to enables interactions in representa-tion. FiLM [34] utilizes Fourier analysis to preserve historical information and remove noisy signals.Unlike these efforts that leverage frequency techniques to improve upon the original architecture suchas Transformer and GNN, in this paper, we propose a new frequency learning architecture that learnsboth channel-wise and time-wise dependencies in the frequency domain.

MLP-based Forecasting Models Several studies have explored the use of MLP-based networks intime series forecasting. N-BEATS [21] utilizes stacked MLP layers together with doubly residuallearning to process the input data to iteratively forecast the future. DEPTS [35] applies Fouriertransform to extract periods and MLPs for periodicity dependencies for univariate forecasting.LightTS [22] uses lightweight sampling-oriented MLP structures to reduce complexity and com-putation time while maintaining accuracy. N-HiTS [36] combines multi-rate input sampling andhierarchical interpolation with MLPs for univariate forecasting. LTSF-Linear [37] proposes a setof embarrassingly simple one-layer linear model to learn temporal relationships between input andoutput sequences. These studies demonstrate the effectiveness of MLP-based networks in time seriesforecasting tasks, and inspire the development of our frequency-domain MLPs in this paper.

# 3 FreTS

In this section, we elaborate on our proposed novel approach, FreTS, based on our redesigned MLPsin the frequency domain for time series forecasting. First, we present the detailed frequency learningarchitecture of FreTS in Section 3.1, which mainly includes two-fold frequency learners with domainconversions. Then, we detailedly introduce our redesigned frequency-domain $M L P s$ adopted byabove frequency learners in Section 3.2. Besides, we also theoretically analyze their superior natureof global view and energy compaction, as aforementioned in Section 1.

Problem Definition Let $[ X _ { 1 } , X _ { 2 } , \cdot \cdot \cdot , X _ { T } ] \in \mathbb { R } ^ { N \times T }$ stand for the regularly sampled multi-variatetime series dataset with $N$ series and $T$ timestamps, where $X _ { t } \in \mathbb { R } ^ { N }$ denotes the multi-variatevalues of $N$ distinct series at timestamp $t$ . We consider a time series lookback window of length- $L$at timestamp $t$ as the model input, namely $\mathbf { X } _ { t } = [ X _ { t - L + 1 } , X _ { t - L + 2 } , \cdot \cdot \cdot \ , X _ { t } ] \in \mathbb { R } ^ { N \times L }$ ; also, weconsider a horizon window of length- $\tau$ at timestamp $t$ as the prediction target, denoted as $\mathbf { Y } _ { t } =$$[ X _ { t + 1 } , X _ { t + 2 } , \cdot \cdot \cdot , X _ { t + \tau } ] \in \mathbb { R } ^ { N \times \tau }$ . Then the time series forecasting formulation is to use historicalobservations $\mathbf { X } _ { t }$ to predict future values $\hat { \mathbf { Y } } _ { t }$ and the typical forecasting model $f _ { \theta }$ parameterized by $\theta$is to produce forecasting results by $\hat { \mathbf Y } _ { t } = f _ { \theta } ( \mathbf X _ { t } )$ .

# 3.1 Frequency Learning Architecture

The frequency learning architecture of FreTS is depicted in Figure 2, which mainly involves DomainConversion/Inversion stages, Frequency-domain MLPs, and the corresponding two learners, i.e.,the Frequency Channel Learner and the Frequency Temporal Learner. Besides, before taken tolearners, we concretely apply a dimension extension block on model input to enhance the modelcapability. Specifically, the input lookback window $\mathbf { X } _ { t } \in \mathbb { R } ^ { N \times L }$ is multiplied with a learnable

![](images/e2a9e3e11d0154f52beff7ddabe0dde243b96c70e843514cec55ea36edc8857c.jpg)



Figure 2: The framework overview of FreTS: the Frequency Channel Learner focuses on modelinginter-series dependencies with frequency-domain MLPs operating on the channel dimensions; theFrequency Temporal Learner is to capture the temporal dependencies by performing frequency-domain MLPs on the time dimensions.


weight vector $\phi _ { d } \in \mathbb { R } ^ { 1 \times d }$ to obtain a more expressive hidden representation $\mathbf { H } _ { t } \in \mathbb { R } ^ { N \times L \times d }$ , yielding$\mathbf { H } _ { t } = \mathbf { X } _ { t } \times \phi _ { d }$ to bring more semantic information, inspired by word embeddings [38].

Domain Conversion/Inversion The use of Fourier transform enables the decomposition of atime series signal into its constituent frequencies. This is particularly advantageous for time seriesanalysis since it benefits to identify periodic or trend patterns in the data, which are often important inforecasting tasks. As aforementioned in Figure 1(a), learning in the frequency spectrum helps capturea greater number of periodic patterns. In view of this, we convert the input $\mathbf { H }$ into the frequencydomain $\mathcal { H }$ by:

$$
\mathcal {H} (f) = \int_ {- \infty} ^ {\infty} \mathbf {H} (v) e ^ {- j 2 \pi f v} \mathrm {d} v = \int_ {- \infty} ^ {\infty} \mathbf {H} (v) \cos (2 \pi f v) \mathrm {d} v + j \int_ {- \infty} ^ {\infty} \mathbf {H} (v) \sin (2 \pi f v) \mathrm {d} v \tag {1}
$$

where $f$ is the frequency variable, $v$ is the integral variable, and $j$ is the imaginary unit, which isdefined as the square root of -1; $\begin{array} { r } { \int _ { - \infty } ^ { \infty } { \bf H } ( v ) \cos ( 2 \pi f v ) \mathrm { d } v } \end{array}$ is the real part of $\mathcal { H }$ and is abbreviatedas $R e ( \mathcal { H } )$ ; $\begin{array} { r } { \int _ { - \infty } ^ { \infty } \mathbf { H } ( v ) \sin ( 2 \pi f v ) \mathrm { d } v } \end{array}$ is the imaginary part and is abbreviated as $I m ( \mathcal { H } )$ . Then wecan rewrite $\mathcal { H }$ in Equation (1) as: $\mathcal { H } = R e ( \mathcal { H } ) + j I m ( \mathcal { H } )$ . Note that in FreTS we operate domainconversion on both the channel dimension and time dimension, respectively. Once completing thelearning in the frequency domain, we can convert $\mathcal { H }$ back into the the time domain using the followinginverse conversion formulation:

$$
\mathbf {H} (v) = \int_ {- \infty} ^ {\infty} \mathcal {H} (f) e ^ {j 2 \pi f v} \mathrm {d} f = \int_ {- \infty} ^ {\infty} \left(R e (\mathcal {H} (f)) + j I m (\mathcal {H} (f)) e ^ {j 2 \pi f v} \mathrm {d} f \right. \tag {2}
$$

where we take frequency $f$ as the integral variable. In fact, the frequency spectrum is expressed as acombination of cos and sin waves in $\mathcal { H }$ with different frequencies and amplitudes inferring differentperiodic properties in time series signals. Thus examining the frequency spectrum can better discernthe prominent frequencies and periodic patterns in time series. In the following sections, we useDomainConversion to stand for Equation (1), and DomainInversion for Equation (2) for brevity.

Frequency Channel Learner Considering channel dependencies for time series forecasting isimportant because it allows the model to capture interactions and correlations between differentvariables, leading to a more accurate predictions. The frequency channel learner enables commu-nications between different channels; it operates on each timestamp by sharing the same weightsbetween $L$ timestamps to learn channel dependencies. Concretely, the frequency channel learnertakes $\mathbf { H } _ { t } \in \mathbb { R } ^ { N \times L \times d }$ as input. Given the $l$ -th timestamp $\mathbf { H } _ { t } ^ { : , ( l ) } \in \dot { \mathbb { R } } ^ { N \times d }$ , we perform the frequencychannel learner by:

$$
\mathcal {H} _ {c h a n} ^ {:, (l)} = \operatorname {D o m a i n C o n v e r s i o n} _ {(c h a n)} \left(\mathbf {H} _ {t} ^ {:, (l)}\right)
$$

$$
\mathcal {Z} _ {\text {c h a n}} ^ {:, (l)} = \operatorname {F r e M L P} \left(\mathcal {H} _ {\text {c h a n}} ^ {:, (l)}, \mathcal {W} ^ {\text {c h a n}}, \mathcal {B} ^ {\text {c h a n}}\right) \tag {3}
$$

$$
\mathbf {Z} ^ {:, (l)} = \operatorname {D o m a i n I n v e r s i o n} _ {(c h a n)} \left(\mathcal {Z} _ {c h a n} ^ {:, (l)}\right)
$$

where $\mathcal { H } _ { c h a n } ^ { : , ( l ) } \ \in \ \mathbb { C } ^ { \frac { N } { 2 } \times d }$ is the frequency components of $\mathbf { H } _ { t } ^ { : , ( l ) }$ ; DomainConversion $( c h a n )$ andDomainInversion(chan) indicates such operations are performed along the channel dimension.FreMLP are frequency-domain MLPs proposed in Section 3.2, which takes $\mathcal { W } ^ { c h a n } = ( \mathcal { W } _ { r } ^ { c h a n } +$$j \mathcal { W } _ { i } ^ { c h a n } ) \in \mathbb { C } ^ { d \times d }$ as the complex number weight matrix with $\mathcal { W } _ { r } ^ { c h a n } \in \mathbb { R } ^ { d \times d }$ and $\mathcal { W } _ { i } ^ { c h a n } \in \mathbb { R } ^ { d \times d }$ ,and $B ^ { c h a n } = ( B _ { r } ^ { c h a n } + j B _ { i } ^ { c h a n } ) \in \mathbb { C } ^ { d }$ as the biases with $B _ { r } ^ { c h a n } \in \mathbb { R } ^ { d }$ and $B _ { i } ^ { c h a n } \in \mathbb { R } ^ { d }$ . And$\mathcal { Z } _ { c h a n } ^ { : , ( l ) } \in \mathbb { C } ^ { \frac { N } { 2 } \times d }$ is the output of FreMLP, also in the frequency domain, which is conversed back totime domain as $\mathbf { Z } ^ { : , ( l ) } \in \mathbb { R } ^ { N \times d }$ . Finally, we ensemble $\mathbf { Z } ^ { : , ( l ) }$ of $L$ timestamps into a whole and outputZt ∈ RN×L×d. $\mathbf { Z } _ { t } \in \mathbb { R } ^ { N \times L \times d }$

Frequency Temporal Learner The frequency temporal learner aims to learn the temporal patternsin the frequency domain; also, it is constructed based on frequency-domain MLPs conducting oneach channel and it shares the weights between $N$ channels. Specifically, it takes the frequencychannel learner output $\mathbf { Z } _ { t } \in \mathbb { R } ^ { N \times L \times d }$ as input and for the $n$ -th channel $\mathbf { Z } _ { t } ^ { ( n ) , : } \in \mathbb { R } ^ { L \times d }$ , we apply thefrequency temporal learner by:

$$
\mathcal {Z} _ {\text {t e m p}} ^ {(n);:} = \operatorname {D o m a i n C o n v e r s i o n} _ {\left(\text {t e m p}\right)} \left(\mathbf {Z} _ {t} ^ {(n); :}\right)
$$

$$
\mathcal {S} _ {\text {t e m p}} ^ {(n),:} = \operatorname {F r e M L P} \left(\mathcal {Z} _ {\text {t e m p}} ^ {(n),:}, \mathcal {W} ^ {\text {t e m p}}, \mathcal {B} ^ {\text {t e m p}}\right) \tag {4}
$$

$$
\mathbf {S} ^ {(n);;} = \mathrm {D o m a i n I n v e r s i o n} _ {(t e m p)} (\mathcal {S} _ {t e m p} ^ {(n);;})
$$

where $\mathcal { Z } _ { t e m p } ^ { ( n ) , : } \in \mathbb { C } ^ { \frac { L } { 2 } \times d }$ is the corresponding frequency spectrum of $\mathbf { Z } _ { t } ^ { ( n ) , : }$ ; DomainConversion(temp)and DomainInversion(temp) indicates the calculations are applied along the time dimension.$\mathcal { W } ^ { t e m p } = ( \mathcal { W } _ { r } ^ { t e m p } + j \mathcal { W } _ { i } ^ { t e m p } ) \in \mathbb { C } ^ { d \times d }$ is the complex number weight matrix with $\mathcal { W } _ { r } ^ { t e m p } \in \mathbb { R } ^ { d \times d }$and $\mathcal { W } _ { i } ^ { t e m p } \in \mathbb { R } ^ { d \times d }$ , and $\mathcal { B } ^ { t e m p } = ( \mathcal { B } _ { r } ^ { t e m p } + j \mathcal { B } _ { i } ^ { t e m p } ) \in \mathbb { C } ^ { d }$ are the complex number biases with$B _ { r } ^ { t e m p } \in \mathbb { R } ^ { d }$ and $B _ { i } ^ { t e m p } \in \mathbb { R } ^ { d }$ . $S _ { t e m p } ^ { ( n ) , : } \in \mathbb { C } ^ { \frac { L } { 2 } \times d }$ is the output of FreMLP and is converted back to thetime domain as $\mathbf { S } ^ { ( n ) , : } \in \mathbb { R } ^ { L \times d }$ . Finally, we incorporate all channels and output $\mathbf { S } _ { t } \in \mathbb { R } ^ { N \times L \times d }$ .

Projection Finally, we use the learned channel and temporal dependencies to make predictions forthe future $\tau$ timestamps $\hat { \mathbf Y } _ { t } \in \mathbb { R } ^ { N \times \tau }$ by a two-layer feed forward network (FFN) with one forwardstep which can avoid error accumulation, formulated as follows:

$$
\hat {\mathbf {Y}} _ {t} = \sigma \left(\mathbf {S} _ {t} \phi_ {1} + \mathbf {b} _ {1}\right) \phi_ {2} + \mathbf {b} _ {2} \tag {5}
$$

where $\mathbf { S } _ { t } \in \mathbb { R } ^ { N \times L \times d }$ is the output of the frequency temporal learner, $\sigma$ is the activation function,$\phi _ { 1 } \in \mathbb { R } ^ { ( L * d ) \times d _ { h } } , \phi _ { 2 } \in \mathbb { R } ^ { d _ { h } \times \tau }$ are the weights, $\mathbf { b } _ { 1 } \in \mathbb { R } ^ { d _ { h } }$ , $\mathbf { b } _ { 2 } \in \mathbb { R } ^ { \tau }$ are the biases, and $d _ { h }$ is theinner-layer dimension size.

# 3.2 Frequency-domain MLPs

As shown in Figure 3, we elaborate our novel frequency-domain MLPs in FreTS that are redesignedfor the complex numbers of frequency components, in order to effectively capture the time series keypatterns with global view and energy compaction, as aforementioned in Section 1.

Definition 1 (Frequency-domain MLPs). Formally, for a complex number input $\mathcal { H } \in \mathbb { C } ^ { m \times d }$given a complex number weight matrix $\mathcal { W } \in \mathbb { C } ^ { d \times d }$ and a complex number bias $\bar { B } \in \mathbb { C } ^ { d }$ , then thefrequency-domain MLPs can be formulated as:

$$
\mathcal {Y} ^ {\ell} = \sigma \left(\mathcal {Y} ^ {\ell - 1} \mathcal {W} ^ {\ell} + \mathcal {B} ^ {\ell}\right) \tag {6}
$$

$$
\mathcal {Y} ^ {0} = \mathcal {H}
$$

where $\mathcal { y } ^ { \ell } \in \mathbb { C } ^ { m \times d }$ is the final output, ℓ denotes the ℓ-th layer, and $\sigma$ is the activation function.

As both $\mathcal { H }$ and $\mathcal { W }$ are complex numbers, according to the rule of multiplication of complex numbers(details can be seen in Appendix C), we further extend the Equation (6) to:

$$
\mathcal {Y} ^ {\ell} = \sigma (R e (\mathcal {Y} ^ {\ell - 1}) \mathcal {W} _ {r} ^ {\ell} - I m (\mathcal {Y} ^ {\ell - 1}) \mathcal {W} _ {i} ^ {\ell} + \mathcal {B} _ {r} ^ {\ell}) + j \sigma (R e (\mathcal {Y} ^ {\ell - 1}) \mathcal {W} _ {i} ^ {\ell} + I m (\mathcal {Y} ^ {\ell - 1}) \mathcal {W} _ {r} ^ {\ell} + \mathcal {B} _ {i} ^ {\ell}) \tag {7}
$$

where $\mathscr { W } ^ { \ell } = \mathscr { W } _ { r } ^ { \ell } + j \mathscr { W } _ { i } ^ { \ell }$ and $B ^ { \ell } = B _ { r } ^ { \ell } + j B _ { i } ^ { \ell }$ . According to the equation, we implement the MLPsin the frequency domain (abbreviated as FreMLP) by the separate computation of the real andimaginary parts of frequency components. Then, we stack them to form a complex number to acquirethe final results. The specific implementation process is shown in Figure 3.

Theorem 1. Suppose that H is the representation of raw time series and $\mathcal { H }$ is the correspondingfrequency components of the spectrum, then the energy of a time series in the time domain is equal tothe energy of its representation in the frequency domain. Formally, we can express this with abovenotations by:

$$
\int_ {- \infty} ^ {\infty} | \mathbf {H} (v) | ^ {2} \mathrm {d} v = \int_ {- \infty} ^ {\infty} | \mathcal {H} (f) | ^ {2} \mathrm {d} f \tag {8}
$$

where $\begin{array} { r } { \mathcal { H } ( \boldsymbol { f } ) = \int _ { - \infty } ^ { \infty } \mathbf { H } ( v ) e ^ { - j 2 \pi f v } \mathrm { d } v , } \end{array}$ v is the time/channel dimension, $f$ is the frequency dimension.

We include the proof in Appendix D.1. The theorem impliesthat if most of the energy of a time series is concentrated in asmall number of frequency components, then the time seriescan be accurately represented using only those components.Accordingly, discarding the others would not significantlyaffect the signal’s energy. As shown in Figure 1(b), in thefrequency domain, the energy concentrates on the smallerpart of frequency components, thus learning in the frequencyspectrum can facilitate preserving clearer patterns.

Theorem 2. Given the time series input H and its corre-sponding frequency domain conversion $\mathcal { H }$ , the operations offrequency-domain MLP on H can be represented as globalconvolutions on H in the time domain. This can be given by:

$$
\mathcal {H W} + \mathcal {B} = \mathcal {F} (\mathbf {H} * W + B) \tag {9}
$$

![](images/c90868c2e3467ba2524619d236e510b8923472ff4d752912d99c91eb81727e46.jpg)



Figure 3: One layer of the frequency-domain MLPs.


where $^ *$ is a circular convolution, $\mathcal { W }$ and B are the complex number weight and bias, W and $B$ arethe weight and bias in the time domain, and $\mathcal { F }$ is DFT.

The proof is shown in Appendix D.2. Therefore, the operations of FreMLP, i.e., $\mathcal { H } \mathcal { W } + B$ , areequal to the operations $( \mathbf { H } * W + B )$ in the time domain. This implies that the operations offrequency-domain MLPs can be viewed as global convolutions in the time domain.

# 4 Experiments

To evaluate the performance of FreTS, we conduct extensive experiments on thirteen real-world timeseries benchmarks, covering short-term forecasting and long-term forecasting settings to comparewith corresponding state-of-the-art methods.

Datasets Our empirical results are performed on various domains of datasets, including traffic,energy, web, traffic, electrocardiogram, and healthcare, etc. Specifically, for the task of short-termforecasting, we adopt Solar 2, Wiki [39], Traffic [39], Electricity 3, ECG [18], METR-LA [40],and COVID-19 [5] datasets, following previous forecasting literature [18]. For the task of long-term forecasting, we adopt Weather [16], Exchange [12], Traffic [16], Electricity [16], and ETTdatasets [15], following previous long time series forecasting works [15, 16, 32, 41]. We preprocessall datasets following [18, 15, 16] and normalize them with the min-max normalization. We split thedatasets into training, validation, and test sets by the ratio of 7:2:1 except for the COVID-19 datasetswith 6:2:2. More dataset details are in Appendix B.1.

Baselines We compare our FreTS with the representative and state-of-the-art models for both short-term and long-term forecasting to evaluate their effectiveness. For short-term forecasting, we compreFreTS against VAR [25], SFM [31], LSTNet [12], TCN [13], GraphWaveNet [29], DeepGLO [39],StemGNN [18], MTGNN [17], and AGCRN [19] for comparison. We also include TAMP-S2GCNets[5], DCRNN [40] and STGCN [42], which require pre-defined graph structures, for comparison. Forlong-term forecasting, we include Informer [15], Autoformer [16], Reformer [20], FEDformer [32],LTSF-Linear [37], and the more recent PatchTST [41] for comparison. Additional details about thebaselines can be found in Appendix B.2.

Implementation Details Our model is implemented with Pytorch 1.8 [43], and all experimentsare conducted on a single NVIDIA RTX 3080 10GB GPU. We take MSE (Mean Squared Error) asthe loss function and report MAE (Mean Absolute Errors) and RMSE (Root Mean Squared Errors)results as the evaluation metrics. For additional implementation details, please refer to Appendix B.3.

# 4.1 Main Results


Table 1: Short-term forecasting comparison. The best results are in bold, and the second best resultsare underlined. Full benchmarks of short-term forecasting are in Appendix F.1.


<table><tr><td rowspan="2">Models</td><td colspan="2">Solar</td><td colspan="2">Wiki</td><td colspan="2">Traffic</td><td colspan="2">ECG</td><td colspan="2">Electricity</td><td colspan="2">COVID-19</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>VAR</td><td>0.184</td><td>0.234</td><td>0.052</td><td>0.094</td><td>0.535</td><td>1.133</td><td>0.120</td><td>0.170</td><td>0.101</td><td>0.163</td><td>0.226</td><td>0.326</td></tr><tr><td>SFM</td><td>0.161</td><td>0.283</td><td>0.081</td><td>0.156</td><td>0.029</td><td>0.044</td><td>0.095</td><td>0.135</td><td>0.086</td><td>0.129</td><td>0.205</td><td>0.308</td></tr><tr><td>LSTNet</td><td>0.148</td><td>0.200</td><td>0.054</td><td>0.090</td><td>0.026</td><td>0.057</td><td>0.079</td><td>0.115</td><td>0.075</td><td>0.138</td><td>0.248</td><td>0.305</td></tr><tr><td>TCN</td><td>0.176</td><td>0.222</td><td>0.094</td><td>0.142</td><td>0.052</td><td>0.067</td><td>0.078</td><td>0.107</td><td>0.057</td><td>0.083</td><td>0.317</td><td>0.354</td></tr><tr><td>DeepGLO</td><td>0.178</td><td>0.400</td><td>0.110</td><td>0.113</td><td>0.025</td><td>0.037</td><td>0.110</td><td>0.163</td><td>0.090</td><td>0.131</td><td>0.169</td><td>0.253</td></tr><tr><td>Reformer</td><td>0.234</td><td>0.292</td><td>0.047</td><td>0.083</td><td>0.029</td><td>0.042</td><td>0.062</td><td>0.090</td><td>0.078</td><td>0.129</td><td>0.152</td><td>0.209</td></tr><tr><td>Informer</td><td>0.151</td><td>0.199</td><td>0.051</td><td>0.086</td><td>0.020</td><td>0.033</td><td>0.056</td><td>0.085</td><td>0.074</td><td>0.123</td><td>0.200</td><td>0.259</td></tr><tr><td>Autoformer</td><td>0.150</td><td>0.193</td><td>0.069</td><td>0.103</td><td>0.029</td><td>0.043</td><td>0.055</td><td>0.081</td><td>0.056</td><td>0.083</td><td>0.159</td><td>0.211</td></tr><tr><td>FEDformer</td><td>0.139</td><td>0.182</td><td>0.068</td><td>0.098</td><td>0.025</td><td>0.038</td><td>0.055</td><td>0.080</td><td>0.055</td><td>0.081</td><td>0.160</td><td>0.219</td></tr><tr><td>GraphWaveNet</td><td>0.183</td><td>0.238</td><td>0.061</td><td>0.105</td><td>0.013</td><td>0.034</td><td>0.093</td><td>0.142</td><td>0.094</td><td>0.140</td><td>0.201</td><td>0.255</td></tr><tr><td>StemGNN</td><td>0.176</td><td>0.222</td><td>0.190</td><td>0.255</td><td>0.080</td><td>0.135</td><td>0.100</td><td>0.130</td><td>0.070</td><td>0.101</td><td>0.421</td><td>0.508</td></tr><tr><td>MTGNN</td><td>0.151</td><td>0.207</td><td>0.101</td><td>0.140</td><td>0.013</td><td>0.030</td><td>0.090</td><td>0.139</td><td>0.077</td><td>0.113</td><td>0.394</td><td>0.488</td></tr><tr><td>AGCRN</td><td>0.123</td><td>0.214</td><td>0.044</td><td>0.079</td><td>0.084</td><td>0.166</td><td>0.055</td><td>0.080</td><td>0.074</td><td>0.116</td><td>0.254</td><td>0.309</td></tr><tr><td>FreTS (Ours)</td><td>0.120</td><td>0.162</td><td>0.041</td><td>0.074</td><td>0.011</td><td>0.023</td><td>0.053</td><td>0.078</td><td>0.050</td><td>0.076</td><td>0.123</td><td>0.167</td></tr></table>


Table 2: Long-term forecasting comparison. We set the lookback window size $L$ as 96 and theprediction length as $\tau \in \{ 9 6 , \bar { 1 } 9 2 , 3 \bar { 3 } 6 , 7 2 0 \}$ except for traffic dataset whose prediction length isset as $\tau \in \{ 4 8 , 9 6 , 1 9 2 , 3 3 \bar { 6 } \}$ . The best results are in bold and the second best are underlined. Fullresults of long-term forecasting are included in Appendix F.2.


<table><tr><td rowspan="2" colspan="2">Models Metrics</td><td colspan="2">FreTS</td><td colspan="2">PatchTST</td><td colspan="2">LTSF-Linear</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td><td colspan="2">Informer</td><td colspan="2">Reformer</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.032</td><td>0.071</td><td>0.034</td><td>0.074</td><td>0.040</td><td>0.081</td><td>0.050</td><td>0.088</td><td>0.064</td><td>0.104</td><td>0.101</td><td>0.139</td><td>0.108</td><td>0.152</td></tr><tr><td>192</td><td>0.040</td><td>0.081</td><td>0.042</td><td>0.084</td><td>0.048</td><td>0.089</td><td>0.051</td><td>0.092</td><td>0.061</td><td>0.103</td><td>0.097</td><td>0.134</td><td>0.147</td><td>0.201</td></tr><tr><td>336</td><td>0.046</td><td>0.090</td><td>0.049</td><td>0.094</td><td>0.056</td><td>0.098</td><td>0.057</td><td>0.100</td><td>0.059</td><td>0.101</td><td>0.115</td><td>0.155</td><td>0.154</td><td>0.203</td></tr><tr><td>720</td><td>0.055</td><td>0.099</td><td>0.056</td><td>0.102</td><td>0.065</td><td>0.106</td><td>0.064</td><td>0.109</td><td>0.065</td><td>0.110</td><td>0.132</td><td>0.175</td><td>0.173</td><td>0.228</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>0.037</td><td>0.051</td><td>0.039</td><td>0.052</td><td>0.038</td><td>0.052</td><td>0.050</td><td>0.067</td><td>0.050</td><td>0.066</td><td>0.066</td><td>0.084</td><td>0.126</td><td>0.146</td></tr><tr><td>192</td><td>0.050</td><td>0.067</td><td>0.055</td><td>0.074</td><td>0.053</td><td>0.069</td><td>0.064</td><td>0.082</td><td>0.063</td><td>0.083</td><td>0.068</td><td>0.088</td><td>0.147</td><td>0.169</td></tr><tr><td>336</td><td>0.062</td><td>0.082</td><td>0.071</td><td>0.093</td><td>0.064</td><td>0.085</td><td>0.080</td><td>0.105</td><td>0.075</td><td>0.101</td><td>0.093</td><td>0.127</td><td>0.157</td><td>0.189</td></tr><tr><td>720</td><td>0.088</td><td>0.110</td><td>0.132</td><td>0.166</td><td>0.092</td><td>0.116</td><td>0.151</td><td>0.183</td><td>0.150</td><td>0.181</td><td>0.117</td><td>0.170</td><td>0.166</td><td>0.201</td></tr><tr><td rowspan="4">Traffic</td><td>48</td><td>0.018</td><td>0.036</td><td>0.016</td><td>0.032</td><td>0.020</td><td>0.039</td><td>0.022</td><td>0.036</td><td>0.026</td><td>0.042</td><td>0.023</td><td>0.039</td><td>0.035</td><td>0.053</td></tr><tr><td>96</td><td>0.020</td><td>0.038</td><td>0.018</td><td>0.035</td><td>0.022</td><td>0.042</td><td>0.023</td><td>0.044</td><td>0.033</td><td>0.050</td><td>0.030</td><td>0.047</td><td>0.035</td><td>0.054</td></tr><tr><td>192</td><td>0.019</td><td>0.038</td><td>0.020</td><td>0.039</td><td>0.020</td><td>0.040</td><td>0.022</td><td>0.042</td><td>0.035</td><td>0.053</td><td>0.034</td><td>0.053</td><td>0.035</td><td>0.054</td></tr><tr><td>336</td><td>0.020</td><td>0.039</td><td>0.021</td><td>0.040</td><td>0.021</td><td>0.041</td><td>0.021</td><td>0.040</td><td>0.032</td><td>0.050</td><td>0.035</td><td>0.054</td><td>0.035</td><td>0.055</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.039</td><td>0.065</td><td>0.041</td><td>0.067</td><td>0.045</td><td>0.075</td><td>0.049</td><td>0.072</td><td>0.051</td><td>0.075</td><td>0.094</td><td>0.124</td><td>0.095</td><td>0.125</td></tr><tr><td>192</td><td>0.040</td><td>0.064</td><td>0.042</td><td>0.066</td><td>0.043</td><td>0.070</td><td>0.049</td><td>0.072</td><td>0.072</td><td>0.099</td><td>0.105</td><td>0.138</td><td>0.121</td><td>0.152</td></tr><tr><td>336</td><td>0.046</td><td>0.072</td><td>0.043</td><td>0.067</td><td>0.044</td><td>0.071</td><td>0.051</td><td>0.075</td><td>0.084</td><td>0.115</td><td>0.112</td><td>0.144</td><td>0.122</td><td>0.152</td></tr><tr><td>720</td><td>0.052</td><td>0.079</td><td>0.055</td><td>0.081</td><td>0.054</td><td>0.080</td><td>0.055</td><td>0.077</td><td>0.088</td><td>0.119</td><td>0.116</td><td>0.148</td><td>0.120</td><td>0.151</td></tr><tr><td rowspan="4">ETThl</td><td>96</td><td>0.061</td><td>0.087</td><td>0.065</td><td>0.091</td><td>0.063</td><td>0.089</td><td>0.072</td><td>0.096</td><td>0.079</td><td>0.105</td><td>0.093</td><td>0.121</td><td>0.113</td><td>0.143</td></tr><tr><td>192</td><td>0.065</td><td>0.091</td><td>0.069</td><td>0.094</td><td>0.067</td><td>0.094</td><td>0.076</td><td>0.100</td><td>0.086</td><td>0.114</td><td>0.103</td><td>0.137</td><td>0.120</td><td>0.148</td></tr><tr><td>336</td><td>0.070</td><td>0.096</td><td>0.073</td><td>0.099</td><td>0.070</td><td>0.097</td><td>0.080</td><td>0.105</td><td>0.088</td><td>0.119</td><td>0.112</td><td>0.145</td><td>0.124</td><td>0.155</td></tr><tr><td>720</td><td>0.082</td><td>0.108</td><td>0.087</td><td>0.113</td><td>0.082</td><td>0.108</td><td>0.090</td><td>0.116</td><td>0.102</td><td>0.136</td><td>0.125</td><td>0.157</td><td>0.126</td><td>0.155</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.052</td><td>0.077</td><td>0.055</td><td>0.082</td><td>0.055</td><td>0.080</td><td>0.063</td><td>0.087</td><td>0.081</td><td>0.109</td><td>0.070</td><td>0.096</td><td>0.065</td><td>0.089</td></tr><tr><td>192</td><td>0.057</td><td>0.083</td><td>0.059</td><td>0.085</td><td>0.060</td><td>0.087</td><td>0.068</td><td>0.093</td><td>0.083</td><td>0.112</td><td>0.082</td><td>0.107</td><td>0.081</td><td>0.108</td></tr><tr><td>336</td><td>0.062</td><td>0.089</td><td>0.064</td><td>0.091</td><td>0.065</td><td>0.093</td><td>0.075</td><td>0.102</td><td>0.091</td><td>0.125</td><td>0.090</td><td>0.119</td><td>0.100</td><td>0.128</td></tr><tr><td>720</td><td>0.069</td><td>0.096</td><td>0.070</td><td>0.097</td><td>0.072</td><td>0.099</td><td>0.081</td><td>0.108</td><td>0.093</td><td>0.126</td><td>0.115</td><td>0.149</td><td>0.132</td><td>0.163</td></tr></table>

Short-Term Time Series Forecasting Table 1 presents the forecasting accuracy of our FreTScompared to thirteen baselines on six datasets, with an input length of 12 and a prediction lengthof 12. The best results are highlighted in bold and the second-best results are underlined. From thetable, we observe that FreTS outperforms all baselines on MAE and RMSE across all datasets, andon average it makes improvement of $9 . 4 \%$ on MAE and $1 1 . 6 \%$ on RMSE. We credit this to the factthat FreTS explicitly models both channel and temporal dependencies, and it flexibly unifies channeland temporal modeling in the frequency domain, which can effectively capture the key patterns withthe global view and energy compaction. We further report the complete benchmarks of short-termforecasting under different steps on different datasets (including METR-LA dataset) in Appendix F.1.

Long-term Time Series Forecasting Table 2 showcases the long-term forecasting results of FreTScompared to six representative baselines on six benchmarks with various prediction lengths. Forthe traffic dataset, we select 48 as the lookback window size $L$ with the prediction lengths $\tau \in$$\{ 4 8 , 9 6 , 1 9 2 , 3 3 6 \}$ . For the other datasets, the input lookback window length is set to 96 and theprediction length is set to $\tau \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ . The results demonstrate that FreTS outperformsall baselines on all datasets. Quantitatively, compared with the best results of Transformer-basedmodels, FreTS has an average decrease of more than $20 \%$ in MAE and RMSE. Compared with morerecent LSTF-Linear [37] and the SOTA PathchTST [41], FreTS can still outperform them in general.In addition, we provide further comparison of FreTS and other baselines and report performanceunder different lookback window sizes in Appendix F.2. Combining Tables 1 and 2, we can concludethat FreTS achieves competitive performance in both short-term and long-term forecasting task.

# 4.2 Model Analysis

Frequency Channel and TemporalLearners We analyze the effectsof frequency channel and temporallearners in Table 3 in both short-termand long-term experimental settings.We consider two variants: FreCL:we remove the frequency temporallearner from FreTS, and FreTL: weremove the frequency channel learnerfrom FreTS. From the comparison,we observe that the frequency chan-


Table 3: Ablation studies of frequency channel and temporallearners in both short-term and long-term forecasting. ’I/O’indicates lookback window sizes/prediction lengths.


<table><tr><td>Tasks</td><td colspan="4">Short-term</td><td colspan="4">Long-term</td></tr><tr><td>Dataset I/O</td><td colspan="2">Electricity 12/12</td><td colspan="2">METR-LA 12/12</td><td colspan="2">Exchange 96/336</td><td colspan="2">Weather 96/336</td></tr><tr><td>Metrics</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>FreCL</td><td>0.054</td><td>0.080</td><td>0.086</td><td>0.168</td><td>0.067</td><td>0.086</td><td>0.051</td><td>0.094</td></tr><tr><td>FreTL</td><td>0.058</td><td>0.086</td><td>0.085</td><td>0.167</td><td>0.065</td><td>0.085</td><td>0.047</td><td>0.091</td></tr><tr><td>FreTS</td><td>0.050</td><td>0.076</td><td>0.080</td><td>0.166</td><td>0.062</td><td>0.082</td><td>0.046</td><td>0.090</td></tr></table>

nel learner plays a more important role in short-term forecasting. In long-term forecasting, we notethat the frequency temporal learner is more effective than the frequency channel learner. In AppendixE.1, we also conduct the experiments and report performance on other datasets. Interestingly, we findout the channel learner would lead to the worse performance in some long-term forecasting cases. Apotential explanation is that the channel independent strategy [41] brings more benefit to forecasting.

FreMLP vs. MLP We further study the effectiveness of FreMLP in time series forecasting. Weuse FreMLP to replace the original MLP component in the existing SOTA MLP-based models (i.e.,DLinear and NLinear [37]), and compare their performances with the original DLinear and NLinearunder the same experimental settings. The experimental results are presented in Table 4. From thetable, we easily observe that for any prediction length, the performance of both DLinear and NLinearmodels has been improved after replacing the corresponding MLP component with our FreMLP.Quantitatively, incorporating FreMLP into the DLinear model brings an average improvement of$6 . 4 \%$ in MAE and $1 1 . 4 \%$ in RMSE on the Exchange dataset, and $4 . 9 \%$ in MAE and $3 . 5 \%$ in RMSEon the Weather dataset. A similar improvement has also been achieved on the two datasets with regardto NLinear, according to Table 4. These results confirm the effectiveness of FreMLP compared toMLP again and we include more implementation details and analysis in Appendix B.5.


Table 4: Ablation study on the Exchange and Weather datasets with a lookback window size of 96and the prediction length $\tau \in \{ 9 6 , 1 9 \bar { 2 } , 3 3 6 , 7 2 0 \}$ . DLinear (FreMLP)/NLinear (FreMLP) meansthat we replace the MLPs in DLinear/NLinear with FreMLP. The best results are in bold.


<table><tr><td>Datasets</td><td colspan="8">Exchange</td><td colspan="8">Weather</td></tr><tr><td>Lengths</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td></tr><tr><td>Metrics</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>DLinear</td><td>0.037</td><td>0.051</td><td>0.054</td><td>0.072</td><td>0.071</td><td>0.095</td><td>0.095</td><td>0.119</td><td>0.041</td><td>0.081</td><td>0.047</td><td>0.089</td><td>0.056</td><td>0.098</td><td>0.065</td><td>0.106</td></tr><tr><td>DLinear (FreMLP)</td><td>0.036</td><td>0.049</td><td>0.053</td><td>0.071</td><td>0.063</td><td>0.071</td><td>0.086</td><td>0.101</td><td>0.038</td><td>0.078</td><td>0.045</td><td>0.086</td><td>0.055</td><td>0.097</td><td>0.061</td><td>0.100</td></tr><tr><td>NLinear</td><td>0.037</td><td>0.051</td><td>0.051</td><td>0.069</td><td>0.069</td><td>0.093</td><td>0.115</td><td>0.146</td><td>0.037</td><td>0.081</td><td>0.045</td><td>0.089</td><td>0.052</td><td>0.098</td><td>0.058</td><td>0.106</td></tr><tr><td>NLinear (FreMLP)</td><td>0.036</td><td>0.050</td><td>0.049</td><td>0.067</td><td>0.067</td><td>0.091</td><td>0.109</td><td>0.139</td><td>0.035</td><td>0.076</td><td>0.043</td><td>0.084</td><td>0.050</td><td>0.094</td><td>0.057</td><td>0.103</td></tr></table>

# 4.3 Efficiency Analysis

The complexity of our proposed FreTS is $\mathcal { O } ( N \log N + L \log L )$ . We perform efficiency comparisonswith some state-of-the-art GNN-based methods and Transformer-based models under differentnumbers of variables $N$ and prediction lengths $\tau$ , respectively. On the Wiki dataset, we conductexperiments over $N \in \{ 1 0 0 0 , 2 0 0 0 , 3 0 0 0 , 4 0 0 0 , 5 0 0 0 \}$ under the same lookback window size of 12

![](images/27f310959e0cd237521ba21b42a5017becf5f8bfd30999d98421848852f8879a.jpg)



(a) Parameters (left) and training time (right) underdifferent variable numbers


![](images/41caf33c6fe13f41b9b84227da44eb06132cc97736033c0ac93f5c9917f08f10.jpg)



(b) Parameters (left) and training time (right) underdifferent prediction lengths



Figure 4: Efficiency analysis (model parameters and training time) on the Wiki and Exchange dataset.(a) The efficiency comparison under different number of variables: the number of variables is enlargedfrom 1000 to 5000 with the input window size as 12 and the prediction length as 12 on Wiki dataset.(b) The efficiency comparison under the prediction lengths: we conduct experiments with predictionlengths prolonged from 96 to 480 under the same window size of 96 on the Exchange dataset.


and prediction length of 12, as shown in Figure 4(a). From the figure, we can find that: (1) Theamount of FreTS parameters is agnostic to $N$ . (2) Compared with AGCRN, FreTS incurs an average$30 \%$ reduction of the number of parameters and $20 \%$ reduction of training time. On the Exchangedataset, we conduct experiments on different prediction lengths $\tau \in \{ 9 6 , 1 9 2 , 3 3 6 , 4 8 0 \}$ with thesame input length of 96. The results are shown in Figure 4(b). It demonstrates: (1) Compared withTransformer-based methods (FEDformer [32], Autoformer [16], and Informer [15]), FreTS reducesthe number of parameters by at least 3 times. (2) The training time of FreTS is averagely 3 timesfaster than Informer, 5 times faster than Autoformer, and more than 10 times faster than FEDformer.These show our great potential in real-world deployment.

# 4.4 Visualization Analysis

In Figure 5, we visualize the learnedweights $\mathcal { W }$ in FreMLP on the Trafficdataset with a lookback window size of48 and prediction length of 192. As theweights $\mathcal { W }$ are complex numbers, we pro-vide visualizations of the real part $\mathcal { W } _ { r }$ (pre-sented in (a)) and the imaginary part $\mathcal { W } _ { i }$(presented in (b)) separately. From the fig-ure, we can observe that both the real andimaginary parts play a crucial role in learn-ing process: the weight coefficients of thereal or imaginary part exhibit energy aggre-gation characteristics (clear diagonal pat-

terns) which can facilitate to learn the significant features. In Appendix E.2, we further conduct adetailed analysis on the effects of the real and imaginary parts in different contexts of forecasting, andthe effects of the two parts in the FreMLP. We examine their individual contributions and investigatehow they influence the final performance. Additional visualizations of the weights on differentdatasets with various settings, as well as visualizations of global periodic patterns, can be found inAppendix G.1 and Appendix G.2, respectively.

![](images/aa0a9ebbc818c485f6ce6e37f1cf6590325e2de947f9a6392f5d9223c486bf5b.jpg)



(a) The real part $\mathcal { W } _ { r }$


![](images/f29b3361a3683c5489a5aa8949fd821c367f92eb46d828802775f9166d680ab1.jpg)



(b) The imaginary part $w _ { i }$



Figure 5: Visualizing learned weights of FreMLP onthe Traffic dataset. $\mathcal { W } _ { r }$ represents the real part of $\mathcal { W }$ ,and $\mathcal { W } _ { i }$ represents the imaginary part.


# 5 Conclusion Remarks

In this paper, we explore a novel direction and make a new attempt to apply frequency-domain MLPsfor time series forecasting. We have redesigned MLPs in the frequency domain that can effectivelycapture the underlying patterns of time series with global view and energy compaction. We thenverify this design by a simple yet effective architecture, FreTS, built upon the frequency-domainMLPs for time series forecasting. Our comprehensive empirical experiments on seven benchmarks ofshort-term forecasting and six benchmarks of long-term forecasting have validated the superiorityof our proposed methods. Simple MLPs have several advantages and lay the foundation of moderndeep learning, which have great potential for satisfied performance with high efficiency. We hope thiswork can facilitate more future research of MLPs on time series modeling.

# Acknowledgments and Disclosure of Funding

The work was supported in part by the National Key Research and Development Program of Chinaunder Grant 2020AAA0104903 and 2019YFB1406300, and National Natural Science Foundation ofChina under Grant 62072039 and 62272048.

# References



[1] Edward N Lorenz. Empirical orthogonal functions and statistical weather prediction, volume 1.Massachusetts Institute of Technology, Department of Meteorology Cambridge, 1956.





[2] Yu Zheng, Xiuwen Yi, Ming Li, Ruiyuan Li, Zhangqing Shan, Eric Chang, and Tianrui Li.Forecasting fine-grained air quality based on big data. In KDD, pages 2267–2276, 2015.





[3] Wei Fan, Pengyang Wang, Dongkun Wang, Dongjie Wang, Yuanchun Zhou, and Yanjie Fu.Dish-ts: A general paradigm for alleviating distribution shift in time series forecasting. In AAAI,pages 7522–7529. AAAI Press, 2023.





[4] Hui He, Qi Zhang, Simeng Bai, Kun Yi, and Zhendong Niu. CATN: cross attentive tree-awarenetwork for multivariate time series forecasting. In AAAI, pages 4030–4038. AAAI Press, 2022.





[5] Yuzhou Chen, Ignacio Segovia-Dominguez, Baris Coskunuzer, and Yulia Gel. TAMP-s2GCNets:Coupling time-aware multipersistence knowledge representation with spatio-supra graph con-volutional networks for time-series forecasting. In International Conference on LearningRepresentations, 2022.





[6] Hui He, Qi Zhang, Shoujin Wang, Kun Yi, Zhendong Niu, and Longbing Cao. Learninginformative representation for fairness-aware multivariate time-series forecasting: A group-based perspective. IEEE Transactions on Knowledge and Data Engineering, pages 1–13,2023.





[7] Benjamin F King. Market and industry factors in stock price behavior. the Journal of Business,39(1):139–190, 1966.





[8] Adebiyi A Ariyo, Adewumi O Adewumi, and Charles K Ayo. Stock price prediction using thearima model. In 2014 UKSim-AMSS 16th international conference on computer modelling andsimulation, pages 106–112. IEEE, 2014.





[9] Charles C Holt. Forecasting trends and seasonal by exponentially weighted moving averages.ONR Memorandum, 52(2), 1957.





[10] Peter Whittle. Prediction and regulation by linear least-square methods. English UniversitiesPress, 1963.





[11] David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilisticforecasting with autoregressive recurrent networks. International Journal of Forecasting,36(3):1181–1191, 2020.





[12] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long- and short-termtemporal patterns with deep neural networks. In SIGIR, pages 95–104, 2018.





[13] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolu-tional and recurrent networks for sequence modeling. CoRR, abs/1803.01271, 2018.





[14] Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu.Scinet: time series modeling and forecasting with sample convolution and interaction. Advancesin Neural Information Processing Systems, 35:5816–5828, 2022.





[15] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and WancaiZhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. InAAAI, pages 11106–11115, 2021.





[16] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decompositiontransformers with auto-correlation for long-term series forecasting. In NeurIPS, pages 22419–22430, 2021.





[17] Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, and Chengqi Zhang.Connecting the dots: Multivariate time series forecasting with graph neural networks. In KDD,pages 753–763, 2020.





[18] Defu Cao, Yujing Wang, Juanyong Duan, Ce Zhang, Xia Zhu, Congrui Huang, Yunhai Tong,Bixiong Xu, Jing Bai, Jie Tong, and Qi Zhang. Spectral temporal graph neural network formultivariate time-series forecasting. In NeurIPS, 2020.





[19] Lei Bai, Lina Yao, Can Li, Xianzhi Wang, and Can Wang. Adaptive graph convolutionalrecurrent network for traffic forecasting. In NeurIPS, 2020.





[20] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. InICLR, 2020.





[21] Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-beats: Neural basisexpansion analysis for interpretable time series forecasting. arXiv preprint arXiv:1905.10437,2019.





[22] Tianping Zhang, Yizhuo Zhang, Wei Cao, Jiang Bian, Xiaohan Yi, Shun Zheng, and JianLi. Less is more: Fast multivariate time series forecasting with light sampling-oriented mlpstructures. arXiv preprint arXiv:2207.01186, 2022.





[23] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time seriesforecasting? arXiv preprint arXiv:2205.13504, 2022.





[24] Duraisamy Sundararajan. The discrete Fourier transform: theory, algorithms and applications.World Scientific, 2001.





[25] Mark W. Watson. Vector autoregressions and cointegration. Working Paper Series, Macroeco-nomic Issues, 4, 1993.





[26] Dimitros Asteriou and Stephen G Hall. Arima models and the box–jenkins methodology.Applied Econometrics, 2(2):265–286, 2011.





[27] Bryan Lim and Stefan Zohren. Time-series forecasting with deep learning: a survey. Philosoph-ical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences,379(2194):20200209, feb 2021.





[28] José Torres, Dalil Hadjout, Abderrazak Sebaa, Francisco Martínez-Álvarez, and Alicia Troncoso.Deep learning for time series forecasting: A survey. Big Data, 9, 12 2020.





[29] Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, and Chengqi Zhang. Graph wavenet fordeep spatial-temporal graph modeling. In IJCAI, pages 1907–1913, 2019.





[30] Kun Yi, Qi Zhang, Longbing Cao, Shoujin Wang, Guodong Long, Liang Hu, Hui He, ZhendongNiu, Wei Fan, and Hui Xiong. A survey on deep learning based time series analysis withfrequency transformation. CoRR, abs/2302.02173, 2023.





[31] Liheng Zhang, Charu C. Aggarwal, and Guo-Jun Qi. Stock price prediction via discoveringmulti-frequency trading patterns. In KDD, pages 2141–2149, 2017.





[32] Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer:Frequency enhanced decomposed transformer for long-term series forecasting. In ICML, 2022.





[33] Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, and Steven C. H. Hoi. Cost:Contrastive learning of disentangled seasonal-trend representations for time series forecasting.In ICLR. OpenReview.net, 2022.





[34] Tian Zhou, Ziqing Ma, Xue Wang, Qingsong Wen, Liang Sun, Tao Yao, Wotao Yin, and RongJin. Film: Frequency improved legendre memory model for long-term time series forecasting.2022.





[35] Wei Fan, Shun Zheng, Xiaohan Yi, Wei Cao, Yanjie Fu, Jiang Bian, and Tie-Yan Liu. DEPTS:deep expansion learning for periodic time series forecasting. In ICLR. OpenReview.net, 2022.





[36] Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler, andArtur Dubrawski. N-hits: Neural hierarchical interpolation for time series forecasting. CoRR,abs/2201.12886, 2022.





[37] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time seriesforecasting? 2023.





[38] Tomás Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of wordrepresentations in vector space. In ICLR (Workshop Poster), 2013.





[39] Rajat Sen, Hsiang-Fu Yu, and Inderjit S. Dhillon. Think globally, act locally: A deep neuralnetwork approach to high-dimensional time series forecasting. In NeurIPS, pages 4838–4847,2019.





[40] Yaguang Li, Rose Yu, Cyrus Shahabi, and Yan Liu. Diffusion convolutional recurrent neuralnetwork: Data-driven traffic forecasting. In ICLR (Poster), 2018.





[41] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series isworth 64 words: Long-term forecasting with transformers. In International Conference onLearning Representations, 2023.





[42] Bing Yu, Haoteng Yin, and Zhanxing Zhu. Spatio-temporal graph convolutional networks: Adeep learning framework for traffic forecasting. In IJCAI, pages 3634–3640, 2018.





[43] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, AndreasKöpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy,Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style,high-performance deep learning library. In NeurIPS, pages 8024–8035, 2019.



# A Notations


Table 5: Notation.


<table><tr><td>Xt</td><td>multivariate time series with a lookback window of L
at timestamps t, Xt ∈ RN×L</td></tr><tr><td>Xt</td><td>the multivariate values of N distinct series at timestamp t, Xt ∈ RN</td></tr><tr><td>Yt</td><td>the prediction target with a horizon window of length τ
at timestamps t, Yt ∈ RN×τ</td></tr><tr><td>Ht</td><td>the hidden representation of Xt, Ht ∈ RN×L×d</td></tr><tr><td>Zt</td><td>the output of the frequency channel learner, Zt ∈ RN×L×d</td></tr><tr><td>St</td><td>the output of the frequency temporal learner, St ∈ RN×L×d</td></tr><tr><td>Hchan</td><td>the domain conversion of Ht on channel dimensions, Hchan ∈ CN×L×d</td></tr><tr><td>Zchan</td><td>the FreMLP output of Hchan, Zchan ∈ CN×L×d</td></tr><tr><td>Ztemp</td><td>the domain conversion of Zt on temporal dimensions, Ztemp ∈ CN×L×d</td></tr><tr><td>Stemp</td><td>the FreMLP output of Ztemp, Stemp ∈ CN×L×d</td></tr><tr><td>Wchan</td><td>the complex number weight matrix of FreMLP in the frequency
channel learner, Wchan ∈ CD×d</td></tr><tr><td>Bchan</td><td>the complex number bias of FreMLP in the frequency channel
learner, Bchan ∈ CD</td></tr><tr><td>Wtemp</td><td>the complex number weight matrix of FreMLP in the frequency
temporal learner, Wtemp ∈ CD×d</td></tr><tr><td>Btemp</td><td>the complex number bias of FreMLP in the frequency
temporal learner, Btemp ∈ CD</td></tr></table>

# B Experimental Details

# B.1 Datasets

We adopt thirteen real-world benchmarks in the experiments to evaluate the accuracy of short-termand long-term forecasting. The details of the datasets are as follows:

Solar4: It is about the solar power collected by National Renewable Energy Laboratory. We choosethe power plant data points in Florida as the data set which contains 593 points. The data is collectedfrom 01/01/2006 to 31/12/2016 with the sampling interval of every 1 hour.

Wiki [39]: It contains a number of daily views of different Wikipedia articles and is collected from1/7/2015 to 31/12/2016. It consists of approximately $1 4 5 k$ time series and we randomly choose $5 k$from them as our experimental data set.

Traffic [39]: It contains hourly traffic data from 963 San Francisco freeway car lanes for short-termforecasting settings while it contains 862 car lanes for long-term forecasting. It is collected since01/01/2015 with a sampling interval of every 1 hour.

ECG5: It is about Electrocardiogram(ECG) from the UCR time-series classification archive. Itcontains 140 nodes and each node has a length of 5000.

Electricity6: It contains electricity consumption of 370 clients for short-term forecasting whileit contains electricity consumption of 321 clients for long-term forecasting. It is collected since01/01/2011. The data sampling interval is every 15 minutes.

COVID-19 [5]: It is about COVID-19 hospitalization in the U.S. state of California (CA) from01/02/2020 to 31/12/2020 provided by the Johns Hopkins University with the sampling interval ofevery day.

METR-LA7: It contains traffic information collected from loop detectors in the highway of LosAngeles County. It contains 207 sensors which are from 01/03/2012 to 30/06/2012 and the datasampling interval is every 5 minutes.

Exchange8: It contains the collection of the daily exchange rates of eight foreign countries includingAustralia, British, Canada, Switzerland, China, Japan, New Zealand, and Singapore ranging from1990 to 2016 and the data sampling interval is every 1 day.

Weather9: It collects 21 meteorological indicators, such as humidity and air temperature, from theWeather Station of the Max Planck Biogeochemistry Institute in Germany in 2020. The data samplinginterval is every 10 minutes.

ETT10: It is collected from two different electric transformers labeled with 1 and 2, and each of themcontains 2 different resolutions (15 minutes and 1 hour) denoted with m and h. We use ETTh1 andETTm1 as our long-term forecasting benchmarks.

# B.2 Baselines

We adopt eighteen representative and state-of-the-art baselines for comparison including LSTM-basedmodels, GNN-based models, and Transformer-based models. We introduce these models as follows:

VAR [25]: VAR is a classic linear autoregressive model. We use the Statsmodels library (https://www.statsmodels.org) which is a Python package that provides statistical computations torealize the VAR.

DeepGLO [39]: DeepGLO models the relationships among variables by matrix factorization andemploys a temporal convolution neural network to introduce non-linear relationships. We downloadthe source code from: https://github.com/rajatsen91/deepglo. We use the recommendedconfiguration as our experimental settings for Wiki, Electricity, and Traffic datasets. For the COVID-19 dataset, the vertical and horizontal batch size is set to 64, the rank of the global model is set to 64,the number of channels is set to [32, 32, 32, 1], and the period is set to 7.

LSTNet [12]: LSTNet uses a CNN to capture inter-variable relationships and an RNN to discoverlong-term patterns. We download the source code from: https://github.com/laiguokun/LSTNet. In our experiment, we use the recommended configuration where the number of CNNhidden units is 100, the kernel size of the CNN layers is 4, the dropout is 0.2, the RNN hidden unitsis 100, the number of RNN hidden layers is 1, the learning rate is 0.001 and the optimizer is Adam.

TCN [13]: TCN is a causal convolution model for regression prediction. We download the source codefrom: https://github.com/locuslab/TCN. We utilize the same configuration as the polyphonicmusic task exampled in the open source code where the dropout is 0.25, the kernel size is 5, thenumber of hidden units is 150, the number of levels is 4 and the optimizer is Adam.

Informer [15]: Informer leverages an efficient self-attention mechanism to encode the dependen-cies among variables. We download the source code from: https://github.com/zhouhaoyi/Informer2020. We use the recommended configuration as the experimental settings where thedropout is 0.05, the number of encoder layers is 2, the number of decoder layers is 1, the learningrate is 0.0001, and the optimizer is Adam.

Reformer [20]: Reformer combines the modeling capacity of a Transformer with an architecture thatcan be executed efficiently on long sequences and with small memory use. We download the source



6https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014





7https://github.com/liyaguang/DCRNN





8https://github.com/laiguokun/multivariate-time-series-data





9https://www.bgc-jena.mpg.de/wetter/





10https://github.com/zhouhaoyi/ETDataset



code from: https://github.com/thuml/Autoformer. We use the recommended configurationas the experimental settings.

Autoformer [16]: Autoformer proposes a decomposition architecture by embedding the seriesdecomposition block as an inner operator, which can progressively aggregate the long-term trend partfrom intermediate prediction. We download the source code from: https://github.com/thuml/Autoformer. We use the recommended configuration as the experimental settings.

FEDformer [32]: FEDformer proposes an attention mechanism with low-rank approximation infrequency and a mixture of expert decomposition to control the distribution shifting. We download thesource code from: https://github.com/MAZiqing/FEDformer. We use FEB-f as the FrequencyEnhanced Block and select the random mode with 64 as the experimental mode.

SFM [31]: On the basis of the LSTM model, SFM introduces a series of different frequency compo-nents in the cell states. We download the source code from: https://github.com/z331565360/State-Frequency-Memory-stock-prediction. We follow the recommended configuration asthe experimental settings where the learning rate is 0.01, the frequency dimension is 10, the hiddendimension is 10 and the optimizer is RMSProp.

StemGNN [18]: StemGNN leverages GFT and DFT to capture dependencies among variables inthe frequency domain. We download the source code from: https://github.com/microsoft/StemGNN. We use the recommended configuration of stemGNN as our experiment setting where theoptimizer is RMSProp, the learning rate is 0.0001, the number of stacked layers is 5, and the dropoutrate is 0.5.

MTGNN [17]: MTGNN proposes an effective method to exploit the inherent dependency relation-ships among multiple time series. We download the source code from: https://github.com/nnzhan/MTGNN. Because the experimental datasets have no static features, we set the parameterload_static_feature to false. We construct the graph by the adaptive adjacency matrix and add thegraph convolution layer. Regarding other parameters, we follow the recommended settings.

GraphWaveNet [29]: GraphWaveNet introduces an adaptive dependency matrix learning to cap-ture the hidden spatial dependency. We download the source code from: https://github.com/nnzhan/Graph-WaveNet. Since our datasets have no prior defined graph structures, we use onlyadaptive adjacent matrix. We add a graph convolutional layer and randomly initialize the adjacentmatrix. We adopt the recommended setting as its experimental configuration where the learning rateis 0.001, the dropout is 0.3, the number of epochs is 50, and the optimizer is Adam.

AGCRN [19]: AGCRN proposes a data-adaptive graph generation module for discovering spatialcorrelations from data. We download the source code from: https://github.com/LeiBAI/AGCRN.We follow the recommended settings where the embedding dimension is 10, the learning rate is 0.003,and the optimizer is Adam.

TAMP-S2GCNets [5]: TAMP-S2GCNets explores the utility of MP to enhance knowledge represen-tation mechanisms within the time-aware DL paradigm. We download the source code from: https://www.dropbox.com/sh/n0ajd5l0tdeyb80/AABGn-ejfV1YtRwjf_L0AOsNa?dl $\scriptstyle { \frac { } { } } = 0$ . TAMP-S2GCNets require a pre-defined graph topology and we use the California State topology providedby the source code as input. We adopt the recommended settings as the experimental configurationfor COVID-19.

DCRNN [40]: DCRNN uses bidirectional graph random walk to model spatial dependency andrecurrent neural network to capture the temporal dynamics. We download the source code from:https://github.com/liyaguang/DCRNN. We use the recommended configuration as our experi-mental settings with the batch size is 64, the learning rate is 0.01, the input dimension is 2 and theoptimizer is Adam. DCRNN requires a pre-defined graph structure and we use the adjacency matrixas the pre-defined structure provided by the METR-LA dataset.

STGCN [42]: STGCN integrates graph convolution and gated temporal convolution through spatial-temporal convolutional blocks. We download the source code from: https://github.com/VeritasYin/STGCN_IJCAI-18. We follow the recommended settings as our experimental config-uration where the batch size is 50, the learning rate is 0.001 and the optimizer is Adam. STGCNrequires a pre-defined graph structure and we leverage the adjacency matrix as the pre-definedstructure provided by the METR-LA dataset.

LTSF-Linear [37]: LTSF-Linear proposes a set of embarrassingly simple one-layer linear models to

learn temporal relationships between input and output sequences. We download the source code from:https://github.com/cure-lab/LTSF-Linear. We use it as our long-term forecasting baselineand follow the recommended settings as experimental configuration.

PatchTST [41]: PatchTST proposes an effective design of Transformer-based models for time seriesforecasting tasks by introducing two key components: patching and channel-independent structure.We download the source code from: https://github.com/PatchTST. We use it as our long-termforecasting baseline and adhere to the recommended settings as the experimental configuration.

# B.3 Implementation Details

By default, both the frequency channel and temporal learners contain one layer of FreMLP withthe embedding size $d$ of 128, and the hidden size $d _ { h }$ is set to 256. For short-term forecasting, thebatch size is set to 32 for Solar, METR-LA, ECG, COVID-19, and Electricity datasets. And for Wikiand Traffic datasets, the batch size is set to 4. For the long-term forecasting, except for the lookbackwindow size, we follow most of the experimental settings of LTSF-Linear [37]. The lookback windowsize is set to 96 which is recommended by FEDformer [32] and Autoformer [16]. In AppendixF.2, we also use 192 and 336 as the lookback window size to conduct experiments and the resultsdemonstrate that FreTS outperforms other baselines as well. For the longer prediction lengths (e.g.,336, 720), we use the channel independence strategy and contain only the frequency temporal learnerin our model. For some datasets, we carefully tune the hyperparameters including the batch size andlearning rate on the validation set, and we choose the settings with the best performance. We tune thebatch size over {4, 8, 16, 32}.

# B.4 Visualization Settings

The Visualization Method for Global View. We follow the visualization methods in LTSF-Linear [37] to visualize the weights learned in the time domain on the input (corresponding tothe left side of Figure 1(a)). For the visualization of the weights learned on the frequency spectrum,we first transform the input into the frequency domain and select the real part of the input frequencyspectrum to replace the original input. Then, we learn the weights and visualize them in the samemanner as in the time domain. The right side of Figure 1(a) shows the weights learned on the Trafficdataset with a lookback window of 96 and a prediction length of 96, Figure 9 displays the weightslearned on the Traffic dataset with a lookback window of 72 and a prediction length of 336, andFigure 10 is the weights learned on the Electricity dataset with a lookback window of 96 and aprediction length of 96.

The Visualization Method for Energy Compaction. Since the learned weights $\mathcal { W } = \mathcal { W } _ { r } + j \mathcal { W } _ { i } \in$$\mathbb { C } ^ { d \times d }$ of the frequency-domain MLPs are complex numbers, we visualize the corresponding real part$\mathcal { W } _ { r }$ and imaginary part $\mathcal { W } _ { i }$ , respectively. We normalize them by the calculation of $1 / \operatorname* { m a x } ( \mathcal { W } ) * \mathcal { W }$and visualize the normalization values. The right side of Figure 1(b) is the real part of $\mathcal { W }$ learnedon the Traffic dataset with a lookback window of 48 and a prediction length of 192. To visualizethe corresponding weights learned in the time domain, we replace the frequency spectrum of input$\mathcal { Z } _ { t e m p } \in \dot { \mathbb { C } } ^ { N \times L \times d }$ with the original time domain input $\mathbf { H } _ { t } \in \dot { \mathbb { R } } ^ { N \times L \times d }$ and perform calculations inthe time domain with a weight $W \in \mathbb { R } ^ { d \times d }$ , as depicted in the left side of Figure 1(b).

# B.5 Ablation Experimental Settings

DLinear decomposes a raw data input into a trend component and a seasonal component, and two one-layer linear layers are applied to each component. In the ablation study part, we replace the two linearlayers with two different frequency-domain MLPs (corresponding to DLinear (FreMLP) in Table 4),and compare their accuracy using the same experimental settings recommended in LTSF-Linear [37].NLinear subtracts the input by the last value of the sequence. Then, the input goes through a linearlayer, and the subtracted part is added back before making the final prediction. We replace the linearlayer with a frequency-domain MLP (corresponding to NLinear (FreMLP) in Table 4), and comparetheir accuracy using the same experimental settings recommended in LTSF-Linear [37].

# C Complex Multiplication

For two complex number values $\mathcal { Z } _ { 1 } = ( a + j b )$ and $\mathcal { Z } _ { 2 } = ( c + j d )$ , where $a$ and $c$ is the realpart of $\mathcal { Z } _ { 1 }$ and $\mathcal { Z } _ { 2 }$ respectively, $b$ and $d$ is the imaginary part of $\mathcal { Z } _ { 1 }$ and $\mathcal { Z } _ { 2 }$ respectively. Then themultiplication of $\mathcal { Z } _ { 1 }$ and $\mathcal { Z } _ { 2 }$ is calculated by:

$$
\mathcal {Z} _ {1} \mathcal {Z} _ {2} = (a + j b) (c + j d) = a c + j ^ {2} b d + j a d + j b c = (a c - b d) + j (a d + b c) \tag {10}
$$

where $j ^ { 2 } = - 1$

# D Proof

# D.1 Proof of Theorem 1

Theorem 1. Suppose that H is the representation of raw time series and $\mathcal { H }$ is the correspondingfrequency components of the spectrum, then the energy of a time series in the time domain is equal tothe energy of its representation in the frequency domain. Formally, we can express this with abovenotations by:

$$
\int_ {- \infty} ^ {\infty} | \mathbf {H} (v) | ^ {2} \mathrm {d} v = \int_ {- \infty} ^ {\infty} | \mathcal {H} (f) | ^ {2} \mathrm {d} f \tag {11}
$$

where $\begin{array} { r } { \mathcal { H } ( f ) = \int _ { - \infty } ^ { \infty } \mathbf { H } ( v ) e ^ { - j 2 \pi f v } \mathrm { d } v , } \end{array}$ , v is the time/channel dimension, $f$ is the frequency dimension.

Proof. Given the representation of raw time series $\mathbf { H } \in \mathbb { R } ^ { N \times L \times d }$ , let us consider performingintegration in either the $N$ dimension (channel dimension) or the $L$ dimension (temporal dimension),denoted as the integral over $v$ , then

$$
\int_ {- \infty} ^ {\infty} | \mathbf {H} (v) | ^ {2} \mathrm {d} v = \int_ {- \infty} ^ {\infty} \mathbf {H} (v) \mathbf {H} ^ {*} (v) \mathrm {d} v
$$

where $\mathbf { H } ^ { * } ( v )$ is the conjugate of $\mathbf { H } ( v )$ . According to IDFT, $\begin{array} { r } { \mathbf { H } ^ { * } ( v ) = \int _ { - \infty } ^ { \infty } \mathcal { H } ^ { * } ( f ) e ^ { - j 2 \pi f v } \mathrm { d } f } \end{array}$ , wecan obtain

$$
\begin{array}{l} \int_ {- \infty} ^ {\infty} | \mathbf {H} (v) | ^ {2} \mathrm {d} v = \int_ {- \infty} ^ {\infty} \mathbf {H} (v) [ \int_ {- \infty} ^ {\infty} \mathcal {H} ^ {*} (f) e ^ {- j 2 \pi f v} \mathrm {d} f ] \mathrm {d} v \\ = \int_ {- \infty} ^ {\infty} \mathcal {H} ^ {*} (f) [ \int_ {- \infty} ^ {\infty} \mathbf {H} (v) e ^ {- j 2 \pi f v} \mathrm {d} v ] \mathrm {d} f \\ = \int_ {- \infty} ^ {\infty} \mathcal {H} ^ {*} (f) \mathcal {H} (f) d f \\ = \int_ {- \infty} ^ {\infty} | \mathcal {H} (f) | ^ {2} \mathrm {d} f \\ \end{array}
$$

Proved.

![](images/96348f648d67ccade13e54d680da5894851a5d9975fcc9d771c9e770bdef41af.jpg)


Therefore, the energy of a time series in the time domain is equal to the energy of its representationin the frequency domain.

# D.2 Proof of Theorem 2

Theorem 2. Given the time series input H and its corresponding frequency domain conversion H,the operations of frequency-domain MLP on H can be represented as global convolutions on H inthe time domain. This can be given by:

$$
\mathcal {H W} + \mathcal {B} = \mathcal {F} (\mathbf {H} * W + B) \tag {12}
$$

where ∗ is a circular convolution, W and B are the complex number weight and bias, W and B arethe weight and bias in the time domain, and $\mathcal { F }$ is DFT.

Proof. Suppose that we conduct operations in the $N$ (i.e., channel dimension) or $L$ (i.e., temporaldimension) dimension, then

$$
\mathcal {F} (\mathbf {H} (v) * W (v)) = \int_ {- \infty} ^ {\infty} (\mathbf {H} (v) * W (v)) e ^ {- j 2 \pi f v} \mathrm {d} v
$$

According to convolution theorem, $\begin{array} { r } { \mathbf { H } ( v ) * W ( v ) = \int _ { - \infty } ^ { \infty } ( \mathbf { H } ( \tau ) W ( v - \tau ) ) \mathrm { d } \tau } \end{array}$ , then

$$
\begin{array}{l} \mathcal {F} (\mathbf {H} (v) * W (v)) = \int_ {- \infty} ^ {\infty} \int_ {- \infty} ^ {\infty} (\mathbf {H} (\tau) W (v - \tau)) e ^ {- j 2 \pi f v} \mathrm {d} \tau \mathrm {d} v \\ = \int_ {- \infty} ^ {\infty} \int_ {- \infty} ^ {\infty} W (v - \tau) e ^ {- j 2 \pi f v} \mathrm {d} v \mathbf {H} (\tau) \mathrm {d} \tau \\ \end{array}
$$

Let $x = v - \tau$ , then

$$
\begin{array}{l} \mathcal {F} (\mathbf {H} (v) * W (v)) = \int_ {- \infty} ^ {\infty} \int_ {- \infty} ^ {\infty} W (x) e ^ {- j 2 \pi f (x + \tau)} \mathrm {d} x \mathbf {H} (\tau) \mathrm {d} \tau \\ = \int_ {- \infty} ^ {\infty} \int_ {- \infty} ^ {\infty} W (x) e ^ {- j 2 \pi f x} e ^ {- j 2 \pi f \tau} d x \mathbf {H} (\tau) d \tau \\ = \int_ {- \infty} ^ {\infty} \mathbf {H} (\tau) e ^ {- j 2 \pi f \tau} d \tau \int_ {- \infty} ^ {\infty} W (x) e ^ {- j 2 \pi f x} d x \\ = \mathcal {H} (f) \mathcal {W} (f) \\ \end{array}
$$

Accordingly, $( \mathbf { H } ( v ) * W ( v ) )$ in the time domain is equal to $( \mathscr { H } ( f ) \mathscr { W } ( f ) )$ in the frequency domain.Therefore, the operations of FreMLP $( { \mathcal { H } } { \mathcal { W } } + B )$ in the channel (i.e., $v = N ,$ ) or temporal dimension(i.e., $v = L$ ), are equal to the operations $( \mathbf { H } * W + B )$ in the time domain. This implies thatfrequency-domain MLPs can be viewed as global convolutions in the time domain. Proved. □

# E Further Analysis

# E.1 Ablation Study

In this section, we further analyze the effects of the frequency channel and temporal learners withdifferent prediction lengths on ETTm1 and ETTh1 datasets. The results are shown in Table 6. Itdemonstrates that with the prediction length increasing, the frequency temporal learner shows moreeffective than the channel learner. Especially, when the prediction length is longer (e.g., 336, 720),the channel learner will lead to worse performance. The reason is that when the prediction lengthsbecome longer, the model with the channel learner is likely to overfit data during training. Thus forlong-term forecasting with longer prediction lengths, the channel independence strategy may be moreeffective, as described in PatchTST [41].


Table 6: Ablation studies of the frequency channel and temporal learners in long-term forecasting.’I/O’ indicates lookback window sizes/prediction lengths.


<table><tr><td>Dataset</td><td colspan="8">ETTm1</td><td colspan="8">ETTh1</td></tr><tr><td>I/O</td><td colspan="2">96/96</td><td colspan="2">96/192</td><td colspan="2">96/336</td><td colspan="2">96/720</td><td colspan="2">96/96</td><td colspan="2">96/192</td><td colspan="2">96/336</td><td colspan="2">96/720</td></tr><tr><td>Metrics</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>FreCL</td><td>0.053</td><td>0.078</td><td>0.059</td><td>0.085</td><td>0.067</td><td>0.095</td><td>0.097</td><td>0.125</td><td>0.063</td><td>0.089</td><td>0.067</td><td>0.093</td><td>0.071</td><td>0.097</td><td>0.087</td><td>0.115</td></tr><tr><td>FreTL</td><td>0.053</td><td>0.078</td><td>0.058</td><td>0.084</td><td>0.062</td><td>0.089</td><td>0.069</td><td>0.096</td><td>0.061</td><td>0.087</td><td>0.065</td><td>0.091</td><td>0.070</td><td>0.096</td><td>0.082</td><td>0.108</td></tr><tr><td>FreTS</td><td>0.052</td><td>0.077</td><td>0.057</td><td>0.083</td><td>0.064</td><td>0.092</td><td>0.071</td><td>0.099</td><td>0.063</td><td>0.089</td><td>0.066</td><td>0.092</td><td>0.072</td><td>0.098</td><td>0.086</td><td>0.113</td></tr></table>

# E.2 Impacts of Real/Imaginary Parts

To investigate the effects of real and imaginary parts, we conduct experiments on Exchange andETTh1 datasets under different prediction lengths $L \in \{ 9 6 , 1 9 2 \}$ with the lookback window of 96.Furthermore, we analyze the effects of $\mathcal { W } _ { r }$ and $\mathcal { W } _ { i }$ in the weights $\mathscr { W } = \mathscr { W } _ { r } + j \mathscr { W } _ { i }$ of FreMLP. Inthis experiment, we only use the frequency temporal learner in our model. The results are shown in

Table 7. In the table, $\mathrm { I n p u t } _ { r e a l }$ indicates that we only feed the real part of the input into the network,and $\operatorname { I n p u t } _ { i m a g }$ indicates that we only feed the imaginary part of the input into the network. $\mathcal { W } ( \mathcal { W } _ { r } )$denotes that we set $\mathcal { W } _ { i }$ to 0 and $\mathcal { W } ( \mathcal { W } _ { i } )$ denotes that we set $\mathcal { W } _ { r }$ to 0. From the table, we can observethat both the real part and imaginary part of input are indispensable and the real part is more importantto the imaginary part, and the real part of $\mathcal { W }$ plays a more significant role for the model performances.


Table 7: Investigation the impacts of real/imaginary parts


<table><tr><td>Dataset</td><td colspan="4">Exchange</td><td colspan="4">ETTh1</td></tr><tr><td rowspan="2">I/O Metrics</td><td colspan="2">96/96</td><td colspan="2">96/192</td><td colspan="2">96/96</td><td colspan="2">96/192</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>Inputreal</td><td>0.048</td><td>0.062</td><td>0.058</td><td>0.074</td><td>0.080</td><td>0.111</td><td>0.083</td><td>0.113</td></tr><tr><td>Inputimag</td><td>0.143</td><td>0.185</td><td>0.143</td><td>0.184</td><td>0.130</td><td>0.156</td><td>0.130</td><td>0.156</td></tr><tr><td>W(Wr)</td><td>0.039</td><td>0.053</td><td>0.051</td><td>0.067</td><td>0.063</td><td>0.089</td><td>0.067</td><td>0.093</td></tr><tr><td>W(Wi)</td><td>0.143</td><td>0.184</td><td>0.142</td><td>0.184</td><td>0.116</td><td>0.138</td><td>0.117</td><td>0.139</td></tr><tr><td>FreTS</td><td>0.037</td><td>0.051</td><td>0.050</td><td>0.067</td><td>0.061</td><td>0.087</td><td>0.065</td><td>0.091</td></tr></table>

# E.3 Parameter Sensitivity

We further perform extensive experiments on the ECG dataset to evaluate the sensitivity of the inputlength $L$ and the embedding dimension size $d$ . (1) Input length: We tune over the input length with thevalue $\{ 6 , 1 2 , 1 8 , 2 4 , 3 0 , 3 6 , 4 2 , 5 0 , 6 0 \}$ on the ECG dataset and the prediction length is 12, and theresult is shown in Figure 6(a). From the figure, we can find that with the input length increasing, theperformance first becomes better because the long input length may contain more pattern information,and then it decreases due to data redundancy or overfitting. (2) Embedding size: We choose theembedding size over the set $\{ 3 2 , 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ on the ECG dataset. The results are shown inFigure 6(b). It shows that the performance first increases and then decreases with the increase of theembedding size because a large embedding size improves the fitting ability of our FreTS but mayeasily lead to overfitting especially when the embedding size is too large.

![](images/7fdf6960b7fa011f138f525563c80da56f55da201a88b7c9983dc23f73a276ee.jpg)



(a) Input window length


![](images/4d29970c0a8acd02e6f694ebac0d75246e6f421744cb07dac67e3347eb9d784c.jpg)



(b) Embedding size



Figure 6: The parameter sensitivity analyses of FreTS.


# F Additional Results

# F.1 Multi-Step Forecasting

To further evaluate the performance of our FreTS in multi-step forecasting, we conduct moreexperiments on METR-LA and COVID-19 datasets with the input length of 12 and the predictionlengths of {3, 6, 9, 12}, and the results are shown in Tables 8 and 9, respectively. In this experiment,we only select the state-of-the-art (i.e., GNN-based and Transformer-based) models as the baselinessince they perform better than other models, such as RNN and TCN. Among these baselines, STGCN,DCRNN, and TAMP-S2GCNets require pre-defined graph structures. The results demonstrate that

FreTS outperforms other baselines, including those models with pre-defined graph structures, atall steps. This further confirms that FreTS has strong capabilities in capturing channel-wise andtime-wise dependencies.


Table 8: Multi-step short-term forecasting results comparison on the METR-LA dataset with theinput length of 12 and the prediction length of $\tau \in \{ 3 , 6 , 9 , 1 2 \}$ . We highlight the best results in boldand the second best results are underline.


<table><tr><td rowspan="2">Length Metrics</td><td colspan="2">3</td><td colspan="2">6</td><td colspan="2">9</td><td colspan="2">12</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>Reformer</td><td>0.086</td><td>0.154</td><td>0.097</td><td>0.176</td><td>0.107</td><td>0.193</td><td>0.118</td><td>0.206</td></tr><tr><td>Informer</td><td>0.082</td><td>0.156</td><td>0.094</td><td>0.176</td><td>0.108</td><td>0.193</td><td>0.125</td><td>0.214</td></tr><tr><td>Autoformer</td><td>0.087</td><td>0.149</td><td>0.091</td><td>0.162</td><td>0.106</td><td>0.178</td><td>0.099</td><td>0.184</td></tr><tr><td>FEDformer</td><td>0.064</td><td>0.127</td><td>0.073</td><td>0.145</td><td>0.079</td><td>0.160</td><td>0.086</td><td>0.175</td></tr><tr><td>DCRNN</td><td>0.160</td><td>0.204</td><td>0.191</td><td>0.243</td><td>0.216</td><td>0.269</td><td>0.241</td><td>0.291</td></tr><tr><td>STGCN</td><td>0.058</td><td>0.133</td><td>0.080</td><td>0.177</td><td>0.102</td><td>0.209</td><td>0.128</td><td>0.238</td></tr><tr><td>GraphWaveNet</td><td>0.180</td><td>0.366</td><td>0.184</td><td>0.375</td><td>0.196</td><td>0.382</td><td>0.202</td><td>0.386</td></tr><tr><td>MTGNN</td><td>0.135</td><td>0.294</td><td>0.144</td><td>0.307</td><td>0.149</td><td>0.328</td><td>0.153</td><td>0.316</td></tr><tr><td>StemGNN</td><td>0.052</td><td>0.115</td><td>0.069</td><td>0.141</td><td>0.080</td><td>0.162</td><td>0.093</td><td>0.175</td></tr><tr><td>AGCRN</td><td>0.062</td><td>0.131</td><td>0.086</td><td>0.165</td><td>0.099</td><td>0.188</td><td>0.109</td><td>0.204</td></tr><tr><td>FreTS</td><td>0.050</td><td>0.113</td><td>0.066</td><td>0.140</td><td>0.076</td><td>0.158</td><td>0.080</td><td>0.166</td></tr></table>


Table 9: Multi-step short-term forecasting results comparison on the COVID-19 dataset with theinput length of 12 and the prediction length of $\tau \in \{ 3 , 6 , \mathbf { \bar { 9 } } , 1 2 \}$ . We highlight the best results in boldand the second best results are underline.


<table><tr><td rowspan="2">Length Metrics</td><td colspan="2">3</td><td colspan="2">6</td><td colspan="2">9</td><td colspan="2">12</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>Reformer</td><td>0.212</td><td>0.282</td><td>0.139</td><td>0.186</td><td>0.148</td><td>0.197</td><td>0.152</td><td>0.209</td></tr><tr><td>Informer</td><td>0.234</td><td>0.312</td><td>0.190</td><td>0.245</td><td>0.184</td><td>0.242</td><td>0.200</td><td>0.259</td></tr><tr><td>Autoformer</td><td>0.212</td><td>0.280</td><td>0.144</td><td>0.191</td><td>0.152</td><td>0.201</td><td>0.159</td><td>0.211</td></tr><tr><td>FEDformer</td><td>0.246</td><td>0.328</td><td>0.169</td><td>0.242</td><td>0.175</td><td>0.247</td><td>0.160</td><td>0.219</td></tr><tr><td>GraphWaveNet</td><td>0.092</td><td>0.129</td><td>0.133</td><td>0.179</td><td>0.171</td><td>0.225</td><td>0.201</td><td>0.255</td></tr><tr><td>StemGNN</td><td>0.247</td><td>0.318</td><td>0.344</td><td>0.429</td><td>0.359</td><td>0.442</td><td>0.421</td><td>0.508</td></tr><tr><td>AGCRN</td><td>0.130</td><td>0.172</td><td>0.171</td><td>0.218</td><td>0.224</td><td>0.277</td><td>0.254</td><td>0.309</td></tr><tr><td>MTGNN</td><td>0.276</td><td>0.379</td><td>0.446</td><td>0.513</td><td>0.484</td><td>0.548</td><td>0.394</td><td>0.488</td></tr><tr><td>TAMP-S2GCNets</td><td>0.140</td><td>0.190</td><td>0.150</td><td>0.200</td><td>0.170</td><td>0.230</td><td>0.180</td><td>0.230</td></tr><tr><td>FreTS</td><td>0.071</td><td>0.103</td><td>0.093</td><td>0.131</td><td>0.109</td><td>0.148</td><td>0.124</td><td>0.164</td></tr></table>

# F.2 Long-Term Forecasting under Varying Lookback Window

In Table 10, we present the long-term forecasting results of our FreTS and other baselines(PatchTST [41], LTSF-linear [37], FEDformer [32], Autoformer [16], Informer [15], and Re-former [20]) under different lookback window lengths $L \in \{ 9 6 , 1 9 2 , 3 3 6 \}$ on the Exchange dataset.The prediction lengths are $\{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ . From the table, we can observe that our FreTSoutperforms all baselines in all settings and achieves significant improvements than FEDformer [32],Autoformer [16], Informer [15], and Reformer [20]. It verifies the effectiveness of our FreTS inlearning informative representation under different lookback window.

# G Visualizations

# G.1 Weight Visualizations for Energy Compaction

We further visualize the weights $\mathscr { W } = \mathscr { W } _ { r } + j \mathscr { W } _ { i }$ in the frequency temporal learner under differentsettings, including different lookback window sizes and prediction lengths, on the Traffic andElectricity datasets. The results are illustrated in Figures 7 and 8. These figures demonstrate that


Table 10: Long-term forecasting results comparison with different lookback window lengths $L \in$$\{ 9 6 , 1 9 2 , 3 3 6 \}$ . The prediction lengths are as $\tau \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ . The best results are in boldand the second best results are underlined.


<table><tr><td rowspan="2" colspan="2">Models Metrics</td><td colspan="2">FreTS</td><td colspan="2">PatchTST</td><td colspan="2">LTSF-Linear</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td><td colspan="2">Informer</td><td colspan="2">Reformer</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td rowspan="4">96</td><td>96</td><td>0.037</td><td>0.051</td><td>0.039</td><td>0.052</td><td>0.038</td><td>0.052</td><td>0.050</td><td>0.067</td><td>0.050</td><td>0.066</td><td>0.066</td><td>0.084</td><td>0.126</td><td>0.146</td></tr><tr><td>192</td><td>0.050</td><td>0.067</td><td>0.055</td><td>0.074</td><td>0.053</td><td>0.069</td><td>0.064</td><td>0.082</td><td>0.063</td><td>0.083</td><td>0.068</td><td>0.088</td><td>0.147</td><td>0.169</td></tr><tr><td>336</td><td>0.062</td><td>0.082</td><td>0.071</td><td>0.093</td><td>0.064</td><td>0.085</td><td>0.080</td><td>0.105</td><td>0.075</td><td>0.101</td><td>0.093</td><td>0.127</td><td>0.157</td><td>0.189</td></tr><tr><td>720</td><td>0.088</td><td>0.110</td><td>0.132</td><td>0.166</td><td>0.092</td><td>0.116</td><td>0.151</td><td>0.183</td><td>0.150</td><td>0.181</td><td>0.117</td><td>0.170</td><td>0.166</td><td>0.201</td></tr><tr><td rowspan="4">192</td><td>96</td><td>0.036</td><td>0.050</td><td>0.037</td><td>0.051</td><td>0.038</td><td>0.051</td><td>0.067</td><td>0.086</td><td>0.066</td><td>0.085</td><td>0.109</td><td>0.131</td><td>0.123</td><td>0.143</td></tr><tr><td>192</td><td>0.051</td><td>0.068</td><td>0.052</td><td>0.070</td><td>0.053</td><td>0.070</td><td>0.080</td><td>0.101</td><td>0.080</td><td>0.102</td><td>0.144</td><td>0.172</td><td>0.139</td><td>0.161</td></tr><tr><td>336</td><td>0.066</td><td>0.087</td><td>0.072</td><td>0.097</td><td>0.073</td><td>0.096</td><td>0.093</td><td>0.122</td><td>0.099</td><td>0.129</td><td>0.141</td><td>0.177</td><td>0.155</td><td>0.181</td></tr><tr><td>720</td><td>0.088</td><td>0.110</td><td>0.099</td><td>0.128</td><td>0.098</td><td>0.122</td><td>0.190</td><td>0.222</td><td>0.191</td><td>0.224</td><td>0.173</td><td>0.210</td><td>0.159</td><td>0.193</td></tr><tr><td rowspan="4">336</td><td>96</td><td>0.038</td><td>0.052</td><td>0.039</td><td>0.053</td><td>0.040</td><td>0.055</td><td>0.088</td><td>0.113</td><td>0.088</td><td>0.110</td><td>0.137</td><td>0.169</td><td>0.128</td><td>0.148</td></tr><tr><td>192</td><td>0.053</td><td>0.070</td><td>0.055</td><td>0.071</td><td>0.055</td><td>0.072</td><td>0.103</td><td>0.133</td><td>0.104</td><td>0.133</td><td>0.161</td><td>0.195</td><td>0.138</td><td>0.159</td></tr><tr><td>336</td><td>0.071</td><td>0.092</td><td>0.074</td><td>0.099</td><td>0.077</td><td>0.100</td><td>0.123</td><td>0.155</td><td>0.127</td><td>0.159</td><td>0.156</td><td>0.193</td><td>0.156</td><td>0.179</td></tr><tr><td>720</td><td>0.082</td><td>0.108</td><td>0.100</td><td>0.129</td><td>0.087</td><td>0.110</td><td>0.210</td><td>0.242</td><td>0.211</td><td>0.244</td><td>0.173</td><td>0.210</td><td>0.168</td><td>0.205</td></tr></table>

the weight coefficients of the real or imaginary part exhibit energy aggregation characteristics (cleardiagonal patterns) which can facilitate frequency-domain MLPs in learning the significant features.

![](images/77b7f069a92987d51ab2d096e647727427839d58ec5a4c8c25bfcfca2c19aaba.jpg)



(a) $\mathcal { W } _ { r }$ under I/O=48/192


![](images/1187a1324bd2df3a218f3f89614bbd5f9ea87ee80d12f7d07bba86df048793f3.jpg)



(b) $\mathcal { W } _ { r }$ under I/O=48/336


![](images/94abd5a3c0b39fa772a7c01779fe3041b22ebeef7f97d698d90066d3cfd8d64f.jpg)



(c) $\mathcal { W } _ { r }$ under I/O=72/336


![](images/9b09f9d4264afb6c3ad54ef6ee1c7f63125c4b38d69e46259dfbf1a86d871592.jpg)



(d) $\mathcal { W } _ { i }$ under I/O=48/192


![](images/911d41e32d409a89adb37e8d124925bdc3c5094ad445af1ba26d1bc8ba3a1891.jpg)



(e) $\mathcal { W } _ { i }$ under I/O=48/336


![](images/af22d296d67bee0b30a762a659a94f7e3e1b0235488f62b7e464f9d7736f64e2.jpg)



(f) $w _ { i }$ under I/O=72/336



Figure 7: The visualizations of the weights $\mathcal { W }$ in the frequency temporal learner on the Traffic dataset.’I/O’ denotes lookback window sizes/prediction lengths. $\mathcal { W } _ { r }$ and $\mathcal { W } _ { i }$ are the real and imaginary partsof $\mathcal { W }$ , respectively.


# G.2 Weight Visualizations for Global View

To verify the characteristics of a global view of learning in the frequency domain, we performadditional experiments on the Traffic and Electricity datasets and compare the weights learned onthe input in the time domain with those learned on the input frequency spectrum. The results arepresented in Figures 9 and 10. The left side of the figures displays the weights learned on the inputin the time domain, while the right side shows those learned on the real part of the input frequencyspectrum. From the figures, we can observe that the patterns learned on the input frequency spectrumexhibit more obvious periodic patterns compared to the time domain. This is attributed to the globalview characteristics of the frequency domain. Furthermore, we visualize the predictions of FreTS on

![](images/cbe223ff57af21d35ead94d37df3a895d130cc6a86d569cb20f526edc0d5f5b0.jpg)



(a) $\mathcal { W } _ { r }$ under $1 / \mathrm { O } { = } 9 6 / 9 6$


![](images/d7d21a76dcc694f0265142d6012c936c4674fb25f897654fb37f1577f4361c89.jpg)



(b) $\mathcal { W } _ { r }$ under I/O=96/336


![](images/d203da652fc4433c272d4d6db46af615516ae7556f3f591fa421ed79b153392c.jpg)



(c) $\mathcal { W } _ { r }$ under I/O=96/720


![](images/adc3c1d0a8ffd5bc3c3d3ba6d7cd93e6e83ae018a42ccb797cad668b2043b4a0.jpg)



(d) $\ w _ { \nu _ { i } }$ under I/O=96/96


![](images/9588ad9e5310af4906c8e8397233800df80634457dc98599dfefe3074cba32ad.jpg)



(e) $\ w _ { \nu _ { i } }$ under I/O=96/336


![](images/917fd9140155103802c39cda78ba16b8b91567609434244200d73b39003d59d1.jpg)



(f) $\mathcal { W } _ { i }$ under I/O=96/720



Figure 8: The visualizations of the weights $\mathcal { W }$ in the frequency temporal learner on the Electricitydataset. ’I/O’ denotes lookback window sizes/prediction lengths. $\mathcal { W } _ { r }$ and $\mathcal { W } _ { i }$ are the real andimaginary parts of $\mathcal { W }$ , respectively.


the Traffic and Electricity datasets, as depicted in Figures 11 and 12, which show that FreTS exhibita good ability to fit cyclic patterns. In summary, these results demonstrate that FreTS has a strongcapability to capture the global periodic patterns, which benefits from the global view characteristicsof the frequency domain.

![](images/74eddfe094e4fbd9154b89df98b8936262589b74087e28b44002cd72d1cb0924.jpg)



(a) Learned on the input


![](images/338ab81e2fe8419e8db3502291b7f3a58ed62a4beab96c59ce0d8ea06c4ac1a5.jpg)



(b) Learned on the frequency spectrum



Figure 9: Visualization of the weights $( L \times \tau )$ on the Traffic dataset with lookback window size of72 and prediction length of 336.


![](images/10a1203354dec7a0ca55b866de296f075df293f95821587274af52d118c24f79.jpg)



(a) Learned on the input


![](images/5afd0d52d9d4dff0c75cc027b0ddbbb973460f10b712572259bbb9f8040f62cb.jpg)



(b) Learned on the frequency spectrum



Figure 10: Visualization of the weights $( L \times \tau )$ on the Electricity dataset with lookback window sizeof 96 and prediction length of 96.


![](images/69d6f5748c988cb11194b18aad4660433216372028e07c637876a2abd7bae413.jpg)



(a) I/O=48/48


![](images/9f589cbdeebdec17fe440a7335e2bf08843f87cc595e1a918d6a28a5f68d9ee9.jpg)



(b) I/O=48/96


![](images/0e147a412186ccba46409c6e168d02ad19c8c4333cbc65c4fb73007065eb00f8.jpg)



(c) I/O=48/192


![](images/26edba8caaf1fca77862455b353e9e5a269ddf8c2e45e38bbd8d9716c1b0437e.jpg)



(d) I/O=48/336



Figure 11: Visualizations of predictions (forecast vs. actual) on the Traffic dataset. ’I/O’ denoteslookback window sizes/prediction lengths.


![](images/83dfb73c43e1e6e70efaadaeb90b8e0723dfcb37c2dc9fe03b6ff6fb54aefde1.jpg)



(a) I/O=96/96


![](images/26708fb05b0d2cc1ccf4ae4a882c7980ea8e40b0c73bbf34777705a39692fa8e.jpg)



(b) I/O=96/192


![](images/9e0c31a34c3fdbd2e2be54cf983f1b613513762c1f7b76a3f8f9b3026f6649e6.jpg)



(c) I/O=96/336


![](images/c6f279e04c10bc24bbee5d6a205129ead567e7592eb5c2862275b11c032a2e2a.jpg)



(d) I/O=96/720



Figure 12: Visualizations of predictions (forecast vs. actual) on the Electricity dataset. ’I/O’ denoteslookback window sizes/prediction lengths.
