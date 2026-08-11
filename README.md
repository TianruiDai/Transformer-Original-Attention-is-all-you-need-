# 现代 Transformer / LLM 实现笔记

本仓库基于 PyTorch 从零实现了一个完整的 Encoder-Decoder Transformer，并在原始结构上引入了若干现代 LLM 中常用的改进（RMSNorm、safesoftmax、scale 缩放等）。

---

## 目录结构

```
LLM/
├── transformer/
│   ├── ults/                      # 基础算子层 (utilities / sub-layers)
│   │   ├── ult.py                 # safesoftmax、RMS_Layernorm
│   │   ├── ScaledDotProductAttention.py
│   │   ├── MultiheadAttention.py  # MHA 实现
│   │   ├── TwoLayers.py            # 两层 FFN
│   │   └── padding.py             # padding mask / subsequent mask
│   ├── layers/                    # Transformer 单元层
│   │   ├── Encoder.py             # EncoderLayer
│   │   ├── Decoder.py             # DecoderLayer (self-attn + cross-attn)
│   │   ├── PE_Layer.py            # Position Embedding (正余弦绝对位置编码)
│   │   └── Constant.py            # 常量 (10000、特殊 token)
│   └── Models/                    # 顶层模型
│       ├── EncoderDecoder.py      # Encoder / Decoder 堆叠
│       └── Transformer.py         # 完整 Transformer
├── train.py                       # 训练入口
├── translate.py                   # 推理 / 翻译入口
├── preprocess.py / learn_bpe.py / apply_bpe.py
└── requirements.txt
```

---

## 理论 ↔ 代码 对照

### 1. MHA / MQA / GQA

**理论背景**：
1. 传统 Multihead Attention 考虑对每个 head 维度（低维子空间）的 Query，都构造一个自己的 K 和 V，这种技术被称为 MHA。
2. MQA 指的是 Multi-Q Attention，即对每个 head 维度（低维子空间）的 Query，共用一套 KV。
3. GQA 指的是对子空间们进行分组，比如说四个一组 Q₁,…,Q₄，然后每四个共用一套 KV。

核心目的是为了降低 KV-cache 的大小（推理阶段的 Decode stage 是直接拿 KV cache 过来使用）。

| 概念 | 说明 | 本项目实现 |
| --- | --- | --- |
| MHA | 每个 head 拥有独立的 K、V | `Multihead_Attention`（标准多头注意力） |
| MQA | 所有 head 共享一套 K、V | 未实现（可由 `w_k`/`w_v` 输出维度改为 `d_k`/`d_v` 而非 `num_head*d_k` 得到） |
| GQA | 多个 head 分组共享 K、V | 未实现 |

本项目对应代码 `transformer/ults/MultiheadAttention.py`：

```7:24:transformer/ults/MultiheadAttention.py
class Multihead_Attention(nn.Module):

    '''多头注意力层'''
    
    def __init__(self, num_head, model_d, d_k, d_v, dropout = 0.1):
        super().__init__()

        self.num_head = num_head
        self.model_d = model_d
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = nn.Dropout(p = dropout)
        self.attention = Scaled_Dot_Product_Attention(scaling = math.sqrt(d_k))
        self.layer_norm = RMS_Layernorm(eps = 1e-5, d = model_d)
        self.w_q = nn.Linear(model_d, num_head*d_k, bias =False)
        self.w_k = nn.Linear(model_d, num_head*d_k, bias =False)
        self.w_v = nn.Linear(model_d, num_head*d_v, bias =False)
        self.fc = nn.Linear(num_head*d_v, model_d, bias = False)
```

`w_k` / `w_v` 的输出维度为 `num_head * d_k`，每个 head 拥有独立子空间，即标准 MHA。若要扩展为 MQA/GQA，只需将这两个 `Linear` 的输出维度改为 `d_k` / `d_v`（MQA）或 `n_group * d_k`（GQA），并在 `forward` 中通过 `expand` / `repeat` 将 K、V 复制到对应 head 维度上即可。

---

### 2. Layer-norm / Batch-norm / RMS-norm

**理论背景**：
1. Layer-norm 指的是对同一 sample 的不同特征进行归一化的操作，即减去均值，除以方差，然后做对角可学习放缩（获得不同特征的表达强度）。
2. Batch-norm 指的是同一特征对不同 sample 做归一化操作，但是在语言模型中，如果出现没见过的长句子会失效。
3. RMS-norm 指的是在特征层面进行不去中心化的归一化操作，然后再做对角可学习放缩，主要目的是防止梯度消失或是梯度爆炸的核心是做尺度归一化，而不是去中心化。

| 归一化方式 | 操作对象 | 是否去中心化 | 本项目实现 |
| --- | --- | --- | --- |
| Layer-norm | 同一 sample 的不同特征 | 是 | — |
| Batch-norm | 同一特征的不同 sample | 是 | — |
| RMS-norm | 同一 sample 的不同特征 | 否，仅做尺度归一化 | `RMS_Layernorm` |

本项目对应代码 `transformer/ults/ult.py`：

```14:27:transformer/ults/ult.py
class RMS_Layernorm(nn.Module):

    ''' RMS_Layernorm 不去中心化, 增加可学习逐feature放缩参数gamma'''

    def __init__(self, eps, d):
        super().__init__()
        self.d = d
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self,x):
        x_sqsum = torch.sqrt(torch.sum(x.pow(2), dim = -1, keepdim = True)/self.d + self.eps)
        x = x/x_sqsum
        return x*self.gamma
```

实现要点：
- 只对最后一维（特征维）做 **RMS 归一化**：`x / sqrt(mean(x²) + eps)`，不减均值；
- 保留可学习的逐特征放缩参数 `gamma`，没有偏置项 `beta`；
- 这与原始 LayerNorm 的核心差异正是：**防止梯度消失/爆炸的核心是尺度归一化，而非去中心化**。

`RMS_Layernorm` 在本项目中被广泛复用：`Multihead_Attention`、`Two_Layers`、`Encoder`、`Decoder` 都使用它作为归一化层。

---

### 3. Pre-norm 与 Post-norm

**理论背景**：
1. Post-norm 指的是 `x_{l+1} = Norm(x_l + layer(x_l))`。
2. Pre-norm 指的是 `x_{l+1} = x_l + layer(Norm(x_l))`。

核心区别是什么时候作用 norm 操作。Pre-norm 更能得到 `x_{l+1}` 与 `x_l` 的递推关系，更符合原始 ResNet 学习残差的核心思想。

| 形式 | 公式 | 本项目实现 |
| --- | --- | --- |
| Post-norm | `x_{l+1} = Norm(x_l + layer(x_l))` | ✅ |
| Pre-norm | `x_{l+1} = x_l + layer(Norm(x_l))` | ❌ |

本项目采用 **Post-norm**，对应 `Multihead_Attention.forward` 中的最后两步：

```47:49:transformer/ults/MultiheadAttention.py
        q = q + res

        q = self.layer_norm(q)
```

`Two_Layers.forward` 中也是同样的"残差 → 归一化"顺序：

```23:25:transformer/ults/TwoLayers.py
        x = x + res

        x = self.layernorm(x)
```

> 备注：现代大模型（GPT-2 之后、LLaMA 等）普遍采用 Pre-norm 以获得更稳定的深层训练。本项目保留 Post-norm 是为了忠实复现原始 Transformer。如需切换为 Pre-norm，只需把 `layer_norm` 移到 `forward` 中残差相加之前，并对 `res` 直接相加。

---

### 4. 位置感知

**理论背景**：
由于 Transformer 是一次性看完所有句子，然后计算每个 token 之间的 attention，故而和 RNN 不同的是，Transformer 需要额外的位置编码用于感知不同 token 之间的位置关系。

1. 原始 Multihead、BERT 等是采用 Position Embedding 的方式，即对每个 token 的位置 `i` 进行编码，得到第 `i` 个 token 的位置嵌入向量 `x_i`，其维度和词嵌入的维度一致。这样直接做相加，再进入 multihead block 里。
2. 现代 LLM 采用子空间旋转的方式对位置进行嵌入，即将词嵌入的维度两两分组，得到若干子空间，然后第 `i` 个 token 有一个相应的旋转角，和子空间的编号以及 `i` 相关，然后对这个二维子空间做旋转操作。本质上就是对 Q、K、V 中做分块，然后再作用大量旋转矩阵构成的矩阵。这样位置的差异或者不同位置的 token 做 attention 可以直接被反馈在旋转角的差值。这种技术被称为 RoPE。

本项目实现的是方案 1 —— 绝对正余弦位置编码，对应 `transformer/layers/PE_Layer.py`：

```14:25:transformer/layers/PE_Layer.py
    def _get_pos_encoding_table(self, num_token, d_model):

        def get_position_angle_vec(position):
            
            return [position / np.power( Constant, 2 * (emb_i // 2) / d_model ) for emb_i in range(d_model)]

        posemb_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_token)])

        posemb_table[:,0::2] = np.sin(posemb_table[:,0::2])
        posemb_table[:,1::2] = np.cos(posemb_table[:,1::2])

        return torch.from_numpy(posemb_table).float().unsqueeze(0)
```

其中 `Constant = 10000`（见 `transformer/layers/Constant.py`）。偶数维用 `sin`、奇数维用 `cos`，是经典的 "Attention is All You Need" 形式。`forward` 中通过 `x + self.pos_table[:, :x.size(1)]` 将位置编码直接加到词嵌入上。

> 扩展提示：若要实现 RoPE，可在此处新增一个 `RoPE` 模块，在 `Multihead_Attention.forward` 中对 `q`、`k`（已 reshape 为 `[B, H, L, d_k]`）按最后一维两两分组做旋转，再送入 `Scaled_Dot_Product_Attention`。

---

### 5. Attention 改造

**理论背景**：
对于 Transformer 来说，计算最大的问题是长句子 attention 计算。Transformer 的计算量实际为 `O(2n²d + 12nd²)`，故而对长句子来说，`n²` 占主导。

现代 LLM 修改方式：

1. **Flash Attention**：IO-Aware exact attention。它的核心思想不是减少计算量，而是将 Q、K 进行分块，然后利用 safesoftmax 进行更新，最终目的是减少 Q、K、V 从 HBM 中提取的数据交换时间，改为从 SRAM 中提取。
2. **Sparse Attention**：考虑 M 为稀疏矩阵，`i-j` 想算 attention 时 `M_{ij}=0`，否则为一个很大的负数。这样用 `logits + M` 再做 softmax 得到 attention score。其中一个做法是 window sliding，即每个词只算它的上下文 attention。
3. **Linear Attention**：寻找某个 kernel feature map，使得 `sim(q,k) = <φ(q),φ(k)>`。这样再利用结合律先算 `φ(k)v`，得到计算量实际上可为 `O(nd²)`。然后再不断更新，这其实本质上说明自回归 Transformer 在找到核函数以后本质上是 RNN。

| 技术 | 思想 | 本项目实现 |
| --- | --- | --- |
| Flash Attention | IO-aware 分块精确 attention | ❌（依赖底层 kernel） |
| Sparse Attention | 稀疏 mask，如 window sliding | ❌ |
| Linear Attention | 用 kernel feature map 把 `sim(q,k)` 写成 `<φ(q),φ(k)>`，先算 `φ(k)v` | ❌ |

本项目实现的是**标准 scaled dot-product attention**，对应 `transformer/ults/ScaledDotProductAttention.py`：

```17:27:transformer/ults/ScaledDotProductAttention.py
    def forward(self,q,k,v,mask=None):

        attn = torch.matmul(q / self.scaling, k.transpose(2,3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.attn_dropout(safesoftmax(attn,dim=-1))
        output = attn @ v

        return output, attn
```

其中两点值得注意：

- **safesoftmax**：用 `x - max(x)` 减去最大值再做 `exp / sum`，避免大 `qk^T` 导致数值溢出，是现代实现的标准做法：

```6:11:transformer/ults/ult.py
def safesoftmax(x,dim = -1): # safesoftmax 用于替代传统softmax
    x_max = torch.max(x,dim=dim,keepdim=True)[0]
    x_exp = torch.exp(x-x_max)
    x_safesoftmax = x_exp / torch.sum(x_exp,dim = dim, keepdim = True)
    
    return x_safesoftmax
```

- **mask 处理**：通过 `masked_fill(mask == 0, -1e9)` 将不应参与 attention 的位置压成极小值，再过 softmax 即得到 0 权重。这对应"稀疏 attention"思想的最朴素形式 —— padding mask 与 subsequent mask 都是稀疏 mask 的特例。

mask 的构造在 `transformer/ults/padding.py`：

```1:14:transformer/ults/padding.py
def get_pad_mask(seq,pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):

    batch_size, len_seq = seq.size()

    subsequent_mask =(1 - torch.triu(
        torch.ones((1, len_seq, len_seq), device = seq.device), diagonal = 1).bool()
    )

    return subsequent_mask.bool()
```

`get_pad_mask` 屏蔽 padding token；`get_subsequent_mask` 生成下三角 causal mask，用于 decoder 防止看到未来 token。二者在 `Transformer.forward` 中按位与组合：

```68:69:transformer/Models/Transformer.py
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
```

---

### 6. 长上下文外推手段

**理论背景**：

1. **直接外推**：inferring > training；代表：vanilla RoPE、AliBi 显示外推。
   - 核心问题：OOD (Out of Distribution) position pattern。

2. **固定放缩的位置嵌入外推方法**：核心思想是对长文本位置进行固定比例压缩（位置嵌入）。
   - 核心问题：局部信息丢失，高频维度被压为低频。

3. **带频率感知的位置嵌入外推方法**：核心思想是对高频不做压缩，只对低频做压缩，同时保持局部感知能力和全局感知能力。代表作：NTK-Aware、YaRN、Dynamic NTK、LongRoPE。

4. **修改 Attention distance**：核心思想是修改 distance difference，对于局部保持不变，对于全局做压缩。

本项目目前**未实现**任何外推策略，位置编码固定为 `num_token=200` 长度的查表（见 `PE_Layer.py` 中 `register_buffer('pos_table', ...)`）。若推理长度超过 `num_token`，会因为 `pos_table` 切片越界而失败，这正是"直接外推 + OOD position pattern"问题的最朴素体现。

可作为后续扩展点：
- 在 `Position_Embedding` 中加入对超长位置的动态插值（NTK-Aware 风格）；
- 或将 `Position_Embedding` 替换为 RoPE，天然支持任意长度。

---

## 顶层模型组装

### EncoderLayer / DecoderLayer

`EncoderLayer` = MHA + Two_Layers FFN，对应 `transformer/layers/Encoder.py`：

```20:24:transformer/layers/Encoder.py
    def forward(self,enc_input,enc_mask = None):
        enc_output,enc_attn = self.attention(enc_input,enc_input,enc_input,enc_mask)
        enc_output = self.twolayer(enc_output)

        return enc_output, enc_attn
```

`DecoderLayer` = self-attention + cross-attention + FFN，对应 `transformer/layers/Decoder.py`：

```23:26:transformer/layers/Decoder.py
    def forward(self,enc_output,dec_input,dec_enc_mask,dec_self_mask = None):
        dec_output,dec_self_attn = self.attention(dec_input,dec_input,dec_input,dec_self_mask)
        dec_output,dec_enc_attn = self.cross_attention(dec_output,enc_output,enc_output,dec_enc_mask )
        dec_output = self.twolayer(dec_output)

        return dec_output, dec_self_attn, dec_enc_attn
```

### Encoder / Decoder 堆叠

`transformer/Models/EncoderDecoder.py` 中 `Encoder` / `Decoder` 将 `n_layer` 个 `EncoderLayer` / `DecoderLayer` 串成 `nn.ModuleList`，并在进入堆叠前依次做：

```
word_emb → (可选 scale_emb) → position_emb → dropout → RMS_Layernorm
```

### Transformer

`transformer/Models/Transformer.py` 把 Encoder、Decoder 与输出投影 `trg_word_prj` 拼起来，并支持三种权重共享模式（`scale_emb_or_prj ∈ {'emb', 'prj', 'none'}`）：

- `trg_emb_prj_weight_sharing`：decoder embedding 与输出投影共享权重；
- `emb_src_trg_weight_sharing`：encoder 与 decoder 的 embedding 共享权重。

forward 流程：

```66:78:transformer/Models/Transformer.py
    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq,src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)

        if self.scale_prj:
            seq_logit *= self.d_model ** (-0.5)

        return seq_logit.view(-1,seq_logit.size(2))
```

---

## 快速开始

### 环境依赖

见 `requirements.txt`，核心依赖：

- `python==3.6.12`
- `pytorch==1.3.1`
- `torchtext`、`spacy`（用于数据预处理）
- `tqdm`、`tensorboard`（训练可视化）

### 训练

```bash
# 1. 预处理 + 学习 BPE
python preprocess.py
python learn_bpe.py
python apply_bpe.py

# 2. 训练（示例脚本）
bash train_multi30k_de_en.sh
# 或直接
python train.py
```

### 翻译 / 推理

```bash
python translate.py
```

---

## 与现代 LLM 的差距与扩展路线

本项目可作为"教学版 Transformer"，要演进到现代 LLM 还可补齐以下模块：

| 方向 | 当前状态 | 建议改造点 |
| --- | --- | --- |
| 注意力分组 | MHA | 在 `Multihead_Attention` 中支持 `n_kv_head` 参数实现 GQA |
| 归一化 | RMSNorm ✅ | 已对齐现代实现 |
| Norm 位置 | Post-norm | 切换为 Pre-norm 以支持更深堆叠 |
| 位置编码 | 绝对正余弦 | 替换为 RoPE，支持长上下文外推 |
| 长序列 Attention | 标准 SDP | 引入 Flash Attention（PyTorch 2.0+ `F.scaled_dot_product_attention`）或 sliding window mask |
| 长上下文外推 | 固定 200 token | 引入 NTK-Aware / YaRN 缩放策略 |
| 激活函数 | ReLU（见 `Two_Layers`） | 替换为 SwiGLU / GeGLU |
| 词表 | BPE | 已支持（`learn_bpe.py` / `apply_bpe.py`） |

---

## 参考

- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021.
- Zhang & Sennrich. *Root Mean Square Layer Normalization.* NeurIPS 2019.
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* 2023.
- Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
