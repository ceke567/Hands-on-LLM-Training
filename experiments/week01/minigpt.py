import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 实验超参数配置 (基准 10M 配置参数)
# ==========================================
vocab_size = 65     # 字符表大小
n_embd = 384        # 隐藏维度 C
n_head = 6          # 注意力头数
block_size = 128    # 最大上下文序列长度 T
batch_size = 4      # 批大小 B

class InputEmbedding(nn.Module):
    """Mini GPT 输入层：Token 嵌入 + 位置嵌入"""
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(0, T, device=device)
        pos_emb = self.position_embedding_table(pos)
        return tok_emb + pos_emb

class CausalSelfAttention(nn.Module):
    """
    Mini GPT 因果多头自注意力层 (Causal Multi-Head Self-Attention)：
    包含 QKV 投影、缩放点积注意力、因果防偷看遮罩、以及输出投影。
    """
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除！"
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head

        # 一次性计算 Q, K, V (从 384 维线性投影到 3 * 384 维)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # 输出投影层 (把所有头的输出整合回 384 维)
        self.c_proj = nn.Linear(n_embd, n_embd)

        # 因果遮罩：构建下三角矩阵，未来位置为 0
        # register_buffer 表示它是模型的一部分，但不是可训练的权重参数
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape

        # 1. 投影得到 Q, K, V
        q, k, v = self.c_attn(x).chunk(3, dim=-1)

        # 2. 变换为多头维度: (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 3. 计算注意力分数: (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))

        # 4. 因果遮罩：把未来位置 (bias == 0) 填充为 -inf
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # 5. Softmax 归一化: -inf 经过 softmax 严格变成 0.0
        att_weights = F.softmax(att, dim=-1)

        # 6. 加权求和 Value，并把多头拼接还原回 (B, T, C)
        y = att_weights @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 7. 输出线性投影
        out = self.c_proj(y)
        return out, att_weights

if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== 测试 Mini GPT 输入层与因果自注意力层 ===")

    # 1. 输入层
    embed_layer = InputEmbedding(vocab_size, n_embd, block_size)
    dummy_idx = torch.randint(0, vocab_size, (batch_size, 4)) # 取 T=4 方便观察
    x = embed_layer(dummy_idx)
    print(f"1. 输入张量形状: {x.shape} (B=4, T=4, C=384)")

    # 2. 因果注意力层
    attn_layer = CausalSelfAttention(n_embd, n_head, block_size)
    out, att_weights = attn_layer(x)
    print(f"2. 注意力层输出形状: {out.shape} (依然完美保持 B=4, T=4, C=384)")

    # 3. 检查因果遮罩是否生效 (观察第 1 个样本第 1 个头的注意力矩阵)
    print("\n3. 因果注意力权重矩阵 (4x4 观察窗口):")
    sample_weights = att_weights[0, 0].detach()
    for row_idx, row in enumerate(sample_weights):
        row_str = " ".join([f"{val:.4f}" for val in row])
        print(f"   位置 {row_idx}: [{row_str}]")

    # 验证右上角 (未来位置) 是否严格为 0
    assert torch.all(sample_weights.triu(diagonal=1) == 0.0), "因果遮罩未生效，存在偷看未来字符！"
    print("\n4. 验证通过！右上角（未来字符）权重严格为 0.0000，模型无法偷看下文！")
