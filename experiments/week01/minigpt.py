import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 实验超参数配置 (基准 10M 参数量级配置)
# ==========================================
vocab_size = 65     # 字符表大小
n_embd = 384        # 隐藏维度 C
n_head = 6          # 注意力头数
n_layer = 6         # Transformer Block 堆叠层数
block_size = 128    # 最大上下文序列长度 T
dropout = 0.1       # 防止过拟合的 Dropout 比例

class InputEmbedding(nn.Module):
    """输入层：Token 嵌入 + 位置嵌入"""
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos = torch.arange(0, T, device=device)   # (T,)
        pos_emb = self.position_embedding_table(pos) # (T, C)
        return tok_emb + pos_emb

class CausalSelfAttention(nn.Module):
    """因果多头自注意力层"""
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    """前馈网络 (MLP)：4 倍维度扩展与 GELU 激活"""
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """标准 Transformer Block：Pre-LayerNorm 架构 + 残差连接"""
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x):
        # 残差连接: x = x + sublayer(norm(x))
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MiniGPT(nn.Module):
    """
    完整的 Mini GPT (Decoder-only Transformer)
    """
    def __init__(self, vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size, dropout=dropout):
        super().__init__()
        self.block_size = block_size
        self.embedding = InputEmbedding(vocab_size, n_embd, block_size)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embedding(idx)           # (B, T, C)
        x = self.blocks(x)                # (B, T, C)
        x = self.ln_f(x)                  # (B, T, C)
        logits = self.lm_head(x)          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # 展平为 (B*T, vocab_size) 与 (B*T,) 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """简单的自回归文本生成"""
        for _ in range(max_new_tokens):
            # 如果上下文超出 block_size，裁剪最近的 block_size 个
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # 只要最后一个时间步的预测打分: (B, vocab_size)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # 依据概率采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)
            # 拼接到已生成的序列后面
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 运行设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

    # 1. 实例化完整 MiniGPT
    model = MiniGPT().to(device)

    # 2. 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n1. 模型总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 3. 构造一个小批次做前后向测试 (B=4, T=8)
    dummy_x = torch.randint(0, vocab_size, (4, 8), device=device)
    dummy_y = torch.randint(0, vocab_size, (4, 8), device=device)

    print(f"\n2. 输入批次形状: x={dummy_x.shape}, 目标形状: y={dummy_y.shape}")

    # 4. 前向传播
    logits, loss = model(dummy_x, dummy_y)
    print(f"3. 输出 Logits 形状: {logits.shape} -> 代表 (B=4, T=8, 词表数={vocab_size})")
    print(f"4. 初始未训练 Loss: {loss.item():.4f} (理论随机猜想期望值 ≈ {-math.log(1/vocab_size):.4f})")

    # 5. 反向传播演练
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"5. 反向传播测试: {'全部参数梯度计算正常！' if has_grad else '部分参数缺失梯度！'}")

    print("\n🎉 Mini GPT 完整架构与 GPU 前后向计算全部核验通过！")
