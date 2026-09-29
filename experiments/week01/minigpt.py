import torch
import torch.nn as nn

# ==========================================
# 实验超参数配置 (约 10M 参数量级基准配置)
# ==========================================
vocab_size = 65     # 字符表大小 (Tiny Shakespeare)
n_embd = 384        # 隐藏维度 C (Embedding 向量长度)
block_size = 128    # 最大上下文序列长度 T (Context Window)
batch_size = 4      # 批次大小 B

class InputEmbedding(nn.Module):
    """
    Mini GPT 输入层：
    负责将输入的字符 Token 编号序列转为连续向量，并叠加位置编码。
    """
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        # 1. 词嵌入：把每个字符的编号 (0~64) 映射为 384 维向量
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 2. 位置嵌入：把每个位置 (0~127) 映射为 384 维向量
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        # idx 形状为: (B, T)，里面装的是整数编号
        B, T = idx.shape
        device = idx.device

        # 计算词嵌入，形状变成: (B, T, C)
        tok_emb = self.token_embedding_table(idx)

        # 生成位置编号 [0, 1, ..., T-1]，并计算位置嵌入，形状为: (T, C)
        pos = torch.arange(0, T, device=device)
        pos_emb = self.position_embedding_table(pos)

        # 词嵌入与位置嵌入直接相加 (广播机制: (B, T, C) + (T, C) -> (B, T, C))
        x = tok_out = tok_emb + pos_emb
        return x

if __name__ == "__main__":
    print("=== 测试 Mini GPT 输入层 ===")
    model_input = InputEmbedding(vocab_size, n_embd, block_size)

    # 模拟输入一个测试批次: 4 个样本，每个样本取 8 个字符 (B=4, T=8)
    # 里面的数字模拟 Token 编号 (范围 0~64)
    dummy_idx = torch.randint(0, vocab_size, (batch_size, 8))
    print(f"1. 输入原始 Token 张量形状: {dummy_idx.shape}  -> 代表 (Batch={batch_size}, Time/长度=8)")

    # 前向计算
    out = model_input(dummy_idx)
    print(f"2. 经过输入层后的张量形状: {out.shape} -> 代表 (Batch={out.shape[0]}, Time/长度={out.shape[1]}, Channel/维度={out.shape[2]})")

    # 验证是否符合预期
    assert out.shape == (batch_size, 8, n_embd), "形状不符合预期！"
    print("3. 输入层验证通过！成功将二维整数编号升维为三维向量特征空间。")
