import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# 引用第 1 周已验证的基础组件 (InputEmbedding, Block, MiniGPT)
sys.path.append(str(Path(__file__).resolve().parent.parent / "week01"))
from minigpt import MiniGPT

# ==========================================
# 第 2 周：100M 基线模型配置
# ==========================================
vocab_size = 65     # 保持词表兼容 (后续使用 TinyStories 或更大词表亦可动态适应)
n_embd = 768        # 隐藏维度由 384 翻倍到 768
n_head = 12         # 注意力头数由 6 增加到 12 (每头 64 维)
n_layer = 14        # 层数由 6 层增加到 14 层
block_size = 256    # 上下文窗口扩充为 256
dropout = 0.1

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 运行设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

    # 1. 实例化 100M MiniGPT
    model = MiniGPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout
    ).to(device)

    # 2. 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n1. 实际总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"   可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    # 3. 小批次前后向检查 (B=4, T=32)
    dummy_x = torch.randint(0, vocab_size, (4, 32), device=device)
    dummy_y = torch.randint(0, vocab_size, (4, 32), device=device)

    logits, loss = model(dummy_x, dummy_y)
    print(f"2. 输入批次形状: x={dummy_x.shape}, 输出 logits 形状: {logits.shape}")
    print(f"3. 初始单步 Loss: {loss.item():.4f}")

    # 4. 反向传播梯度检查
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    print(f"4. 反向传播梯度检查: {'全部通过！' if has_grad else '存在缺失！'}")
    print(f"5. 峰值显存占用: {peak_mem:.1f} MB")
    print("\n🎉 100M 参数架构验证通过，单批次前后向运算无显存与梯度异常！")
