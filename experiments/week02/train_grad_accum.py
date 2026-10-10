import os
import sys
import time
import math
from pathlib import Path
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent.parent
data_path = repo_root / "data" / "raw" / "tinystories.txt"
ckpt_dir = repo_root / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).resolve().parent.parent / "week01"))
from minigpt import MiniGPT

# ==============================================================
# 实验 W02-03：梯度累积单变量对比 (在 BF16 基础上保持有效 Batch 不变)
# 对比 W02-02 (直接 Batch=16, 累积=1, 显存 4606.6MB)
# 本实验设为: 单次微批大小=8, 累积步数=2 -> 有效批大小严格等于 16！
# ==============================================================
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 运行设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

n_layer = 14          # 14 层
n_embd = 768          # 隐藏维度 768
n_head = 12           # 注意力头数 12
block_size = 256      # 上下文窗口 256

micro_batch_size = 8  # 单次放入显存的小批大小 (从 16 缩减为 8)
grad_accum_steps = 2  # 累积步数 2
effective_batch_size = micro_batch_size * grad_accum_steps # 8 * 2 = 16 (与 W02-02 严格一致！)

learning_rate = 3e-4  # 学习率保持 3e-4
max_iters = 200       # 保持 200 次参数更新步数
eval_interval = 50    # 每 50 步评测一次
eval_iters = 10

print(f"配置核对: 单次批大小={micro_batch_size}, 累积次数={grad_accum_steps}, 有效批大小={effective_batch_size}")

# 1. 加载同一份数据与词表
with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, b_size):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (b_size,))
    x = torch.stack([d[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, micro_batch_size)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# 2. 初始化模型与优化器
model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=0.1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 3. 开始带梯度累积的训练
print(f"\n=== 开始 100M 模型梯度累积实验 (共 {max_iters} 次参数更新) ===")
start_train_time = time.time()
step_times = []

for iter in range(max_iters + 1):
    step_start = time.time()

    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        avg_step_ms = (sum(step_times[-50:]) / len(step_times[-50:]) * 1000) if step_times else 0
        print(f"step {iter:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | 显存: {peak_mem:.1f} MB | 单次更新耗时: {avg_step_ms:.1f} ms")

    if iter == max_iters:
        break

    # 【核心实现】：在单次参数更新前清空梯度
    optimizer.zero_grad(set_to_none=True)

    # 循环累积微批次
    for micro_step in range(grad_accum_steps):
        xb, yb = get_batch("train", micro_batch_size)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
            # 关键：损失除以累积步数，保证累加后的梯度大小与正常平均梯度完全等价！
            loss = loss / grad_accum_steps
        loss.backward()

    # 真正更新参数
    optimizer.step()

    step_times.append(time.time() - step_start)

total_time = time.time() - start_train_time
total_tokens = max_iters * effective_batch_size * block_size
tokens_per_sec = total_tokens / total_time
peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

print(f"\n=== W02-03 (100M 梯度累积对比) 训练完成 ===")
print(f"总耗时: {total_time:.2f} 秒 (W02-02无累积: 21.42 秒)")
print(f"峰值显存: {peak_mem:.1f} MB (W02-02无累积: 4606.6 MB)")
print(f"平均吞吐速率: {tokens_per_sec:,.0f} tokens/秒 (W02-02无累积: 38,237 tokens/秒)")

# 4. 保存 Checkpoint
ckpt_path = ckpt_dir / "week02_grad_accum.pt"
torch.save({
    "step": max_iters,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "vocab_size": vocab_size,
    "block_size": block_size,
}, ckpt_path)
print(f"权重已成功保存至: {ckpt_path}")
