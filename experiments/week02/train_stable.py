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
# 实验 W02-04：学习率调度 (Warmup + Cosine Decay) 与梯度裁剪
# 保持 100M 架构与 BF16 精度，加入工业级训练稳定性保障
# ==============================================================
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 运行设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

n_layer = 14          # 14 层
n_embd = 768          # 隐藏维度 768
n_head = 12           # 注意力头数 12
block_size = 256      # 上下文窗口 256
batch_size = 16       # 批大小 16
max_iters = 200       # 200 步
eval_interval = 50    # 每 50 步评测一次
eval_iters = 10

# 学习率调度配置
max_lr = 3e-4         # 峰值学习率
min_lr = 3e-5         # 最低学习率 (衰减终点，设为 10% 峰值)
warmup_iters = 20     # 线性预热步数 (前 10% 步数逐步爬坡)
lr_decay_iters = 200  # 余弦衰减总步数

# 梯度裁剪阈值 (安全气囊)
max_grad_norm = 1.0

def get_lr(it):
    """根据当前步数计算 Warmup + Cosine Decay 学习率"""
    # 1. 线性预热阶段: 0 -> max_lr
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    # 2. 超出衰减周期: 维持最低学习率
    if it > lr_decay_iters:
        return min_lr
    # 3. 中间余弦衰减阶段: max_lr -> min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

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

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
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
            X, Y = get_batch(split)
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

optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr)

# 3. 开始带调度与裁剪的稳定性训练
print(f"\n=== 开始 W02-04 稳定性训练 (Warmup={warmup_iters}步, CosineDecay, ClipNorm={max_grad_norm}) ===")
start_train_time = time.time()
step_times = []
last_grad_norm = 0.0

for iter in range(max_iters + 1):
    step_start = time.time()

    # 动态更新当前步数的学习率
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        avg_step_ms = (sum(step_times[-50:]) / len(step_times[-50:]) * 1000) if step_times else 0
        print(f"step {iter:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e} | grad_norm {last_grad_norm:.2f} | 显存: {peak_mem:.1f} MB | 单步耗时: {avg_step_ms:.1f} ms")

    if iter == max_iters:
        break

    xb, yb = get_batch("train")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 【核心安全机制】：梯度裁剪并记录裁剪前的范数
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    last_grad_norm = norm.item() if hasattr(norm, "item") else float(norm)

    optimizer.step()

    step_times.append(time.time() - step_start)

total_time = time.time() - start_train_time
total_tokens = max_iters * batch_size * block_size
tokens_per_sec = total_tokens / total_time
peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

print(f"\n=== W02-04 (学习率调度与梯度稳定性) 训练完成 ===")
print(f"总耗时: {total_time:.2f} 秒")
print(f"峰值显存: {peak_mem:.1f} MB")
print(f"平均吞吐速率: {tokens_per_sec:,.0f} tokens/秒")

# 4. 保存 Checkpoint (带当前 lr 和 step 信息)
ckpt_path = ckpt_dir / "week02_stable.pt"
torch.save({
    "step": max_iters,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "vocab_size": vocab_size,
    "block_size": block_size,
    "lr": lr,
}, ckpt_path)
print(f"权重已成功保存至: {ckpt_path}")
