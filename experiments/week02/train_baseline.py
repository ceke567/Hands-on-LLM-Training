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

# ==========================================
# 实验 W02-01：100M 基线 (FP32 单精度) 配置
# ==========================================
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 运行设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

n_layer = 14          # 14 层 Transformer Block
n_embd = 768          # 隐藏维度 768
n_head = 12           # 注意力头数 12
block_size = 256      # 上下文窗口 256
batch_size = 16       # 批大小 16 (单步 16 * 256 = 4,096 tokens)
learning_rate = 3e-4  # AdamW 初始学习率
max_iters = 200       # 基线评测步数: 200 步
eval_interval = 50    # 每 50 步评测一次验证 loss
eval_iters = 10       # 评测时抽样 10 个 batch 求平均

# 1. 读取 TinyStories 数据并构建 Tokenizer
print("1. 正在加载 TinyStories 数据集...")
with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

print(f"   数据集字符总量: {len(text):,} 字符")
print(f"   动态字符表大小 (vocab_size): {vocab_size}")

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"   训练集: {len(train_data):,} tokens | 验证集: {len(val_data):,} tokens")

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
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# 2. 初始化 100M MiniGPT (FP32 默认精度)
model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=0.1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f"2. 100M 模型初始化完成，总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

# 3. 训练前基线采样 (固定使用 TinyStories 经典提示词: 'Once upon a time, ')
prompt = "Once upon a time, "
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
print("\n--- [训练前基线生成样例 (未训练纯随机)] ---")
gen_tokens_before = model.generate(context, max_new_tokens=100)[0].tolist()
print(repr(decode(gen_tokens_before)))
print("------------------------------------------\n")

# 4. 开始 FP32 基线训练
print(f"=== 开始 100M 模型基线训练 (FP32，共 {max_iters} 步) ===")
start_train_time = time.time()
step_times = []

for iter in range(max_iters + 1):
    step_start = time.time()

    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        avg_step_ms = (sum(step_times[-50:]) / len(step_times[-50:]) * 1000) if step_times else 0
        print(f"step {iter:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | 显存: {peak_mem:.1f} MB | 单步耗时: {avg_step_ms:.1f} ms")

    if iter == max_iters:
        break

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    step_times.append(time.time() - step_start)

total_time = time.time() - start_train_time
total_tokens = max_iters * batch_size * block_size
tokens_per_sec = total_tokens / total_time
peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

print(f"\n=== W02-01 (100M FP32 基线) 训练完成 ===")
print(f"总耗时: {total_time:.2f} 秒")
print(f"峰值显存: {peak_mem:.1f} MB")
print(f"平均吞吐速率: {tokens_per_sec:,.0f} tokens/秒")

# 5. 保存基线 Checkpoint
ckpt_path = ckpt_dir / "week02_baseline_fp32.pt"
torch.save({
    "step": max_iters,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "vocab_size": vocab_size,
    "block_size": block_size,
}, ckpt_path)
print(f"基线权重已成功保存至: {ckpt_path}")

# 6. 训练后生成样例对比
print("\n--- [训练后生成样例 (同一提示词)] ---")
gen_tokens_after = model.generate(context, max_new_tokens=150)[0].tolist()
print(repr(decode(gen_tokens_after)))
print("------------------------------------------")
