import os
import sys
import time
import math
from pathlib import Path
import torch
import torch.nn.functional as F

# 路径定位
repo_root = Path(__file__).resolve().parent.parent.parent
data_path = repo_root / "data" / "raw" / "input.txt"
ckpt_dir = repo_root / "checkpoints"
ckpt_dir.mkdir(exist_ok=True)

# 引入 MiniGPT 模型架构
sys.path.append(str(Path(__file__).resolve().parent))
from minigpt import MiniGPT

# 1. 实验超参数设定
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 运行设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

batch_size = 32       # 批大小
block_size = 128      # 上下文窗口
max_iters = 200       # 短跑验证步数: 200 步
eval_interval = 50    # 每 50 步评测一次验证 loss
eval_iters = 20       # 评测时抽样 20 个批次求平均
learning_rate = 3e-4  # AdamW 学习率

# 2. 读取数据与构建字符级 Tokenizer
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# 90% 训练，10% 验证划分
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
    """在训练集和验证集上分别评估平滑后的 Loss"""
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

# 3. 初始化 10.7M MiniGPT 与 AdamW 优化器
model = MiniGPT(vocab_size=vocab_size, block_size=block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 4. 训练前基线采样 (固定使用换行符 '\n' 作为基线 prompt)
prompt = "\n"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
print("\n--- [训练前基线生成样例 (纯随机权重)] ---")
gen_tokens_before = model.generate(context, max_new_tokens=100)[0].tolist()
print(repr(decode(gen_tokens_before)))
print("------------------------------------------\n")

# 5. 执行短跑训练 (200 步)
print(f"=== 开始短跑训练 (共 {max_iters} 步) ===")
start_time = time.time()

for iter in range(max_iters + 1):
    # 定期评估训练 loss 和验证 loss
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        print(f"step {iter:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | 显存占用: {peak_mem:.1f} MB")

    if iter == max_iters:
        break

    # 单步参数更新
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

total_time = time.time() - start_time
print(f"\n短跑 200 步完成，总耗时: {total_time:.2f} 秒")

# 6. 保存 Checkpoint (包含模型和优化器状态)
ckpt_path = ckpt_dir / "ckpt_short_200.pt"
torch.save({
    "step": max_iters,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "vocab_size": vocab_size,
    "block_size": block_size,
}, ckpt_path)
print(f"Checkpoint 已成功保存至: {ckpt_path}")

# 7. 短跑 200 步后的初步生成样例 (使用相同的 prompt 对比)
print("\n--- [短跑 200 步后生成样例] ---")
gen_tokens_after = model.generate(context, max_new_tokens=100)[0].tolist()
print(repr(decode(gen_tokens_after)))
print("------------------------------------------")
