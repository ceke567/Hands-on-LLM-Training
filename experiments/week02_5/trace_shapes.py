import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(repo_root / "experiments" / "week01"))
sys.path.append(str(repo_root / "experiments" / "week02"))
from minigpt import MiniGPT

model = MiniGPT(vocab_size=256, block_size=256, n_layer=1, n_head=12, n_embd=768, dropout=0.0)
model.eval()

input_ids = torch.tensor([[65, 66, 67, 68]])
B, T = input_ids.shape
print("=== 节点 0: 原始输入 Token IDs ===")
print(f"形状: {list(input_ids.shape)}，内容: {input_ids.tolist()}\n")

tok_emb = model.transformer.wte(input_ids)
pos = torch.arange(0, T, dtype=torch.long)
pos_emb = model.transformer.wpe(pos)
x = tok_emb + pos_emb
print("=== 节点 1: 进 Embedding 并加上位置编码 ===")
print(f"形状变化: [{B}, {T}] -> {list(x.shape)} (每个字变成了 768 维向量)\n")

block = model.transformer.h[0]
x_norm = block.ln_1(x)
qkv = block.attn.c_attn(x_norm)
q, k, v = qkv.split(768, dim=2)

k = k.view(B, T, 12, 64).transpose(1, 2)
q = q.view(B, T, 12, 64).transpose(1, 2)
v = v.view(B, T, 12, 64).transpose(1, 2)

print("=== 节点 2: 切分多头 (Multi-Head) ===")
print(f"Q 形状: {list(q.shape)} -> [Batch={B}, 头数=12, 序列长={T}, 每头维度=64]\n")

att_raw = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(64))
print("=== 节点 3: 算完 Q @ K^T (4x4 原始相关度，以第 1 个头为例) ===")
print(f"形状: {list(att_raw.shape)}")
print(att_raw[0, 0].detach().round(decimals=2))

mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
att_masked = att_raw.masked_fill(mask == 0, float('-inf'))
print("\n👉 加上下三角掩码 (Mask) 后的矩阵:")
print(att_masked[0, 0].detach())

att_probs = F.softmax(att_masked, dim=-1)
print("\n👉 过完 Softmax 后的注意力权重 (右上角彻底变成 0.0000，每行和为 1.0):")
print(att_probs[0, 0].detach().round(decimals=4))
print()

y = att_probs @ v
y = y.transpose(1, 2).contiguous().view(B, T, 768)
print("=== 节点 4: 乘以 V 并重拼多头 ===")
print(f"形状变回: {list(y.shape)}\n")

logits = model.lm_head(y)
print("=== 节点 5: 经过最终的 lm_head 投影到词表 ===")
print(f"最终输出 Logits 形状: {list(logits.shape)} -> [Batch={B}, 序列长={T}, 词表大小=256]\n")
print("✅ 第一关：张量变形记全流程打通！")
