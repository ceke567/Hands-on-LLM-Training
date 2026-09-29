import torch
from pathlib import Path

# 自动定位仓库根目录与数据路径
repo_root = Path(__file__).resolve().parent.parent.parent
data_path = repo_root / "data" / "raw" / "input.txt"

if not data_path.exists():
    raise FileNotFoundError(f"未找到原始数据文件: {data_path}，请确认已上传到 data/raw/input.txt")

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. 构建字符表与 Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# 2. 将全部文本转为 PyTorch 整数张量 (Tensor)
data = torch.tensor(encode(text), dtype=torch.long)

# 3. 划分训练集 (90%) 与验证集 (10%)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("=== 数据划分结果 ===")
print(f"总 token 数: {len(data):,}")
print(f"训练集大小 (90%): {len(train_data):,} tokens")
print(f"验证集大小 (10%): {len(val_data):,} tokens")

# 4. 抽查 3 组“输入 x -> 目标 y”样本 (以窗口长度 8 为例)
torch.manual_seed(42)
block_size = 8
print(f"\n=== 抽查 3 组移位样本 (窗口长度 block_size={block_size}) ===")

for i in range(3):
    idx = torch.randint(len(train_data) - block_size, (1,)).item()
    x = train_data[idx : idx + block_size]
    y = train_data[idx + 1 : idx + block_size + 1]

    print(f"\n--- 样本 {i + 1} ---")
    print(f"输入 x (数字): {x.tolist()}")
    print(f"目标 y (数字): {y.tolist()}")
    print(f"输入 x (文本): {decode(x.tolist())!r}")
    print(f"目标 y (文本): {decode(y.tolist())!r}")
    is_shifted = torch.equal(x[1:], y[:-1])
    print(f"向右错开一位核对: {'通过 (严格一致)' if is_shifted else '失败'}")
