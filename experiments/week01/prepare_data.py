import os
from pathlib import Path

# 自动定位仓库根目录与数据路径，无论在哪个目录下运行都能找到
repo_root = Path(__file__).resolve().parent.parent.parent
data_path = repo_root / "data" / "raw" / "input.txt"

if not data_path.exists():
    raise FileNotFoundError(f"未找到原始数据文件: {data_path}，请确认已上传到 data/raw/input.txt")

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. 统计总字符数与不重复字符集合
total_chars = len(text)
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"数据读取成功！")
print(f"文本总字符数: {total_chars:,}")
print(f"不重复字符总数 (vocab_size 词表大小): {vocab_size}")
print(f"包含的全部字符: {''.join(chars)!r}")

# 2. 建立字符与数字编号的双向映射（Tokenizer 字典）
stoi = {ch: i for i, ch in enumerate(chars)}  # 字符 -> 数字 ID
itos = {i: ch for i, ch in enumerate(chars)}  # 数字 ID -> 字符

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# 3. 验证“编码后再还原”
test_str = "Hello, MiniGPT!"
encoded = encode(test_str)
decoded = decode(encoded)

print("\n--- Tokenizer 编解码测试 ---")
print(f"原始测试文字: {test_str!r}")
print(f"编码后数字列表: {encoded}")
print(f"解码还原后文字: {decoded!r}")
assert decoded == test_str, "还原失败，解码结果与原始文字不一致！"
print("测试通过：任意文本能 100% 编号并无损还原！")
