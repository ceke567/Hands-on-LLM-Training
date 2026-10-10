import sys
from pathlib import Path
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent.parent
data_path = repo_root / "data" / "raw" / "tinystories.txt"
ckpt_path = repo_root / "checkpoints" / "week02_stable.pt"

sys.path.append(str(Path(__file__).resolve().parent.parent / "week01"))
from minigpt import MiniGPT

if not ckpt_path.exists():
    raise FileNotFoundError(f"未找到模型权重文件: {ckpt_path}")

# 1. 加载 Tokenizer 词表
with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

# 2. 加载 100M 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 正在加载 100M 模型权重 (设备: {device}) ===")
checkpoint = torch.load(ckpt_path, map_location=device)

model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=768,
    n_head=12,
    n_layer=14,
    block_size=checkpoint["block_size"],
    dropout=0.0
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("🎉 100M 模型加载成功！进入交互式续写模式。")
print("你可以输入任意英文单词（比如: Lily, Once, A dog, The, Tim），模型会自回归往下编故事。")
print("提示：输入 'q' 按回车即可退出。\n" + "=" * 55)

# 3. 采样生成函数 (带温度调节，0.7~0.8 会更聚焦且生动)
@torch.no_grad()
def generate_text(prompt, max_new_tokens=100, temperature=0.75):
    # 过滤词表中没有的字符
    valid_tokens = encode(prompt)
    if not valid_tokens:
        valid_tokens = [stoi[' ']]
    idx = torch.tensor([valid_tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -checkpoint["block_size"]:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(idx_cond)
        
        # 应用温度调节
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return decode(idx[0].tolist())

# 4. 交互循环
while True:
    try:
        user_input = input("\n👉 请输入提示词 (Prompt): ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'q':
            print("退出交互模式。")
            break

        output = generate_text(user_input, max_new_tokens=120, temperature=0.75)
        print("\n🤖 模型续写生成：")
        print(output.strip())
        print("-" * 55)
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")
        break
