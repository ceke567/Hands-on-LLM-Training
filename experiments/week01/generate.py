import sys
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
data_path = repo_root / "data" / "raw" / "input.txt"
ckpt_path = repo_root / "checkpoints" / "minigpt_final.pt"

sys.path.append(str(Path(__file__).resolve().parent))
from minigpt import MiniGPT

if not ckpt_path.exists():
    raise FileNotFoundError(f"未找到权重文件: {ckpt_path}，请先运行 train.py 训练并保存权重。")

# 1. 重建字符级 Tokenizer
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# 2. 初始化模型并重新加载 Checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 验证 Checkpoint 重新加载 (设备: {device}) ===")

checkpoint = torch.load(ckpt_path, map_location=device)
model = MiniGPT(vocab_size=checkpoint["vocab_size"], block_size=checkpoint["block_size"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"成功恢复 step={checkpoint['step']} 训练状态！模型已进入 eval 评估模式。")

# 3. 测试不同 Prompt 生成
test_prompts = ["\n", "KING:", "To be, or not to be"]

for prompt in test_prompts:
    print(f"\n--- [提示词 Prompt: {prompt!r}] ---")
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out_tokens = model.generate(context, max_new_tokens=150)[0].tolist()
    print(decode(out_tokens))

print("\n🎉 Checkpoint 加载与独立推理生成全部测试通过！")
