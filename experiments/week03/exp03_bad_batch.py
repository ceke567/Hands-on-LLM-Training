import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(repo_root / "experiments" / "week01"))
sys.path.append(str(repo_root / "experiments" / "week02"))
from minigpt import MiniGPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== W03-03: 脏数据与异常 Batch 定位实验 ===")
print(f"运行设备: {device}")

model_args = dict(
    vocab_size=256,
    block_size=256,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0
)

def run_bad_batch_experiment():
    model = MiniGPT(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 模拟数据集中混入 3 类典型脏数据
    # Case 1: 正常批次
    # Case 2: 注入越界 Token (Token ID >= vocab_size)
    # Case 3: 注入全零/超短退化全重复序列
    batch_scenarios = [
        ("Step 1 [正常 Batch]", 0, False),
        ("Step 2 [正常 Batch]", 0, False),
        ("Step 3 [脏样本: 越界 Token 注入]", 9999, False), # 越界 Embedding
        ("Step 4 [正常 Batch]", 0, False),
    ]
    
    for name, inject_val, is_empty in batch_scenarios:
        print(f"\n--- 正在执行: {name} ---")
        x = torch.randint(0, model_args["vocab_size"], (16, model_args["block_size"]), device=device)
        y = torch.randint(0, model_args["vocab_size"], (16, model_args["block_size"]), device=device)
        
        if inject_val > 0:
            # 在样本 ID 7 的中间位置注入非法 Token 9999
            print("🚨 [数据层] 注入非法越界 Token ID: 9999 (合法范围 0~255) 于样本索引 7")
            x[7, 128] = inject_val
            
        try:
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            print(f"✅ 执行完成，Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"❌ 捕获运行时崩溃！错误类型: {type(e).__name__}")
            print(f"错误细节: {e}")
            print("💡 结论: 必须在 Dataset/DataLoader 层构建 Token 范围断言 (0 <= token < vocab_size)，防止脏数据穿透到 GPU 算子引发 HIP/CUDA Device-side Assert 崩溃。")

if __name__ == "__main__":
    run_bad_batch_experiment()
