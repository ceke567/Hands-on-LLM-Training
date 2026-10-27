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
print(f"=== W03-01: OOM 显存边界探测与阶段捕获 ===")
print(f"当前训练设备: {device}")
if device.type == "cuda":
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    print(f"显存总量: {total_mem:.2f} MB")

model_args = dict(
    vocab_size=256,
    block_size=256,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0
)

def test_oom_boundary(batch_sizes=[16, 32, 64, 128, 256, 512]):
    results = []
    for bs in batch_sizes:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        print(f"\n--- 测试 Batch Size = {bs} ---")
        try:
            model = MiniGPT(**model_args).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randint(0, model_args["vocab_size"], (bs, model_args["block_size"]), device=device)
            y = torch.randint(0, model_args["vocab_size"], (bs, model_args["block_size"]), device=device)
            
            stage = "前向计算 (Forward)"
            if device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
                
            stage = "反向传播 (Backward)"
            loss.backward()
            
            stage = "优化器更新 (Optimizer Step)"
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            if device.type == "cuda":
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
                print(f"✅ 成功通过！峰值显存: {peak_mem:.2f} MB")
                results.append((bs, "成功", peak_mem, "无"))
            else:
                print(f"✅ CPU 模拟通过！")
                results.append((bs, "成功", 0, "无"))
                
            del model, optimizer, x, y, logits, loss
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_msg = str(e)
            print(f"❌ 发生异常！阶段: [{stage}]")
            print(f"错误详情: {err_msg[:300]}")
            results.append((bs, f"失败: {stage}", 0, err_msg[:100]))
            break
            
    print("\n=== OOM 探测测试汇总 ===")
    for r in results:
        print(f"Batch Size: {r[0]:<4} | 状态: {r[1]:<20} | 峰值显存: {r[2]:.1f} MB")

if __name__ == "__main__":
    test_oom_boundary()
