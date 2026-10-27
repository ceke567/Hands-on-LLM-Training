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
print(f"=== W03-02: Loss Spike 与 NaN/Inf 故障定位实验 ===")
print(f"运行设备: {device}")

model_args = dict(
    vocab_size=256,
    block_size=256,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0
)

def run_lr_spike_experiment():
    print("
[实验 1: 制造超大学习率引发 Loss Spike 与数值发散]")
    model = MiniGPT(**model_args).to(device)
    danger_lr = 10.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=danger_lr)
    
    for step in range(15):
        x = torch.randint(0, model_args["vocab_size"], (16, model_args["block_size"]), device=device)
        y = torch.randint(0, model_args["vocab_size"], (16, model_args["block_size"]), device=device)
        
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
            logits, loss = model(x, y)
            
        is_nan_loss = torch.isnan(loss).item() or torch.isinf(loss).item()
        loss_val = loss.item() if not is_nan_loss else float('nan')
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        optimizer.step()
        
        print(f"Step {step+1:02d} | LR: {danger_lr:.1f} | Loss: {loss_val:<10.4f} | 梯度范数 Grad Norm: {total_norm:.2f}")
        if is_nan_loss or total_norm > 1e5:
            print(f"💥 捕获异常！Step {step+1} 触发严重 Loss Spike / NaN 发散！")
            break

def run_nan_probe_experiment():
    print("
[实验 2: 构建网络各层 NaN 探针，精确定位首个发散环节]")
    model = MiniGPT(**model_args).to(device)
    first_nan_layer = None
    def make_nan_probe(name):
        def hook(module, input, output):
            nonlocal first_nan_layer
            if first_nan_layer is not None:
                return
            tensor_to_check = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor_to_check, torch.Tensor):
                if torch.isnan(tensor_to_check).any() or torch.isinf(tensor_to_check).any():
                    first_nan_layer = name
                    print(f"🚨 [探针报警] 检测到首个产生 NaN/Inf 的层: {name}")
        return hook

    for name, module in model.named_modules():
        module.register_forward_hook(make_nan_probe(name))
        
    print("注入极端权重数值，测试前向传播探针定位响应...")
    with torch.no_grad():
        x = torch.randint(0, model_args["vocab_size"], (16, model_args["block_size"]), device=device)
        model.transformer.h[0].attn.c_attn.weight.data.fill_(1e10)
        
    try:
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
            logits, loss = model(x, x)
    except Exception as e:
        print(f"捕获执行异常: {e}")
        
    print(f"🎯 首个发散环节判定完毕: [{first_nan_layer}]")

if __name__ == '__main__':
    run_lr_spike_experiment()
    run_nan_probe_experiment()
