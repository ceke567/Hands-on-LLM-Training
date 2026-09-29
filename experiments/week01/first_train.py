import torch
from torch import nn

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    print("device: cpu")

x = torch.linspace(-1, 1, 256).reshape(-1, 1).to(device)
y = 3 * x + 2

model = nn.Linear(1, 1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for step in range(201):
    prediction = model(x)
    loss = loss_fn(prediction, y)

    if step % 50 == 0:
        print(f"step={step:3d} loss={loss.item():.8f}")

    if step == 200:
        break

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"weight={model.weight.item():.4f}")
print(f"bias={model.bias.item():.4f}")
