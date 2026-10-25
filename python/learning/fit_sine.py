import torch

data = torch.linspace(-3, 3, 1000).unsqueeze(1)
targets = torch.sin(data)
net = torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,1),
)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

for i in range(2000):
    prediction = net(data)
    loss = ((prediction - targets) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%200 == 0:
        print(f"epoch {i}: loss {loss.item():.5f}")

tests = torch.tensor([[-2.0], [0.0], [1.0], [2.5]])
with torch.no_grad():
    print(net(tests), torch.sin(tests))
