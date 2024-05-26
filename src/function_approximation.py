import torch
import matplotlib.pyplot as plt
import numpy as np

#f = lambda x: torch.sin(x) + .3 * torch.exp(x)
#f = lambda x: torch.sin(x) * 1/x + 0.3 * torch.cos(x) - 0.2 * 1/x + torch.exp(x) * 0.1
f = lambda x: torch.log(x)
x = torch.linspace(2, 8, 50)
n = 64

class Net(torch.nn.Module):
    def __init__(self, n):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(1, n)
        self.output = torch.nn.Linear(n, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = self.output(x)
        return x

net = Net(n)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(1000):
    running_loss = 0.0
    optimizer.zero_grad()
    outputs = net(x.unsqueeze(1))
    loss = criterion(outputs.squeeze(), f(x))
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    if epoch % 100 == 0:
        print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))
        for name, param in net.named_parameters():
                #print(f"Grad:{param.grad}")
                grad_np = param.grad.numpy()
                non_zero_cnt = np.count_nonzero(
                    grad_np) / grad_np.size
                print(f"Non-zero amount: {non_zero_cnt:.2f}")

x_plot = torch.linspace(2, 8, 50)
actual_y = torch.tensor([f(p) for p in x_plot])
predicted_y = net(x.unsqueeze(1)).squeeze()
plt.plot(x, predicted_y.detach().numpy(), 'b', label='Predicted Function')
plt.plot(x_plot, actual_y, 'g', label='Actual Function')
plt.legend()
plt.show()