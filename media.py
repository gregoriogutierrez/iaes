import torch
import random

# define the network
net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

# define the optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# define the loss function
loss_fn = torch.nn.MSELoss()

# train the network
numeroDePuntos = 10
numeroDeEpocas = 10000
for epoch in range(numeroDeEpocas):
    ##print(epoch)
    # get the data
    mediaAleatoria = random.randint(0, 10)
    x = torch.randn(numeroDePuntos, 1) + mediaAleatoria
    ##print(x)
    y = torch.mean(x)

    # forward pass
    y_hat = net(x)

    # calculate the loss
    loss = loss_fn(y_hat, y)
    print('Epoch: {} Test Loss: {:.4f}'.format(epoch+1, loss/numeroDePuntos))    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# calculate the mean of a sequence
numeroDePuntosDePrueba = 100
x = torch.randn(numeroDePuntosDePrueba, 1)+5
y_hat = net(x)
##print(x)
mean = torch.mean(y_hat).item()
print(mean,torch.mean(x).item())