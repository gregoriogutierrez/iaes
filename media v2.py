#calcula la media con una red neuronal
import torch
import random

# define la red neuronal
net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
# define el optimizador
optimizador = torch.optim.SGD(net.parameters(), lr=0.01)

# define la función de pérdida
Perdida = torch.nn.MSELoss()

# train the network
numeroDePuntos = 10
numeroDeEpocas = 10000
for epoca in range(numeroDeEpocas):
    # get the data
    mediaDeEstaEpoca = random.randint(0, 10)
    valores = torch.randn(numeroDePuntos, 1) + mediaDeEstaEpoca
    mediaExacta = torch.mean(valores)

    # forward pass
    mediaPredicha = net(valores)

    # calculate the loss
    perdida = Perdida(mediaPredicha, mediaExacta)
    print('Época: {} Pérdida: {:.4f}'.format(epoca+1, perdida/numeroDePuntos)) 
    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()

# calcula la media de una secuencia de números
numeroDePuntosDePrueba = 100
valores = torch.randn(numeroDePuntosDePrueba, 1)+5
mediaPredicha = net(valores)
print("Media estimada: ",torch.mean(mediaPredicha).item(),"Media aritmética: ",torch.mean(valores).item())