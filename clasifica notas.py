import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
import time

inicio = time.time()
totalEpocas = 10000
neuronasEntrada = 30
multiplicadorNeuronas = 2
variacionNotas = 3
pruebasTotales = 1000
velocidadAprendizaje = 0.005
momentoSGD = 0#0.5

# Definir la arquitectura de la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(neuronasEntrada, multiplicadorNeuronas * neuronasEntrada)
        self.fc3 = nn.Linear(multiplicadorNeuronas * neuronasEntrada, multiplicadorNeuronas * neuronasEntrada//2)
        self.fc4 = nn.Linear(multiplicadorNeuronas * neuronasEntrada//2, multiplicadorNeuronas * neuronasEntrada//4)
        self.fc2 = nn.Linear(multiplicadorNeuronas * neuronasEntrada//4, 10)
        
    def forward(self, x):
        ##x = x.view(-1, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)        
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim = 1)

# Instanciar el modelo, la función de pérdida y el optimizador
modelo = Net()
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr = velocidadAprendizaje)

# Entrenamiento del modelo
for epoca in range(totalEpocas):
    perdida = 0.0
    optimizador.zero_grad()
    numeroAleatorio = random.randint(0,9)
    entradas = torch.full((1,neuronasEntrada),numeroAleatorio).float()
    for cuentaEntrada in range(neuronasEntrada):
        entradas[0,cuentaEntrada] = numeroAleatorio + random.randint(-variacionNotas,variacionNotas)
    media = torch.mean(entradas)
    etiquetas = torch.tensor([numeroAleatorio]).long()
    outputs = modelo(entradas)
    loss = criterio(outputs, etiquetas)
    loss.backward()
    optimizador.step()
    perdida += loss.item()
    if epoca % (totalEpocas/10) == 0:
        print('[%d, %5d] loss: %.6f' % (epoca + 1,  1, perdida / 100))
        perdida = 0.0

# Pruebas del modelo       
pruebasCorrectas = 0
for cuentaPruebas in range (pruebasTotales):
    numeroAleatorio = random.randint(0,9)
    entradas = torch.full((1,neuronasEntrada),numeroAleatorio).float()
    for cuentaEntrada in range(neuronasEntrada):
        entradas[0,cuentaEntrada] = numeroAleatorio + random.randint(-variacionNotas,variacionNotas)
    media = torch.mean(entradas)
    prediccion = modelo(entradas)
    prediccionMejor = torch.argmax(prediccion)
    prediccion[0,prediccionMejor] = -10000000
    prediccionSegundoMejor = torch.argmax(prediccion)
    if (prediccionMejor == numeroAleatorio):
        pruebasCorrectas += 1
print ("Pruebas correctas = ",int(pruebasCorrectas/pruebasTotales*100),"%")
fin = time.time()
print("Tiempo de ejecución: ",fin - inicio)
