#conv1d
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

tamañoLote = 4
totalDeHoras = 600
canalesEntrada = 1
totalEpocas = 100
cuentaPintados = 0
totalDatos = 100
proporcionUnosEnRuido = 1
proporcionCerosEnRuido = 1 - proporcionUnosEnRuido

# Definir una clase de red neuronal que utiliza MaxPool1d y Conv1d para clasificación binaria
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(8 * totalDeHoras, 3)  # 3 clases: 0, 1 y 2

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Crear una instancia del modelo
modelo = Net()
cuentaPintados = False
def CreaDatos():
    arrayRuido =  np.random.choice([1, 0], size=totalDeHoras, p=[proporcionUnosEnRuido, proporcionCerosEnRuido])
    global cuentaPintados
    global graficoSeñalPintado
    datosEntrada = torch.zeros(totalDatos, 1, totalDeHoras)
    etiquetasDeDatos = torch.zeros(totalDatos, dtype = torch.long)
    for contadorDatos in range(totalDatos):
        tipoFalta = random.randint(0, 2)
        ##print (tipoFalta)
        y = np.zeros(totalDeHoras)
        if tipoFalta == 0:
            proporcionFaltas = random.random() * 0.1
            proporcionAsistencia = 1 - proporcionFaltas
            arrayFalta = np.random.choice([1, 0], size=totalDeHoras, p=[proporcionFaltas, proporcionAsistencia])
            ##input_data[contadorDatos,0,:] = torch.randint(0, 2, (totalDeHoras,),dtype=torch.float32) * arrayRuido
            datosEntrada[contadorDatos,0,:] = torch.from_numpy(arrayFalta * arrayRuido).long()
            etiquetasDeDatos[contadorDatos] = torch.tensor([0])
            if (cuentaPintados<3): 
                plt.title('Tipo 0 '+ str(contadorDatos))
                plt.plot(datosEntrada[contadorDatos,0,:])
                plt.show()
                cuentaPintados+=1
        elif tipoFalta == 1:
            etiquetasDeDatos[contadorDatos] = torch.tensor([1])
            x = np.linspace(0,10, totalDeHoras)
            numeroDeCampanas = random.randint(1, 5)
            for contadorCampanas in range(numeroDeCampanas):
                valorMedio = random.random() * 10
                desviacionEstandar = .1
                if contadorCampanas == 0:
                    y = 1/(np.sqrt(2 * np.pi * desviacionEstandar**2)) * np.exp(- (x - valorMedio)**2 / (2 * desviacionEstandar**2))
                else:
                    y = y + 1/(np.sqrt(2 * np.pi * desviacionEstandar**2)) * np.exp(- (x - valorMedio)**2 / (2 * desviacionEstandar**2))
            media = np.mean(y)
            datosEntrada[contadorDatos,0,:] = torch.tensor(y, dtype = torch.float32) * arrayRuido
            datosEntrada[contadorDatos,0,:]  = torch.where(datosEntrada[contadorDatos,0,:] > media, torch.tensor(1), torch.tensor(0))
            if (cuentaPintados<3): 
                plt.title('Tipo 1 ' + str(contadorDatos))
                plt.plot(datosEntrada[contadorDatos,0,:])
                plt.show()
                cuentaPintados+=1
        else:
            proporcionFaltas = random.random() * 0.9 + 0.1
            proporcionAsistencia = 1 - proporcionFaltas
            arrayFalta = np.random.choice([1, 0], size=totalDeHoras, p=[proporcionFaltas, proporcionAsistencia])
            datosEntrada[contadorDatos,0,:] = torch.from_numpy(arrayFalta * arrayRuido).long()
            etiquetasDeDatos[contadorDatos] = torch.tensor([2])
            if (cuentaPintados<3): 
                plt.title('Tipo 2 '+ str(contadorDatos))
                plt.plot(datosEntrada[contadorDatos,0,:])
                plt.show()
                cuentaPintados+=1
    datosEntrada = torch.tensor(datosEntrada, dtype = torch.float32)
    print(etiquetasDeDatos)
    return datosEntrada, etiquetasDeDatos

# Crear dataset y dataloader

class MyDataset(Dataset):
    def __init__(self):
        # Definir los datos de entrada y salida
        self.input_data,self.target_data = CreaDatos()
        
    def __len__(self):
        # Devolver la longitud total del dataset
        return len(self.input_data)

    def __getitem__(self, idx):
        # Obtener un ejemplo específico del dataset
        input_sample = self.input_data[idx]
        target_sample = self.target_data[idx]

        return input_sample, target_sample

# Crear una instancia del dataset
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=tamañoLote, shuffle=True)

# Generar datos aleatorios
input_data = torch.randn(tamañoLote, canalesEntrada, totalDeHoras)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(modelo.parameters(), lr=0.01)

# Entrenamiento

for epoca in range(totalEpocas):
    for inputs, targets in dataloader:
        # Reiniciar los gradientes
        optimizer.zero_grad()

        # Pasar los datos a través del modelo
        outputs = modelo(inputs)

        # Calcular la pérdida
        loss = criterion(outputs, targets)

        # Retropropagación y optimización
        loss.backward()
        optimizer.step()

    # Imprimir la pérdida del epoch actual cada 1000 epochs
    if (epoca+1) % (totalEpocas//10) == 0:
        print(f"Epoch [{epoca+1}/{totalEpocas}], Loss: {loss.item()}")

#prueba
testSet = MyDataset()
testLoader = DataLoader(testSet, batch_size=tamañoLote, shuffle=True)
totalAciertos0 = 0
totalAciertos1 = 0
totalAciertos2 = 0
totalEtiquetas0 = 0
totalEtiquetas1 = 0
totalEtiquetas2 = 0

for inputs, targets in testLoader:
    outputs = modelo(inputs)
    loss = criterion(outputs, targets)
    _, predicted = torch.max(outputs.data, 1)
    totalEtiquetas0 += (targets==0).sum().item()
    totalEtiquetas1 += (targets==1).sum().item()
    totalEtiquetas2 += (targets==2).sum().item()
    totalAciertos0 += ((predicted == targets)&(targets==0)).sum().item()
    totalAciertos1 += ((predicted == targets)&(targets==1)).sum().item()
    totalAciertos2 += ((predicted == targets)&(targets==2)).sum().item()

    print(f"Loss: {loss.item()}")
    print(f"Targets: {targets}")
    print(f"Outputs: {outputs}")
print(f"Precisión 0: {totalAciertos0 / totalEtiquetas0*100}%")
print(f"Precisión 1: {totalAciertos1 / totalEtiquetas1*100}%")
print(f"Precisión 2: {totalAciertos2 / totalEtiquetas2*100}%")
# Guardar el modelo entrenado
torch.save(modelo.state_dict(), "model.pth")
 