import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import datetime
#calcula el tiempo total de ejecución
import time
inciaTiempo = time.time()
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
else:
    device = torch.device("cpu")
fraccionEntrenamiento = 0.8
tamañoLote = 200
velocidadAprendizaje = 0.1
totalDeHoras = 0
canalesDeEntrada = 1
neuronasPorNota = 10
numeroDeAsignaturas = 10
totalEpocas = 200
serieEpocas =[]
totalDatos = 0
perdidas_entrenamiento = []
perdidas_evaluacion = []
maximaPrecision = 0
epocaMaximaPrecision = 0

diccionarioVentanaPanoramica = {
                                1: '(A) ABSENTISTA', 
                                2: '(NR) NO SE RELACIONA',
                                3: '(NP) NO PUEDE',  
                                4: '(NS) NO SE COMPORTA',
                                5: '(SN) SE NIEGA A TRABAJAR O ESTUDIAR EN EL AULA',
                                6: '(NA) NO ATIENDE',
                                7: '(NE) NO ESTUDIA Y NO TRABAJA EN CASA'
                                }

numeroDeCodigos = len(diccionarioVentanaPanoramica)

notasDatos = torch.ones(totalDatos, 1, numeroDeAsignaturas)
faltasDatos = torch.ones(totalDatos, 1, totalDeHoras)

# Definir una clase de red neuronal que utiliza MaxPool1d y Conv1d para clasificación binaria

def ImportaDatos():
    global totalDatos, totalDeHoras
    notasDatos=torch.load('multiplesNotasDatos.pth')
    etiquetasDeDatosOriginal=torch.load('multiplesEtiquetasDeDatos.pth')
    etiquetasDeDatos=torch.clamp(etiquetasDeDatosOriginal[:,0].long()-1,min=0)
    faltasDatos=torch.load('multiplesFaltasDatos.pth')
    totalDatos = len(notasDatos)
    totalDeHoras = len(faltasDatos[0,0,:])
    return faltasDatos, notasDatos, etiquetasDeDatos

# Crear dataset y dataloader
class MyDataset(Dataset):
    def __init__(self):
        # Definir los datos de entrada y salida
        self.faltasDatos, self.notasDatos, self.objetivoDatos = ImportaDatos() 
        
    def __len__(self):
        # Devolver la longitud total del dataset
        return len(self.notasDatos)

    def __getitem__(self, idx):
        global totalDatos,totalDeHoras
        # Obtener un ejemplo específico del dataset
        faltasMuestra = self.faltasDatos[idx]
        notasMuestra = self.notasDatos[idx]
        objetivoMuestra = self.objetivoDatos[idx]
        return faltasMuestra, notasMuestra, objetivoMuestra

dataset = MyDataset()

datosTotales = len(dataset)

entrenamientoTamaño = int(fraccionEntrenamiento * datosTotales)
evaluacionTamaño = datosTotales - entrenamientoTamaño

indices = list(range(datosTotales))
entrenamientoIndices, evaluacionIndices = train_test_split(indices, test_size=evaluacionTamaño)

# Crea los subconjuntos utilizando los índices
entrenamientoSubset = Subset(dataset, entrenamientoIndices)
evaluacionSubset = Subset(dataset, evaluacionIndices)

# Crea los dataloaders para entrenamiento y evaluación
entrenamientoDataloader = DataLoader(entrenamientoSubset, batch_size=tamañoLote, shuffle=True)
evaluacionDataloader = DataLoader(evaluacionSubset, batch_size=tamañoLote, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.capaNotas = nn.Linear(numeroDeAsignaturas, neuronasPorNota)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(8 * totalDeHoras, numeroDeCodigos)  # 7 codigos de ventana panoramica
        # self.capaOculta = nn.Linear(16 * 8, 10)
        self.capaOculta = nn.Linear(totalDeHoras*8+neuronasPorNota, totalDeHoras*8)

    def forward(self, faltas, nota):
        faltas = self.conv(faltas)
        nota = self.capaNotas(nota)
        faltas = nn.functional.relu(faltas)
        faltas = self.maxpool(faltas)
        faltas = faltas.view(faltas.size(0), -1)
        nota = nota.view(nota.size(0), -1)
        salidaConcatenada = torch.cat((faltas, nota), dim=1)
        salidaCapaOculta = self.capaOculta(salidaConcatenada)
        capaOculta = nn.functional.relu(salidaCapaOculta)
        salida =  self.fc(capaOculta)
        return salida
modelo = Net()
modelo.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(modelo.parameters(), lr = velocidadAprendizaje)

fig, ax = plt.subplots(figsize=(10, 6))
plt.ion()
linea_entrenamiento, = ax.plot([], [], label='Pérdida de entrenamiento')
linea_evaluacion, = ax.plot([], [], label='Pérdida de evaluación')
ax.set_xlabel('Época')
ax.set_ylabel('Pérdida')
ax.set_title('Curvas de aprendizaje')
ax.set_xlim(0, totalEpocas-1)
ax.set_ylim(0 , 2)
ax.legend()
# Entrenamiento
for epoca in range(totalEpocas):
    modelo.train()
    cuentaPerdidas = 0
    perdidaEpoca = 0
    for inputFaltas, inputNotas, targets in entrenamientoDataloader:
        if torch.cuda.is_available():
            inputFaltas = inputFaltas.to(device)
            inputNotas = inputNotas.to(device)
            targets = targets.to(device)
        # Reiniciar los gradientes
        optimizer.zero_grad()

        # Pasar los datos a través del modelo
        outputs = modelo(inputFaltas, inputNotas)

        # Calcular la pérdida
        perdida = criterion(outputs, targets)
        perdidaEpoca+=perdida.item()
        cuentaPerdidas+=1
        # Retropropagación y optimización
        perdida.backward()
        optimizer.step() 
    perdidas_entrenamiento.append(perdidaEpoca/cuentaPerdidas)
    

    # Imprimir la pérdida del epoch actual cada 1000 epochs
    if totalEpocas>9:
        if (epoca+1) % (totalEpocas//10) == 0:
            print(f"Epoch [{epoca+1}/{totalEpocas}], Loss: {perdida.item()}")
    ##perdidas_entrenamiento.append(1)
    print (epoca)
    serieEpocas.append(epoca)
    #prueba
    modelo.eval()
    if totalDatos/10>199:
        totalDatos=200
    totalEtiquetas = np.zeros(numeroDeCodigos)
    totalAciertos = np.zeros(numeroDeCodigos)
    perdidaAcumulada = 0
    numeroPerdidas = 0
    for inputFaltas, inputNotas, targets in evaluacionDataloader:
        if torch.cuda.is_available():
            inputFaltas, inputNotas, targets = inputFaltas.to(device), inputNotas.to(device), targets.to(device)
        outputs = modelo(inputFaltas, inputNotas)
        perdida = criterion(outputs, targets)
        perdidaAcumulada += perdida.item()
        numeroPerdidas += 1
        _, predicted = torch.max(outputs.data, 1)

        for i in range(numeroDeCodigos):
            totalEtiquetas[i] += (targets==i).sum().item()
            totalAciertos[i] += ((predicted == targets)&(targets==i)).sum().item()
    perdidas_evaluacion.append(perdidaAcumulada/numeroPerdidas)    
    ##perdidas_evaluacion.append(0.5)
    linea_entrenamiento.set_data(serieEpocas, perdidas_entrenamiento)
    linea_evaluacion.set_data(serieEpocas, perdidas_evaluacion)
    #ax.relim()
    #ax.autoscale_view()
    fig.canvas.draw()
    #fig.canvas.flush_events()
    plt.pause(0.00001)
    
    aciertosMediosPonderados=0
    totalIntentos = 0
    for i in range(numeroDeCodigos):
        totalIntentos += totalEtiquetas[i]
        aciertosMediosPonderados += totalAciertos[i]
        if totalEtiquetas[i]==0:
                print(f'Aciertos: {0}%, intentos: {0} {diccionarioVentanaPanoramica[i+1]}')
        else:
            print(f'Aciertos: {round(totalAciertos[i]/totalEtiquetas[i]*100)}%, intentos: {int(totalEtiquetas[i])} {diccionarioVentanaPanoramica[i+1]}')
    print(f'Acierto medio ponderado: {round(aciertosMediosPonderados/totalIntentos*100)}%')
    if maximaPrecision<aciertosMediosPonderados/totalIntentos:
        maximaPrecision = aciertosMediosPonderados/totalIntentos
        epocaMaximaPrecision = epoca
        torch.save(modelo.state_dict(), "modelo.pth")
#imprime el tiempo total de ejecución
print("--- %s segundos ---" % (time.time() - inciaTiempo))
#crea archivo de texto con todos los parametros del modelo con fecha y hora en el nombre del archivo
now = datetime.datetime.now()
nombreArchivo ="entrenamiento "+ now.strftime("%Y-%m-%d %Hh%Mm%Ss")
archivo = open(nombreArchivo + ".txt", "w")
archivo.write("fraccionEntrenamiento: " + str(fraccionEntrenamiento) + "\n")
archivo.write("tamañoLote: " + str(tamañoLote) + "\n")
archivo.write("velocidadAprendizaje: " + str(velocidadAprendizaje) + "\n")
archivo.write("canalesDeEntrada: " + str(canalesDeEntrada) + "\n")
archivo.write("neuronasPorNota: " + str(neuronasPorNota) + "\n")
archivo.write("totalEpocas: " + str(totalEpocas) + "\n")
archivo.write("perdidas_entrenamiento: " + str(perdidas_entrenamiento) + "\n")
archivo.write("perdidas_evaluacion: " + str(perdidas_evaluacion) + "\n")
archivo.write("maximaPrecision: " + str(maximaPrecision) + "\n")
archivo.write("epocaMaximaPrecision: " + str(epocaMaximaPrecision) + "\n")
archivo.write("totalDatos: " + str(totalDatos) + "\n")
archivo.write("tiempoTotal: " + str(time.time() - inciaTiempo) + "\n")
archivo.write("nombre Archivo: "+ __file__ + "\n")
archivo.close()
input("Presiona Enter para terminar...")