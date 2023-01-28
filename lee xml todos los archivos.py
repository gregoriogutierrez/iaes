#lee xml todos los archivos
#lee lista de archivos del directorio
def leeUnArchivoXml(nombreArchivo):
    global fileListaFaltas
    import xml.etree.ElementTree as ET
    nombreInstituto = "IES Sierra blanca"
    grupo = ""
    apellido1 = ""
    apellido2 = ""
    nombreAlumno = ""
    fechaFalta = ""
    horaFalta = ""
    tipoFalta = ""#justificado o injustificado o retraso 
    diccionarioX_TRAMO = {}
    def factorial(root,n):
        nonlocal grupo, diccionarioX_TRAMO, apellido1, apellido2, nombreAlumno, fechaFalta, horaFalta, tipoFalta
        global fileListaFaltas, cuentaLineas
        if n == 0:
            return
        else:
            for child in root:
                if (child is None):
                    return
                tabuladoresRepetidos="\t"*(profundidad-n)
                ##file.write(tabuladoresRepetidos + str(child.tag) + str(child.attrib)+str(child.text)+'\n')
                if ("X_TRAMO"==str(child.tag)):
                    xTramo=str(child.text)
                if ("T_HORCEN"==str(child.tag)):
                    diccionarioX_TRAMO[xTramo]=str(child.text)
                if ("T_NOMBRE"==str(child.tag)):
                    grupo=str(child.text)
                if ("T_APELLIDO1"==str(child.tag)):
                    apellido1=str(child.text)
                if ("T_APELLIDO2"==str(child.tag)):
                    apellido2=str(child.text)
                if ("T_NOMBRE"==str(child.tag)):
                    nombreAlumno=str(child.text)
                if("F_FALASI"==str(child.tag)):
                    fechaFalta=str(child.text)
                if((fechaFalta!="")and("X_TRAMO"==str(child.tag))and(str(child.text) in diccionarioX_TRAMO)):
                        horaFalta=diccionarioX_TRAMO[str(child.text)]
                if("C_TIPFAL")==str(child.tag):
                    tipoFalta=str(child.text)
                if("L_DIACOM")==str(child.tag):
                    if (child.text=="N"):
                        cuentaLineas+=1
                        fileListaFaltas.write(str(cuentaLineas) +";"+ nombreInstituto+";"+grupo+";"+apellido1+";"+apellido2+";"+nombreAlumno+";"+fechaFalta+";"+horaFalta+";"+tipoFalta+";"+"\n")
                    if (child.text=="S"):
                        for xTramo in diccionarioX_TRAMO:
                            if ((diccionarioX_TRAMO[xTramo]<"7ª")and(diccionarioX_TRAMO[xTramo]!="10ª hora")and(diccionarioX_TRAMO[xTramo]!="11ª hora")):
                                fileListaFaltas.write(str(cuentaLineas) +";"+ nombreInstituto+";"+grupo+";"+apellido1+";"+apellido2+";"+nombreAlumno+";"+fechaFalta+";"+diccionarioX_TRAMO[xTramo]+";"+tipoFalta+";"+"\n")
                factorial(child,n-1)
    tree = ET.parse('datos.xml')
    root = tree.getroot()
    profundidad = 10
    ##with open('faltas.txt', 'w') as file:
    factorial(root,profundidad)
    ##file.close() 
    print("fin "+nombreArchivo)
import os
from os import listdir
from os.path import isfile, join
#mypath = "/home/usuario/Documentos/faltas/"
mypath = ""
onlyfiles = [f for f in listdir() if (isfile(join(mypath, f))and f.endswith(".xml"))]
print(onlyfiles)
fileListaFaltas = open('listafaltas.txt', 'w') 
cuentaLineas=0
for cadaArchivo in onlyfiles:
    leeUnArchivoXml(cadaArchivo)
fileListaFaltas.close()
