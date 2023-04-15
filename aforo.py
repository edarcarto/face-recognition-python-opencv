# -*- coding: utf-8 -*-

# instalar librerias
#pip install dlib
#pip install cmake
#pip install face-recognition
#pip install numpy
#pip install opencv-python

# importar librerias
import cv2
import numpy as np
import face_recognition as fr
import os
import random
from datetime import datetime

# Path, fotos del personal
path = "./Personal"
images = []
clases = []
lista = os.listdir(path)

# Para verificar se lista
print(lista)

# Se declara variable de comparación al 100% y contador
comp1 = 100
nombres_registrados = set()
# Se agrega a variables globales las imagenes
for lis in lista:
  imgdb = cv2.imread(f'{path}/{lis}')
  images.append(imgdb)
  clases.append(os.path.splitext(lis)[0])

# Funcion que le da un codigo al rostro
def codrostros(images):
  listacod = []
  for img in images:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cod = fr.face_encodings(img)[0]
    listacod.append(cod)
  return listacod

# Función validar si el personal ya ha sido registrado
def middlewarePersonal(name):
  #si existe el nombre en los registrados
  if name not in nombres_registrados:
    #registro el horario
    nombres_registrados.add(nombre)
    horario(name)

# Funcion para firmar el horario
def horario(nombre):
  with open('horario.csv','a',newline='') as h:
    info = datetime.now()
    fecha = info.strftime('%Y/%m/%d')
    hora = info.strftime('%H:%M:%S')
    h.writelines(f'\n{nombre},{fecha},{hora}')
        

# A la captura del video se asigna un codigo RGB
rostroscod = codrostros(images)

# Inicializa la camara
cap = cv2.VideoCapture(0)

# Mientras la camara está abierta
while cap.isOpened():
  
  # Se obtiene un frame 
  ret, frame = cap.read()

  # Se reduce la imagen para que sea más pequeña y no consuma memoria
  frame2 = cv2.resize(frame,(0,0), None, 0.25, 0.25)
  rgb = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
  faces = fr.face_locations(rgb)
  facescod = fr.face_encodings(rgb, faces)

  # recorrer los objetos encontrados
  for facecod, faceloc in zip(facescod,faces):
    
    # Busca y compara
    comparacion = fr.compare_faces(rostroscod,facecod)

    # Busca similitudes
    similitud = fr.face_distance(rostroscod,facecod)
    min = np.argmin(similitud)
    
    # si encuentra similitud
    if comparacion[min]:

      # Cambia el nombre a mayuscula
      nombre = clases[min].upper()
      
      # print(clases[min])

      # Se ubica coodenadas
      yi, xf, yf, xi = faceloc
      yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4
      indice = comparacion.index(True)
      
      # Se dibuja un rectangulo
      if comp1 != indice:
        r = random.randrange(0,255,50)
        g = random.randrange(0,255,50)
        b = random.randrange(0,255,50)
        comp1 = indice
      if comp1 == indice:
        # si hay similitud se agrega horario y dibuja un rectangulo
        cv2.rectangle(frame, (xi,yi), (xf,yf), (r,g,b), 3)
        cv2.rectangle(frame, (xi,yf - 35), (xf,yf), (r,g,b), cv2.FILLED)
        cv2.putText(frame, nombre, (xi + 6, yf - 6),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        middlewarePersonal(nombre)

  # Titulo de la ventana
  cv2.imshow('Reconocimiento Facial',frame)
  
  # si se presiona ESC se detiene la app
  t = cv2.waitKey(5)
  if t == 27:
    break