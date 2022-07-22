# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 19:59:44 2022

MuestreoEtapa4Analisis1.py
Generador de muestras aleatorias representativas.
Ver documento:D:\OperacionesTwitterArticuloSpyder\Etapa4Analisis\Calculo tamaño muestra.xlsx
Evernote:Estad Estadistica Error muestral,nivel de confianza y tamaño de muestras.
incluyen a otros que mencionan al mencionar al top
Se generan archivos como Muestra70-4PA.csv o Muestra385-4PA.csv, esto para cada pais.

@author: Administrador
"""

import pandas as pd
from datetime import datetime as dt
pais="NI"# cambia según país
carpeta='D:\\OperacionesTwitterArticuloSpyder\\Etapa4Analisis\\Datos\\'
print("Inició")
tiempoInicia=dt.now()
df=pd.read_csv(carpeta+'Usuarios4'+pais+'_locdesc.csv', sep=",",encoding='mbcs')#mbcs es ansi. Pero el problema surge cuando
# existen comas en los textos que no pudieron ser filtradas previamente, entonces surgen nuevas columnas por la delimitación de campos
# que provocan los errores al incorporar la información en un formato tabular como pandas.

# Nivel de confianza 90%, error muestral 10%, p=0.5 . Valores cuantitativos, para proporción, población desconocida.
dfrevueltos1=df.sample(frac=1,random_state=1).reset_index(drop=True)# al asignar frac=1, o sea, 100%, devuelve todas
# las filas pero al aplicar aleatoriedad, retornan en un orden diferente aleatoriamente, o sea, se revuelve toda la 
# data. 
dfsample1=dfrevueltos1.sample(70,random_state=1) # de la data revuelta anterior, se extrae la muestra.
dfsample1.to_csv(carpeta+'Muestra70-4'+pais+'.csv',index=False)

# Nivel de confianza 95%, error muestral 5%, p=0.5 . Valores cuantitativos, para proporción, población desconocida.
dfrevueltos2=df.sample(frac=1,random_state=5).reset_index(drop=True)# Importante cambiar el random o se mostrarán todos los de la muestra 385
dfsample2=dfrevueltos2.sample(385,random_state=5)
dfsample2.to_csv(carpeta+'Muestra385-4'+pais+'.csv',index=False)

print("Terminó")
tiempoTermina=dt.now()   
diferencia=tiempoTermina-tiempoInicia
print()
print("Inició el proceso a las = ", tiempoInicia)
print("Terminó el procesar a las= ",tiempoTermina)
print("Diferencia h/m/s:",diferencia)
print()