# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 09:43:00 2022
analisisRandomForest4
@author: Administrador
"""

import pandas as pd
import numpy as np
from datetime import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sys
import warnings
warnings.filterwarnings("ignore")

pd.set_option('expand_frame_repr', False) # despliega todas las columnas
pd.set_option("display.max_rows",None)
pais="NI" # la única información que se modifica en el script de forma manual




# CONSTRUCCIÓN DE LAS ACCIONES 4096

menciones=[
        ["","",""],
        ["","menciones_a_Out",""],
        ["","menciones_a_Out","menciones_de_In"],
        ["","","menciones_de_In"],
        ["menciones_a_In","",""],
        ["menciones_a_In","","menciones_de_In"],
        ["menciones_a_In","menciones_a_Out",""],
        ["menciones_a_In","menciones_a_Out","menciones_de_In"]
    
    ]

retweets=[
        ["","",""],
        ["","rt_a_Out",""],
        ["","rt_a_Out","rt_de_In"],
        ["","","rt_de_In"],
        ["rt_a_In","",""],
        ["rt_a_In","","rt_de_In"],
        ["rt_a_In","rt_a_Out",""],
        ["rt_a_In","rt_a_Out","rt_de_In"]
    
    ]

replicas=[
        ["","",""],
        ["","rp_a_Out",""],
        ["","rp_a_Out","rp_de_In"],
        ["","","rp_de_In"],
        ["rp_a_In","",""],
        ["rp_a_In","","rp_de_In"],
        ["rp_a_In","rp_a_Out",""],
        ["rp_a_In","rp_a_Out","rp_de_In"]
    
    ]

rquotes=[
        ["","",""],
        ["","rq_a_Out",""],
        ["","rq_a_Out","rq_de_In"],
        ["","","rq_de_In"],
        ["rq_a_In","",""],
        ["rq_a_In","","rq_de_In"],
        ["rq_a_In","rq_a_Out",""],
        ["rq_a_In","rq_a_Out","rq_de_In"]
    
    ]


cadmenciones=""
cadrt=""
cadrp=""
cadrq=""
lista=[]
cont=0
cada1000=0
for m in menciones:
    cadmenciones=""
    if m[0]!="":
        cadmenciones=cadmenciones+m[0]+","
        
    if m[1]!="":
        cadmenciones=cadmenciones+m[1]+","
       
    if m[2]!="":
        cadmenciones=cadmenciones+m[2]+","
        
    # print(cadena[:-1])    
    # print()
    # cadena=""
    for rt in retweets:
        cadrt=cadmenciones
        if rt[0]!="":
            cadrt=cadrt+rt[0]+","
        if rt[1]!="":
            cadrt=cadrt+rt[1]+","
        if rt[2]!="":
            cadrt=cadrt+rt[2]+","
        for rp in replicas:
            cadrp=cadrt
            if rp[0]!="":
                cadrp=cadrp+rp[0]+","
            if rp[1]!="":
                cadrp=cadrp+rp[1]+","
            if rp[2]!="":
                cadrp=cadrp+rp[2]+","
            for rq in rquotes:
                cadrq=cadrp
                if rq[0]!="":
                    cadrq=cadrq+rq[0]+","
                if rq[1]!="":
                    cadrq=cadrq+rq[1]+","
                if rq[2]!="":
                    cadrq=cadrq+rq[2]+","
                lista.append(cadrq[:-1])
             
                cont=cont+1
                #print("No:",cont)

dfrelaciones=pd.DataFrame(lista,columns=["columnas"])      
# 4096 registros 

# CONSTRUCCIÓN DE EXTERNAS, INTERNAS Y ACTIVIDA Y UNIÓN CON ACCIONES
metricas_externas=[[],["followers","following","tweet_count","listed_count"]]
 
metricas_internas=[[],["cant_tweets_muestra","rt","vreplicas","likes","rtquotes"]]

actividad=["","Actividad"]
               
   
cadme=""
cadmi=""
cada=""
cadcrit=""
cadbal=""
cadrel=""    
lista2=[]
cont=0
for me in metricas_externas:
    cadme=""
    if len(me)!=0:
        cadme=cadme+','.join(me)+","
    #print("me:",me," cadme:",cadme)    
    for mi in metricas_internas:
        cadmi=cadme
        if len(mi)!=0:
            cadmi=cadmi+','.join(mi)+","
        #print("mi:",mi," cadmi:",cadmi)
        for a in actividad:
            cada=cadmi
            if a!="":
                cada=cada+a+","
            #print("cada:",cada)        
            for i in dfrelaciones.itertuples():
                cadrel=cada
                if i[1]!="":
                    cadrel=cadrel+i[1]+","
                lista2.append(cadrel[:-1])
                #print("cadrel:",cadrel[:-1])
                cont=cont+1
                print("No:",cont)
                print()

dfcolumnas=pd.DataFrame(lista2,columns=["columnas"])      
# dfcolumnas.to_clipboard()  
# 32,768 registros.
# c=0
# for i in dfcolumnas.itertuples():
#     c=c+1
#     print(i[0])
# print("Total:", c)

# PROCESAMIENTO INLCUYENDO LOS PARAMS DE ARBOL Y GRID
print()
print("----------------------------------------------")
print()
tiempoInicia=dt.now()
print("Pais:",pais)
print("Inició procesamiento por árboles: ",tiempoInicia)
print("Procesando.")


# *********************Segmento 1. Carga del conjunto de datos
dsUsuarios = pd.read_csv('D:\\OperacionesTwitterArticuloSpyder\\Etapa4Analisis\Datos\\muestra4'+pais+'385paraTablaCompleta.csv')
dsUsuarios=dsUsuarios.drop(columns= ["author_id","created_at","username","verified"], axis=1)
dsUsuariosRecortado=[]
columnas=[]
cont=0
param_grid={}
criterio=[]
balance=[]
lista=[]
columnasRetiradas=""
errores=[]
cont_error=0
giro=0
contfor=0
inicioGiro=dt.now()
duraciongiros=[]
for crit in [1,2]:#criterio: 1 es 'gini',2 es 'entropy'
    
    for bal in [0,1,2]:#balance:0 ninguno, 1 balance, 2  balanced_subsample

        for i in dfcolumnas.itertuples():# i contiene las columnas que se deben retirar del dataframe , concretamente en i[1]
            contfor=contfor+1
            if contfor==32768:
                giro=giro+1
                contfor=0
                print("--------------------------")
                print("Entra a nuevo Giro.",giro)
                print("Informe de Giro.")
                print("crit:",crit," bal:",bal," Giro:",giro," No:",cont)
                print("Errores:",cont_error)
                finGiro=dt.now()   
                diferencia=finGiro-inicioGiro
                print("Tiempo transcurrido :",diferencia, " del giro anterior:",giro-1)
                print("Total de registros:",cont+1)
                duraciongiros.append([giro-1,diferencia])
                print("Duración giros: no giro, duración")
                print(duraciongiros)
                inicioGiro=finGiro
                print()
                
            dsUsuariosRecortado=dsUsuarios.copy()
            #cont=cont+1
            if i[1]!="":# devuelve el contenido, un string
                columnasRetiradas=i[1] # esto es solo para el final del script a la hora de registrar qué columnas se usaron y no se usaron
                datos= dsUsuariosRecortado.drop(columns=i[1].split(","), axis=1)#i[1].split(",") convierte a una lista
                
            else:
                columnasRetiradas="Nope" # esto es solo para el final del script a la hora de registrar qué columnas se usaron y no se usaron
                datos=dsUsuariosRecortado
            
            cad="datos.drop(columns = 'paisSN'),datos['paisSN'],test_size=0.20,random_state = 123"
            X_train, X_test, y_train, y_test = train_test_split(
                                                datos.drop(columns = 'paisSN'),
                                                datos['paisSN'],
                                                test_size=0.20,
                                                random_state = 123,
                                                #stratify=datos['paisSN']
                                            )
        
            if crit==1:
                criterio=['entropy'] 
            else:
                criterio=['gini'] 
            
            if bal==0:
                param_grid = {'n_estimators': [10],
                       'max_features': [3],
                       #'max_depth'   : [None],
                       'criterion'   : criterio
                     
                       }
            else:
                if bal==1:
                    balance=['balanced_subsample']
                else:
                    balance=['balanced']
                param_grid = {'n_estimators': [10],
                       #'max_features': [3],If None, then max_features=n_features
                       #'max_depth'   : [None],
                       'criterion'   : criterio,
                       'class_weight' :balance # ['balanced','balanced_subsample'],default=None
                       }
        
        
            grid = GridSearchCV(
                    estimator  = RandomForestClassifier(random_state = 123),
                    param_grid = param_grid,
                    #verbose    = 10,
                    n_jobs=-1,
                    refit=True
                    #return_train_score = True
                    )
            
            
            try:
                grid.fit(X = X_train, y = y_train)
            except:
                cont_error=cont_error+1
                print("Falló en fit línea 258.")
                print("Columnas retiradas:",columnasRetiradas)
                print("No anterior:",cont)
                errores.append([cont_error,cont,columnasRetiradas])
                print()
                continue # guarda el error y los datos que les corresponde y continúa con el siguiente ciclo del for
               
            modelo_final = grid.best_estimator_
        
        
            predicciones = modelo_final.predict(X = X_test)
            # predicciones_prob = modelo_final.predict_proba(X = X_test)
        
        
            # dfresult = pd.DataFrame(list(zip(y_test,predicciones,predicciones_prob[:,0],predicciones_prob[:,1])), columns = ['y_test','y_Pred','prob0','prob1'])
        
        
            mat_confusion = confusion_matrix(
                                y_true    = y_test,
                                y_pred    = predicciones
                            )
            
            precision = precision_score(
                        y_true    = y_test,
                        y_pred    = predicciones
                        #normalize = True
                        )
            
            strparamgrid=str(param_grid)
            strgrid=str(grid).replace('\n','').replace(' ','')
            strbest=str(grid.best_params_)
            reg=[pais,mat_confusion[0,0],mat_confusion[0,1], mat_confusion[1,0],mat_confusion[1,1],
                   columnasRetiradas,cad,strparamgrid,strgrid,strbest]
            cont=cont+1
            cada1000=cada1000+1
            
            lista.append(reg)# se agregan aquí los 196,608 registros que al final se convierten en datafram para convertir a csv
            if cada1000==1000:
                print("Inició:",tiempoInicia)
                print("Pais:",pais)
                print("crit:",crit," bal:",bal,"Giro:",giro," No:",cont)
                print("Iteración for:",contfor)
                tiempoahora=dt.now()   
                diferencia=tiempoahora-tiempoInicia
                print("Tiempo transcurrido:",diferencia)
                print("Procesando.")
                print("No. ",cont)
                print("Errores:",cont_error)
                print("Duración giros: no giro, duración")
                print(duraciongiros)
                cada1000=0
                print()
   
print()
print("-------------------------------")
print("Terminación del procesamiento, fase anterior a la descarga de resultados en csv.")
print("Inició:",tiempoInicia)
print("Pais:",pais)
tiempoahora=dt.now()   
diferencia=tiempoahora-tiempoInicia
print("Tiempo transcurrido:",diferencia)
print("Total de registros:",cont)
print("Pais:",pais)
print()
              
# Se copian los resultados en archivo csv
print()
print("-------------------------------")
tiempoIniciaCarga=dt.now()
print("Inició cargar en csv:",tiempoIniciaCarga)
print("Pais:",pais)
print("Duración giros: no giro, duración")
print(duraciongiros)
dsfinal=pd.DataFrame(lista,columns=['pais','v0','f1','f0','v1','cols','split','params','grid','best'])   
tiempoahora=dt.now()  
diferencia=tiempoahora-tiempoIniciaCarga
print("Tiempo transcurrido de carga:",diferencia)
print("Pais:",pais)
#dsfinal.to_clipboard()
dsfinal.to_csv('D:\\OperacionesTwitterArticuloSpyder\\Etapa4Analisis\Datos\\resultados'+pais+'385.csv')

# 196,608 registros     
    
print()
print("-------------------------------")
print("Terminó tod el procesamiento del Script.")
print("Pais:",pais)
print("No. ",cont)
print("Errores:",cont_error)
print("Duración giros: no giro, duración")
print(duraciongiros)
tiempoTermina=dt.now()   
diferencia=tiempoTermina-tiempoInicia
print()
print("Inició el proceso a las = ", tiempoInicia)
print("Terminó todo el proceso a las= ",tiempoTermina)
print("Diferencia h/m/s:",diferencia)
print()      



