# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 22:52:46 2022
AplicarModelo.py
Se divide en dos partes. 
-Primera parte: Genera el módulo en base a las caratcerísticas y parámetros del mejor resultado 
que tenemos en la tabla resultadosPA385(generada a partir de analisisRandomForest4.py)  en la 
primera fila SELECT  [num],[v0],[f1],[f0],[v1],[cols],[best] 
FROM [Etapa4Analisis].[dbo].[resultadosPA385]  where f1=0 order by v1 desc 
y se usa como data de entrenamiento muestra4PA385paraTablaCompleta.csv que es de 385 registros.

-La segunda parte:aplica el modelo a usuarios4PA.txt y genera el archivo usuarios4PAClasificado.csv
que es lo mismo que usuarios4PA pero con tres columnas más:prob0,prob1 y pred.

@author: Administrador
"""

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV



pd.set_option('expand_frame_repr', False) # despliega todas las columnas
pd.set_option("display.max_rows",None)

pais="NI"

# *********************Segmento 1. Carga del conjunto de datos
dsUsuarios = pd.read_csv('D:\\OperacionesTwitterArticuloSpyder\\Etapa4Analisis\Datos\\muestra4'+pais+'385paraTablaCompleta.csv')

#["author_id","username","created_at","verified",
# "followers","following","tweet_count","listed_count",
# "cant_tweets_muestra","rt","vreplicas","likes","rtquotes",
# "menciones_a_In","menciones_a_Out","menciones_de_In",
# "rt_a_In","rt_a_Out","rt_de_In",
# "rp_a_In","rp_a_Out","rp_de_In",
# "rq_a_In","rq_a_Out","rq_de_In",
# "Actividad","paisSN"]

columnas=["author_id","username","created_at","verified",
         "followers","following","tweet_count","listed_count",
          "cant_tweets_muestra","rt","vreplicas","likes","rtquotes",
          "Actividad",
          "menciones_a_In","menciones_a_Out","menciones_de_In",
          "rt_a_In","rt_a_Out","rt_de_In",
          "rp_a_In","rp_a_Out","rp_de_In",
          "rq_a_Out"
          ]
    
        

dsUsuariosRecortado= dsUsuarios.drop(columns= columnas, axis=1)
datos=dsUsuariosRecortado


cad="datos.drop(columns = 'paisSN'),datos['paisSN'],test_size=0.20,random_state = 123"
X_train, X_test, y_train, y_test = train_test_split(
                                    datos.drop(columns = 'paisSN'),
                                    datos['paisSN'],
                                    test_size=0.20,
                                    random_state = 123,
                                    #stratify=datos['paisSN']
                                )
#{'n_estimators': [10], 'criterion': ['gini'], 'class_weight': ['balanced_subsample']}
param_grid = {'n_estimators': [10],
         
          'criterion'   : ['entropy'] ,#['gini','entropy']
          'class_weight' :['balanced_subsample']# ['balanced','balanced_subsample'],default=None
          }

#GridSearchCV(estimator=RandomForestClassifier(random_state=123),n_jobs=-1,param_grid={'class_weight':['balanced_subsample'],'criterion':['gini'],'n_estimators':[10]})
grid = GridSearchCV(
        estimator  = RandomForestClassifier(random_state = 123),
        param_grid = param_grid,
        n_jobs=-1,
        refit=True,
        return_train_score = True
        )

grid.fit(X = X_train, y = y_train)

   
modelo_final = grid.best_estimator_


predicciones = modelo_final.predict(X = X_test)
predicciones_prob = modelo_final.predict_proba(X = X_test)


dfresult = pd.DataFrame(list(zip(y_test,predicciones,predicciones_prob[:,0],predicciones_prob[:,1])), columns = ['y_test','y_Pred','prob0','prob1'])


mat_confusion = confusion_matrix(
                    y_true    = y_test,
                    y_pred    = predicciones
                )

precision = precision_score(
            y_true    = y_test,
            y_pred    = predicciones
            #normalize = True
           )
print("Matriz de confusión")
print("-------------------")
print(mat_confusion)
print("")
print(f"La precision de test es: {100 * precision} %")


# Aplicar el modelo a toda la data.
dsUsuariosCompleto = pd.read_csv('D:\\OperacionesTwitterArticuloSpyder\\Etapa4Analisis\Datos\\usuarios4'+pais+'.txt')
dsUsuariosCompletoRecortado= dsUsuariosCompleto.drop(columns= columnas, axis=1)

prediccionesCompleto = modelo_final.predict(X = dsUsuariosCompletoRecortado)
prediccionesCompleto_prob = modelo_final.predict_proba(X = dsUsuariosCompletoRecortado)

dsUsuariosClasificado=dsUsuariosCompleto.copy()
dsUsuariosClasificado['pred']=prediccionesCompleto
dsUsuariosClasificado['prob0']=prediccionesCompleto_prob[:,0]
dsUsuariosClasificado['prob1']=prediccionesCompleto_prob[:,1]

dsUsuariosClasificado.to_csv('D:\\OperacionesTwitterArticuloSpyder\\Etapa4Analisis\Datos\\usuarios4'+pais+'Clasificado.csv')
# Por ejemplo usuarios4PAClasificado.csv

'''
SELECT  [num]
      ,[pais]
      ,[v0]
      ,[f1]
      ,[f0]
      ,[v1]
      ,[cols]
      ,[split]
      ,[params]
      ,[grid]
      ,[best]
  FROM [Etapa4Analisis].[dbo].[resultadosPA385]
  where f1=0
  order by v1 desc,f0 asc
163575
'''



















