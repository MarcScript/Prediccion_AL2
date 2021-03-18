#Importamos las librerias de pandas,numpy y sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from imblearn.under_sampling import NearMiss 
from math import ceil
 #definimos funcion para mostrar los resultados
def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=['Predic_Pasa','Predic_NoPasa'], yticklabels=['Pasó','No Pasó'], annot=True, fmt="d");
    plt.title("Confusion matrix balanceado")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
def tabla_frecuencia_antes(Y):
    count_classes = pd.value_counts(Y['Aprobado'], sort = True)
    count_classes.plot(kind = 'bar', rot=0)
    plt.xticks(range(2))
    plt.title("Tabla de frecuencia de Aprobados Antes del Sampling")
    plt.xlabel("Aprobación(0: No aprobó, 1:Aprobó)")
    plt.ylabel("Nro de estudiantes");
def tabla_frecuencia_despues(Y_dps):
    count_classes2 = pd.value_counts(Y_dps['Aprobado'], sort = True)
    count_classes2.plot(kind = 'bar', rot=0)
    plt.xticks(range(2))
    plt.title("Tabla de frecuencia de Aprobados Despues del Sampling")
    plt.xlabel("Aprobación(0: No aprobó, 1:Aprobó)")
    plt.ylabel("Nro de estudiantes");
#Importamos nuestra base de datos
url = 'https://raw.githubusercontent.com/diegostaPy/cursoIA/main/datosRendimiento/datosfiltrados.csv'
df = pd.read_csv(url)
dfcopia = df.copy()

dfcopia= dfcopia[dfcopia['Asignatura']=='ALGEBRA LINEAL 2']

#dfcopia=dfcopia.set_index('id_anony')
#cols= ['Convocatoria','Anho','Aprobado','Anho.Firma','Primer.Par','Segundo.Par','AOT']
cols= ['Aprobado','Primer.Par','Segundo.Par']
dfcopia=dfcopia[cols]
dfcopia['Aprobado']=dfcopia['Aprobado'].replace(['S', 'N'],['1','0'])
X = dfcopia[['Primer.Par','Segundo.Par']]
Y = dfcopia[['Aprobado']]


print(Y.Aprobado.value_counts().sort_index())
#Separamos los datos para entrenamiento y test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state = 1234,shuffle = True)
#Grafico de aprobados vs no aprobados
tabla_frecuencia_antes(Y_test)

#Se escalan los datos
print(Y_train.Aprobado.value_counts().sort_index())
algoritmo = LogisticRegression()
#Entrenamos el modelo


us = NearMiss(n_neighbors=3, version=2)
X_train_res, Y_train_res = us.fit_resample(X_train, Y_train)

print ("Distribucion antes del resampling {}".format(Counter(Y_train['Aprobado'])))
print ("Distribucion despues del resampling {}".format(Counter(Y_train_res['Aprobado'])))
tabla_frecuencia_despues(Y_train_res)
algoritmo.fit(X_train_res,Y_train_res)
#Realizamos la prediccion
Y_pred = algoritmo.predict(X_test)

#Imprimimos la matriz de confusion
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(Y_test,Y_pred)
print(matriz)
#Calculamos la precision del modelo
print('Score del modelo')
print(algoritmo.score(X_train,Y_train))
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_pred, average="binary", pos_label='1') 
print('Precisión del modelo:')
print(precision)
#Calculo la exactitud del modelo
from sklearn.metrics import accuracy_score
exactitud = accuracy_score(Y_test, Y_pred)
print('Exactitud del modelo:')
print(exactitud)
#Calculo la sensibilidad del modelo
from sklearn.metrics import recall_score
sensibilidad = recall_score(Y_test, Y_pred, pos_label='1')
print('Sensibilidad del modelo:')
print(sensibilidad)
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))
mostrar_resultados(Y_test,Y_pred)
#Mostramos el nro de secciones predicho por el modelo
print('Numero de secciones que necesitaremos abrir para la correlativa:')
secc = (Y_pred[Y_pred == '1'].size)/60
print(ceil(secc))
print('Numero de secciones que necesitaremos abrir para la gente que no paso:')
secc = (Y_pred[Y_pred == '0'].size)/60
print(ceil(secc))

X_new = pd.DataFrame({'Primer.Par':[0] , 'Segundo.Par': [15]})
pasa = algoritmo.predict(X_new)

if(pasa[0] == 1):
 print('\nPasas la materia')
else:
 print('\nNo pasas')
 
 


 