"""
Regresion Logistica
    Implementacion del algoritmo de regresion logistica con uso de bibliotecas
    de aprendizaje.
    
Autor:
    Alejandro Domi­nguez Lugo
    A01378028
    
Fecha:
    08 de septiembre de 2022
    
"""

#----------------------------------Libreri­as-----------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#-----------------------------Variables Globales-------------------------------

#------------------------------------Main--------------------------------------
def train (X, Y, train_size = 0.8, random_state_split = None, shuffle = True, 
           stratify = None, penalty = "l2", dual = False, tol = 0.001, C = 1.0, 
           fit_intercept = True, intercept_scaling = 1, class_weight = None, 
           random_state_regression = None, solver = "lbfgs", max_iter = 100, 
           multi_class = "auto", verbose = 0, warm_start = False, 
           n_jobs = None,l1_ratio = None):
    
    print("Escalando datos")
    scaler = StandardScaler(copy = False)
    scaler.fit_transform(X)
    print("Media encontrada: ", scaler.mean_)
    print("Varianza encontrada: ", scaler.var_)
    
    print("Dividiendo datos en train y test")
    xTrain, yTrain, xTest, yTest = train_test_split(X, Y, 
                                                    train_size = train_size,
                                                    random_state_split = 
                                                    random_state_split,
                                                    shuffle = shuffle,
                                                    stratify = stratify)
    
    print("Preparando modelo")
    model = LogisticRegression(penalty = penalty, dual = dual, tol = tol, 
                               C = C, fit_intercept = fit_intercept,
                               intercept_scaling = intercept_scaling,
                               class_weight = class_weight, 
                               random_state = random_state_regression,
                               solver = solver, max_iter = max_iter, 
                               multi_class = multi_class, verbose = verbose,
                               warm_start = warm_start, n_jobs = n_jobs,
                               l1_ratio = l1_ratio)
    
    print("Entrenando")
    model.fit(xTrain, yTrain)
    
    print("Caracteristicas finales del modelo:")
    print("\tClases identificadas: ", model.classes_)
    print("\tCoeficientes: ", model.coef_)
    print("\tIntercepto: ", model.intercept_)
    print("\tEvaluacion del modelo con datos de entrenamiento: ", 
          model.score(xTrain, yTrain))
    print("\tEvaluacion del modelo con datos de validacion: ", 
          model.score(xTest, yTest))
    
    print("Generando Confusion-Matrix")
    cm = confusion_matrix(yTest, model.predict(xTest))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
    plt.show()
    
    print("Evaluacion del modelo: ")
    print(classification_report(yTest, model.predict(xTest)))
    
    return model

#-----------------------------------Pruebas------------------------------------
df = pd.read_csv("/home/lex/Escritorio/MomentoRetroM2/MomentoRetro2M2/iris.csv")