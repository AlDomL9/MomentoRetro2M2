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
def train (X, Y, num_params = 1, train_size = 0.8, random_state_split = None, 
           shuffle = True, stratify = None, penalty = "l2", dual = False, 
           tol = 0.001, C = 1.0, fit_intercept = True, intercept_scaling = 1, 
           class_weight = None, random_state_regression = None, 
           solver = "lbfgs", max_iter = 100, multi_class = "auto", verbose = 0, 
           warm_start = False, n_jobs = None, l1_ratio = None):
    
    """
    train
        Entrenamiento y evaluación de regresión logística
    
    Argumentos
        X (numpy array):
            Datos de entrada.
        
        Y (numpy array):
            Valores de salida.
        
        num_params (int):
            Número de parámetros del modelo. x >= 1. Default 1.
        
        train_size (int):
            Tamaño de set de entrenamiento. 1 > x > 0. Default 0.8.
        
        random_state_split (int):
            Establece aleatoriedad. Default = None.
            
        shuffle (bool):
            Establece si se realiza o no una mezcla de los datos
        
        stratify (numpy array):
            Los datos se dividen utilizando estos nombres como las clases. 
            Default = None
        
        penalty (string):
            Tipo de penalty. {'none', 'l1', 'l2', 'elasticnet'}. Default = 'l2'
            
        dual (bool):
            Formulación unica o doble. {Solo impementable con 'l2' y 
            'liblinear'}. Default = False
        
        tol (float):
            Tolerancia para el criterio de alto. Default = 0.001
        
        C (float):
            Inverso de la fuerza de la regularización. x > 0. Default = 1
        
        fit_intercept (bool):
            Establece si se identifica una constante. Default = True
        
        intercept_scaling (float):
            Factor de escalamiento del intercepto. si 'liblinear' y 
            fit_interpcept = True. Default = 1
            
        class_weight (diccionario):
            Establece peso de la clase. Formato {clase : peso} o 'balanced'.
            Default = None
        
        random_state_regression (int):
            Establece si se realiza o no una mezcla de los datos. Si 
            'liblinear', 'sag' o 'saga'. Default = None
        
        solver (string):
            Metodo de resolución para modelo. {'newton-cg' - ['l2', 'none'],
            'lbfgs' - ['l2', 'none'], 'liblinear' . ['l1', 'l2'], 'sag' - [
            'l2', 'none'], 'saga' - ['elasticnet', 'l1', 'l2', 'none']}
            
        max_iter (int):
            Cantidad maxima de iteraciones para encontrar modelo. Default = 100
            
        multi_class (string):
            Condiciones para parametros multiples. {'auto', 'ovr', 
            'multinomial'}. Default = 'auto'
        
        verbose (int):
            Para 'liblinear' y 'lbfgs'. x >= 0. Default = 0.
            
        warm_start (bool):
            Reutiliza solución previa para inizialización. Si 'lbfgs', 
            'newton-cg', 'sag' y 'saga'. Default = False
        
        n_jobs (int):
            Numero de cores del CPU a utilizar. -1 = todos. Default = None = 1
        
        l1_ratio (float):
            Cantidad de l1. Si 'elasticnet'. 1 >= x >= 0. Default = None
        
    Return:
        model(LogisticRegresion()):
            Modelo generado
            
    """
    # Preparar para escalamiento
    if (num_params == 1):
        X = np.reshape(X, (len(X), 1))
    
    # Escalar
    print("Escalando datos")
    scaler = StandardScaler(copy = False)
    scaler.fit_transform(X)
    print("Media encontrada: ", scaler.mean_)
    print("Varianza encontrada: ", scaler.var_)
    
    # Dividir datos en train y test
    print("Dividiendo datos en train y test")
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, 
                                                    train_size = train_size,
                                                    random_state = 
                                                    random_state_split,
                                                    shuffle = shuffle,
                                                    stratify = stratify)
    # Generar modelo
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
    
    # Entrenar modelo
    print("Entrenando")
    model.fit(xTrain, yTrain)
    
    # Mostrar características del modelo
    print("Caracteristicas finales del modelo:")
    print("\tClases identificadas: ", model.classes_)
    print("\tCoeficientes: ", model.coef_)
    print("\tIntercepto: ", model.intercept_)
    print("\tEvaluacion del modelo con datos de entrenamiento: ", 
          model.score(xTrain, yTrain))
    print("\tEvaluacion del modelo con datos de validacion: ", 
          model.score(xTest, yTest))
    
    # Generar matriz de confucion
    print("Generando Confusion-Matrix")
    cm = confusion_matrix(yTest, model.predict(xTest))

    # Mostrar matriz de confusion
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
    plt.show()
    
    # Mostrar reporte de modelo
    print("Evaluacion del modelo: ")
    print(classification_report(yTest, model.predict(xTest)))
    
    return model

#-----------------------------------Pruebas------------------------------------
# Descargar datos
df= pd.read_csv("./Data/iris.csv")
df.drop(axis = 1, columns = "x0", inplace = True)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width",
              "species"]

# Preparar pruebas prediccion
xP = [[-1.90], [-1.002], [0.03], [1.03], [2]]
x2P = [[-1.90, -2], [-1.002, -0.98], [0.03, 0.1], [1.03, 0.80], [2, 1.80]]


#__________________________________Prueba 1____________________________________
# Remplazar resultados a valores 1 o 0
df.replace(to_replace = "Iris-setosa", value = "Setosa", inplace = True)
df.replace(to_replace = "Iris-versicolor", value = "NoSetosa", inplace = True)
df.replace(to_replace = "Iris-virginica", value = "NoSetosa", inplace = True)

# Entrenar modelo
Y = df["species"].values
X = df["petal_length"].values
model1 = train(X, Y)

# Generar predicciones
yP = model1.predict(xP)

# Mostrar predicciones
legends = ["Registros", "Predicciones"]
fig, ax = plt.subplots()
plt.scatter(X, Y, color = "#67CD32", marker = "H")
plt.scatter(xP, yP, color = "#9832CD", marker = "D")
ax.set_xlabel("Largo de petalo (escalado)")
ax.set_ylabel("Especie")
ax.set_title("Setosa si largo de petalo")
ax.legend(legends)
plt.show()


#__________________________________Prueba 2____________________________________
# Entrenar modelo
Y = df["species"].values
X = df[["petal_width","petal_length"]].values
model2 = train(X, Y, num_params = 2)

# Generar predicciones
y2P = model2.predict(x2P)

# Mostrar predicciones
legends = ["Registros", "Predicciones"]
fig, ax = plt.subplots()
plt.scatter(X[:, 0], Y, color = "#2C72D3", marker = "H")
plt.scatter([fila[0] for fila in x2P], y2P, color = "#D38D2C", marker = "D")
ax.set_xlabel("Ancho de petalo (escalado)")
ax.set_ylabel("Especie")
ax.set_title("Setosa si ancho de petalo")
ax.legend(legends)
plt.show()

#__________________________________Prueba 3____________________________________
# Entrenar modelo
Y = df["species"].values
X = df[["sepal_width", "sepal_length"]].values
model3 = train(X, Y, num_params = 2)

# Generar predicciones
y2P = model3.predict(x2P)

# Mostrar predicciones
legends = ["Registros", "Predicciones"]
fig, ax = plt.subplots()
plt.scatter(X[:, 0], Y, color = "#16E956", marker = "H")
plt.scatter([fila[0] for fila in x2P], y2P, color = "#E916A9", marker = "D")
ax.set_xlabel("Ancho de sepalo (escalado)")
ax.set_ylabel("Especie")
ax.set_title("Setosa si ancho de sepalo")
ax.legend(legends)
plt.show()

legends = ["Registros", "Predicciones"]
fig, ax = plt.subplots()
plt.scatter(X[:, 1], Y, color = "#21CBDE", marker = "H")
plt.scatter([fila[1] for fila in x2P], y2P, color = "#DE3421", marker = "D")
ax.set_xlabel("Largo de sepalo (escalado)")
ax.set_ylabel("Especie")
ax.set_title("Setosa si largo de sepalo")
ax.legend(legends)
plt.show()

#__________________________________Prueba 4____________________________________
# Entrenar modelo
model4 = train(df[["sepal_width"]], Y, num_params = 2)

#__________________________________Prueba 5____________________________________
# Entrenar modelo
model5 = train(df[["petal_width", "petal_length", "sepal_width", 
                   "sepal_length"]], Y, num_params = 2)

#__________________________________Prueba 6____________________________________
# Entrenar modelo
model6 = train(df[["sepal_width", "sepal_length"]], Y, num_params = 2, 
               train_size = 0.85, random_state_split = 42, penalty = "l1",
               C = 0.8, random_state_regression = 42, solver = "liblinear",
               max_iter = 200, n_jobs = -1)

#__________________________________Prueba 7____________________________________
# Entrenar modelo
model7 = train(df[["sepal_width", "sepal_length"]], Y, num_params = 2, 
               train_size = 0.70, random_state_split = 42, 
               penalty = "elasticnet", class_weight = "balanced", 
               C = 0.8, random_state_regression = 42, solver = "saga",
               max_iter = 200, warm_start = True, n_jobs = -1,
               l1_ratio = 0.8)

#__________________________________Prueba 8____________________________________
# Entrenar modelo
model8 = train(df[["sepal_width", "sepal_length"]], Y, num_params = 2, 
               train_size = 0.4, shuffle = False, random_state_split = 42, 
               penalty = "none", class_weight = "sepal_width", C = 1, 
               random_state_regression = 42, solver = "sag", max_iter = 1, 
               warm_start = True, n_jobs = -1)

#__________________________________Prueba 9____________________________________
# Volver a descargar datos
df= pd.read_csv("./Data/iris.csv")
df.drop(axis = 1, columns = "x0", inplace = True)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width",
              "species"]

# Entrenar modelo
Y = df["species"].values
X = df["petal_width"].values
model9 = train(X, Y)

# Generar predicciones
yP = model9.predict(xP)

# Mostrar predicciones
legends = ["Registros", "Predicciones"]
fig, ax = plt.subplots()
plt.scatter(X, Y, color = "#17A1E8", marker = "H")
plt.scatter(xP, yP, color = "#E85E17", marker = "D")
ax.set_xlabel("Largo de petalo (escalado)")
ax.set_ylabel("Especie")
ax.set_title("Especie si largo de petalo")
ax.legend(legends)
plt.show()