#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Plantilla de Pre Procesado

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # Selecciona todas las columnas menos la ultima
y = dataset.iloc[:, -1].values # Selecciona la ultima columna


# Tratamiento de los NAs
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean") # Crea un objeto de la clase SimpleImputer . Esta clase se utiliza para imputar valores faltantes en un conjunto de datos.
imputer.fit_transform(X[:, 1:3]) # Ajusta el objeto imputer al subconjunto de X que contiene las columnas de la 1 a la 3 (la 3 no incluida) y, a continuación, transforma ese subconjunto de X imputando los valores faltantes con la media.
X[:, 1:3] = np.round(imputer.transform(X[:, 1:3]), 1) # Se utiliza el objeto imputer para transformar el subconjunto de X que contiene las columnas de la 1 a la 3  imputando los valores faltantes con la media. Luego, se guarda esta transformación en el conjunto de datos original X. Luego lo ajusta para que solo tenga un decimal


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # Convertir las etiquetas categóricas de la primera columna de la matriz X en valores numéricos utilizando el metodo fit_transform() del objeto labelencoder_X.
ct = ColumnTransformer([('Transformacion Pais', OneHotEncoder(), [0])], remainder='passthrough') # se instancia un objeto de la clase ColumnTransformer con el parámetro one_hot_encoder que representa el nombre que se le da a la transformación que se aplicará.. El parámetro remainder especifica lo que debe hacerse con las columnas que no se incluyen en la transformación.
X = ct.fit_transform(X) # se aplica la transformación que se definió anteriormente 

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# X_train, X_test, y_train, y_test = train_test_split(X[:, 3:], y, test_size=0.2, random_state=0)
# train_test_split (Variable independiente, Variable que se quiere predecir, % datos para pruebas (20), semilla para el generador de números aleatorios. )


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # Escalamos el conjunto de entrenamiento 
X_test = sc_X.transform(X_test) # Escala los datos de test con la misma transormacion que haya hecho con los datos de test
# La variable dependiente solo es recomendable escalarla para algoritmos de predicción. En este caso como es de clasificación no sería necesario
