# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 01:08:25 2023

@author: David
"""

import pandas as pd

# Importar el data set
df = pd.read_csv('Data.csv')

# Seleccionar todas las columnas
todas = df.iloc[:, :]

# Seleccionar todas menos la última columna
todas_menos_ultima = df.iloc[:, :-1]

# Seleccionar la ultima columna
ultimas_tres = df.iloc[:, -1]

# Seleccionar todas menos la antepenúltima columna
todas_menos_antepenultima = df.iloc[:, :-3].join(df.iloc[:, -2:])

# Seleccionar las 3 últimas columnas
ultimas_tres = df.iloc[:, -3:]

# Seleccionar las 3 primeras columnas
primeras_tres = df.iloc[:, :3]

# Seleccionar de la tercera a la quinta columna
tercera_quinta = df.iloc[:, 2:5]

# Seleccionar la segunda y la cuarta columna
segunda_cuarta = df.iloc[:, [1, 3]]

# Seleccionar la séptima columna
septima = df.iloc[:, 6]