#Producción de energía eléctrica de una central eléctrica  de ciclo combinado
#En este proyecto construiremos un modelo para predecir la producción de energía eléctrica de una central eléctrica  de ciclo combinado, que utiliza una combinación de turbinas de gas, turbinas de vapor y generadores de vapor con recuperación de calor para generar energía.  Disponemos de un conjunto de 9568 lecturas ambientales medias horarias procedentes de sensores de la central eléctrica que utilizaremos en nuestro modelo.
#Las columnas de los datos consisten en variables ambientales medias horarias:
#- Temperatura (AT) en el rango de 1,81°C a 37,11°C,
#- Presión ambiente (AP) en el rango 992,89-1033,30 milibares,
#- Humedad relativa (RH) en la gama de 25,56% a 100,16%
#- Vacío de escape (V) en el rango 25,36-81,56 cm Hg
#- Producción horaria neta de energía eléctrica (PE) 420,26-495,76 MW (Objetivo que intentamos predecir)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Carga de datos
data = pd.read_csv("CCPP_data.csv") 
X = data[["AT", "V", "AP", "RH"]]
y = data["PE"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicción y métricas
y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")