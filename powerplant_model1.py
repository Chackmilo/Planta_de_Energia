import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Carga y preparación de datos
try:
    data = pd.read_csv("CCPP_data.csv")
except FileNotFoundError:
    print("Error: No se encontró el archivo CCPP_data.csv")
    exit()

# Separar features y target
X = data[["AT", "V", "AP", "RH"]]
y = data["PE"]

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Búsqueda de mejores hiperparámetros
#reduciedos en una dimension 300, 30, 10, 4
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,#de 5 a 3
    n_jobs=-1,
    scoring='neg_root_mean_squared_error',
    verbose=1
)

# Entrenamiento con búsqueda de hiperparámetros
grid_search.fit(X_train, y_train)

# Mejores parámetros e importancia de características
print("\nMejores parámetros encontrados:")
print(grid_search.best_params_)

# Usar el mejor modelo
best_model = grid_search.best_estimator_

# Predicciones
y_pred = best_model.predict(X_test)

# Métricas de evaluación
print("\nMétricas de evaluación:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Validación cruzada
cv_scores = cross_val_score(
    best_model, X_scaled, y, 
    cv=5, 
    scoring='neg_root_mean_squared_error'
)
print("\nResultados de validación cruzada:")
print(f"RMSE medio: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Importancia de características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Visualización de la importancia de características
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Importancia de las Características')
plt.tight_layout()
plt.show()

# Gráfico de predicciones vs valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.tight_layout()
plt.show()
