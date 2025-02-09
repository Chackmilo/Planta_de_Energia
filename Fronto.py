# -*- coding: utf-8 -*-
"""
Evaluación Comparativa de Modelos de ML para Predicción de Producción Energética
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =============================================================================
# 1. Carga y Preparación de Datos
# =============================================================================
try:
    data = pd.read_csv("CCPP_data.csv")
except FileNotFoundError:
    print("Error: Archivo no encontrado")
    exit()

X = data[["AT", "V", "AP", "RH"]]
y = data["PE"]

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# =============================================================================
# 2. Definición de Modelos y Parámetros
# =============================================================================
models = {
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(objective='reg:squarederror', random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
    },
    'Red Neuronal': {
        'model': MLPRegressor(max_iter=1000, random_state=42),
        'params': {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam']
        }
    },
    'Regresión Lineal': {
        'model': LinearRegression(),
        'params': {}
    }
}

# =============================================================================
# 3. Entrenamiento y Evaluación de Modelos
# =============================================================================
results = []
best_models = {}

for name, config in models.items():
    print(f"\nEvaluando {name}...")
    
    # Búsqueda de hiperparámetros
    grid = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    
    # Predicciones y métricas
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Validación cruzada
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
    
    results.append({
        'Modelo': name,
        'RMSE': rmse,
        'R²': r2,
        'RMSE CV': f"{-cv_scores.mean():.2f} ± {cv_scores.std():.2f}",
        'Mejores Parámetros': grid.best_params_
    })

# =============================================================================
# 4. Análisis Comparativo
# =============================================================================
# Crear DataFrame con resultados
results_df = pd.DataFrame(results)

# Tabla comparativa
print("\n" + "="*80)
print("Comparación de Modelos".center(80))
print("="*80)
print(results_df[['Modelo', 'RMSE', 'R²', 'RMSE CV']].to_string(index=False))

# Gráfico de comparación de RMSE
plt.figure(figsize=(12, 6))
sns.barplot(x='RMSE', y='Modelo', data=results_df.sort_values('RMSE'))
plt.title('Comparación de RMSE entre Modelos', fontsize=14)
plt.xlabel('RMSE (MW)', fontsize=12)
plt.ylabel('')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Gráfico de importancia de características (para modelos basados en árboles)
for name, model in best_models.items():
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Característica': X.columns,
            'Importancia': model.feature_importances_
        }).sort_values('Importancia', ascending=False)
        
        plt.figure(figsize=(10, 4))
        sns.barplot(x='Importancia', y='Característica', data=importance)
        plt.title(f'Importancia de Características - {name}', fontsize=14)
        plt.xlabel('Importancia Relativa', fontsize=12)
        plt.ylabel('')
        plt.tight_layout()
        plt.show()

# =============================================================================
# 5. Selección del Mejor Modelo
# =============================================================================
best_model_name = results_df.loc[results_df['RMSE'].idxmin(), 'Modelo']
best_model = best_models[best_model_name]

print("\n" + "="*80)
print(f"Mejor Modelo Seleccionado: {best_model_name}".center(80))
print("="*80)
print(f"Parámetros: {results_df.loc[results_df['RMSE'].idxmin(), 'Mejores Parámetros']}")
print(f"RMSE en Test: {results_df.loc[results_df['RMSE'].idxmin(), 'RMSE']:.2f}")
print(f"R² en Test: {results_df.loc[results_df['RMSE'].idxmin(), 'R²']:.2f}")

# Guardar mejor modelo
joblib.dump(best_model, 'mejor_modelo.pkl')
print("\nModelo guardado como: mejor_modelo.pkl")

# =============================================================================
# 6. Ejemplo de Predicción
# =============================================================================
sample_data = pd.DataFrame([[27.5, 62.5, 1013.2, 68.7]], columns=X.columns)
sample_scaled = scaler.transform(sample_data)
prediccion = best_model.predict(sample_scaled)
print(f"\nPredicción para muestra:\nEntrada: {sample_data.values[0]}\nProducción Estimada: {prediccion[0]:.2f} MW")