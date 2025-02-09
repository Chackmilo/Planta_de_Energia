# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Cargar modelos y escalador
models = {
    'Random Forest': joblib.load('./models/random_forest_model.pkl'),
    'XGBoost': joblib.load('./models/xgboost_model.pkl'),
    'SVR': joblib.load('./models/svr_model.pkl'),
    'Linear Regression': joblib.load('./models/linear_regression_model.pkl'),
    'Red Neuronal': joblib.load('./models/mlp_model.pkl')
}

scaler = joblib.load('./models/scaler.pkl')

@app.route('/')
def home():
    return render_template('comparison.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([[
            float(data['AT']),
            float(data['V']),
            float(data['AP']),
            float(data['RH'])
        ]], columns=["AT", "V", "AP", "RH"])
        
        # Escalar datos
        scaled_data = scaler.transform(input_data)
        
        # Hacer predicciones con todos los modelos
        predictions = {}
        for name, model in models.items():
            predictions[name] = round(model.predict(scaled_data)[0], 2)
        
        # Generar gráficas comparativas
        plot_urls = generate_comparison_plots(predictions)
        
        return jsonify({
            'predictions': predictions,
            'plots': plot_urls
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

def generate_comparison_plots(predictions):
    # Gráfico de barras comparativo
    plt.figure(figsize=(10, 6))
    names = list(predictions.keys())
    values = list(predictions.values())
    
    plt.bar(names, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#E91E63'])
    plt.title('Comparación de Predicciones por Modelo')
    plt.ylabel('Producción (MW)')
    plt.ylim(min(values)-5, max(values)+5)
    
    buf1 = BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    plot1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close()
    
    # Gráfico de importancia de características (usando Random Forest como ejemplo)
    importance = models['Random Forest'].feature_importances_
    features = ["AT", "V", "AP", "RH"]
    
    plt.figure(figsize=(8, 8))
    plt.pie(importance, labels=features, autopct='%1.1f%%', startangle=140)
    plt.title('Importancia de Características (Random Forest)')
    
    buf2 = BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    plot2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close()
    
    return {
        'predictions_chart': plot1,
        'feature_importance': plot2
    }

if __name__ == '__main__':
    app.run(debug=True)
    