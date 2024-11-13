import streamlit as st
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import DMatrix

# Configurar la página en modo amplio
st.set_page_config(layout="wide")

# Cargar el modelo, métricas y curvas desde el archivo .pkl
def cargar_modelo_y_métricas():
    with open("metricas_y_modelo_rfm.pkl", "rb") as f:
        model_records = pickle.load(f)
    return model_records

# Cargar el diccionario que contiene el modelo y las métricas
model_records = cargar_modelo_y_métricas()
modelo = model_records['best_instance']  # El modelo preentrenado
metricas = model_records  # Todas las métricas y datos

# Asegurarse de que el conjunto de validación tiene las columnas correctas
X_valid = pd.DataFrame(model_records['X_valid'], columns=model_records['feature_names'])
X_valid_dmatrix = DMatrix(X_valid)  # Convertir a DMatrix con las columnas correctas
y_valid = model_records['y_valid']

# Mostrar métricas y gráficos en el dashboard
st.title("Dashboard de Modelo Clasificador con RFM")
st.write("Este dashboard permite ver las métricas de rendimiento del modelo preentrenado y realizar predicciones.")

# Métricas del modelo
st.header("Métricas del Modelo")
st.write("**Mejor AUC en Validación Cruzada**:", metricas['best_score_auc'])
st.write("**Promedio AUC de Validación Cruzada**:", np.mean(metricas['auc_scores']))
st.write("**Promedio de Precisión (Accuracy) en Validación Cruzada**:", np.mean(metricas['accuracy_scores']))

# Reporte de Clasificación
st.write("**Reporte de Clasificación del Mejor Modelo:**")
st.write(metricas['classification_report'])
st.write("**AUC de la Curva ROC:**", metricas['best_score_auc'])

# Curva ROC
st.header("Curva ROC del Mejor Modelo")
plt.figure()
plt.plot(metricas['best_fpr'], metricas['best_tpr'], label=f"Curva ROC (AUC = {metricas['best_score_auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC del Mejor Modelo")
plt.legend(loc="lower right")
st.pyplot(plt)

# Curvas de Pérdida durante el Entrenamiento y Validación
st.header("Curvas de Pérdida en Entrenamiento y Validación")
plt.figure(figsize=(10, 6))
plt.plot(metricas['best_evals_result']['train']['logloss'], label='Pérdida en Entrenamiento')
plt.plot(metricas['best_evals_result']['valid']['logloss'], label='Pérdida en Validación')
plt.xlabel("Iteración")
plt.ylabel("Log Loss")
plt.title("Gráfica de Pérdida durante el Entrenamiento")
plt.legend(loc="upper right")
plt.grid()
st.pyplot(plt)

# Curvas de Aprendizaje
st.header("Curvas de Aprendizaje del Mejor Modelo")
plt.figure(figsize=(10, 6))
train_sizes = metricas['learning_curve']['train_sizes']
train_scores_mean = metricas['learning_curve']['train_scores_mean']
train_scores_std = metricas['learning_curve']['train_scores_std']
valid_scores_mean = metricas['learning_curve']['valid_scores_mean']
valid_scores_std = metricas['learning_curve']['valid_scores_std']
plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="AUC en Entrenamiento")
plt.plot(train_sizes, valid_scores_mean, 'o-', color="red", label="AUC en Validación")
plt.fill_between(train_sizes, np.array(train_scores_mean) - np.array(train_scores_std), 
                 np.array(train_scores_mean) + np.array(train_scores_std), color="blue", alpha=0.2)
plt.fill_between(train_sizes, np.array(valid_scores_mean) - np.array(valid_scores_std), 
                 np.array(valid_scores_mean) + np.array(valid_scores_std), color="red", alpha=0.2)
plt.title("Curvas de Aprendizaje")
plt.xlabel("Tamaño del Conjunto de Entrenamiento")
plt.ylabel("AUC")
plt.legend(loc="best")
plt.grid()
st.pyplot(plt)

# Matriz de Confusión con Threshold Ajustado
st.header("Matriz de Confusión con Threshold Ajustado")
threshold = st.slider("Umbral de decisión para la Clase 1", 0.0, 1.0, 0.5)

# Hacer predicciones en el conjunto de validación ajustado
y_proba = modelo.predict(X_valid_dmatrix)
y_pred_adjusted = (y_proba >= threshold).astype(int)

# Graficar matriz de confusión con el umbral ajustado
cm_adjusted = confusion_matrix(y_valid, y_pred_adjusted)
disp_adjusted = ConfusionMatrixDisplay(confusion_matrix=cm_adjusted)
fig, ax = plt.subplots()
disp_adjusted.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Matriz de Confusión Ajustada")
plt.xlabel("Predicciones")
plt.ylabel("Valores Verdaderos")
st.pyplot(fig)

# Realizar predicciones con el modelo cargado
st.header("Realizar Predicciones con el Modelo")
st.write("Sube un archivo con los datos para predecir:")

# Entrada para archivo de datos a predecir
uploaded_file = st.file_uploader("Selecciona un archivo CSV", type="csv")
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.write(input_data.head())

    # Asegurarse de que input_data tenga las mismas columnas que el modelo espera
    input_data = input_data.reindex(columns=model_records['feature_names'], fill_value=0)
    
    # Convertir a DMatrix y hacer predicciones
    dmatrix_data = DMatrix(input_data)
    pred_proba = modelo.predict(dmatrix_data)
    pred_labels = (pred_proba >= threshold).astype(int)

    # Mostrar resultados de predicción
    st.write("Predicciones:")
    st.write(pred_labels)
    
    # Descargar los resultados
    resultados = pd.DataFrame({"Predicción": pred_labels})
    resultados["Probabilidad Clase 1"] = pred_proba
    csv = resultados.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar predicciones",
        data=csv,
        file_name="predicciones.csv",
        mime="text/csv"
    )
