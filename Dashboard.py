import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score

# Configurar la página en modo amplio
st.set_page_config(layout="wide")

# Cargar el modelo, métricas y curvas
def cargar_modelo_y_métricas():
    # Cargar el modelo guardado
    with open("modelo_mejorado_pca_v2.pkl", "rb") as f:
        modelo = pickle.load(f)

    # Cargar las métricas y resultados guardados (simulados para este ejemplo)
    metricas = {
        "best_auc": 0.90,  # Valor de ejemplo
        "accuracy": 0.88,
        "roc_auc_score": 0.90,
        "classification_report": {
            "precision": 0.87,
            "recall": 0.85,
            "f1_score": 0.86
        }
    }
    
    # Simulación de datos de curvas
    # Cargar las curvas de pérdida del entrenamiento y validación
    loss_train = [0.6, 0.5, 0.4, 0.3]  # Valores de ejemplo para pérdida en entrenamiento
    loss_valid = [0.65, 0.55, 0.45, 0.35]  # Valores de ejemplo para pérdida en validación

    # Cargar los tamaños y puntajes de las curvas de aprendizaje
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = np.linspace(0.7, 0.9, 10) + np.random.rand(10) * 0.05  # Ejemplo de puntajes en entrenamiento
    valid_scores = np.linspace(0.65, 0.85, 10) + np.random.rand(10) * 0.05  # Ejemplo de puntajes en validación

    # Cargar la curva ROC (simulada)
    fpr = np.linspace(0, 1, 100)
    tpr = fpr ** 0.5  # Curva ROC simulada para el ejemplo
    return modelo, metricas, loss_train, loss_valid, train_sizes, train_scores, valid_scores, fpr, tpr

# Llamar a la función para cargar el modelo y las métricas
modelo, metricas, loss_train, loss_valid, train_sizes, train_scores, valid_scores, fpr, tpr = cargar_modelo_y_métricas()

# Mostrar métricas y gráficos en el dashboard
st.title("Dashboard de Modelo Clasificador")
st.write("Este dashboard permite ver las métricas de rendimiento del modelo preentrenado y realizar predicciones.")

# Métricas del modelo
st.header("Métricas del Modelo")
st.write("**Mejor AUC**:", metricas["best_auc"])
st.write("**Exactitud (Accuracy)**:", metricas["accuracy"])
st.write("**ROC-AUC Score**:", metricas["roc_auc_score"])
st.write("**Reporte de Clasificación:**")
st.write(metricas["classification_report"])

# Curva ROC
st.header("Curva ROC del Mejor Modelo")
plt.figure()
plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {metricas['roc_auc_score']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC del Mejor Modelo")
plt.legend(loc="lower right")
st.pyplot(plt)

# Curvas de Pérdida durante el Entrenamiento y Validación
st.header("Curvas de Pérdida en Entrenamiento y Validación")
plt.figure(figsize=(10, 6))
plt.plot(loss_train, label='Pérdida en Entrenamiento')
plt.plot(loss_valid, label='Pérdida en Validación')
plt.xlabel("Iteración")
plt.ylabel("Log Loss")
plt.title("Gráfica de Pérdida durante el Entrenamiento")
plt.legend(loc="upper right")
plt.grid()
st.pyplot(plt)

# Curvas de Aprendizaje
st.header("Curvas de Aprendizaje del Mejor Modelo")
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, 'o-', color="blue", label="AUC en Entrenamiento")
plt.plot(train_sizes, valid_scores, 'o-', color="red", label="AUC en Validación")
plt.fill_between(train_sizes, train_scores - np.std(train_scores), train_scores + np.std(train_scores), color="blue", alpha=0.2)
plt.fill_between(train_sizes, valid_scores - np.std(valid_scores), valid_scores + np.std(valid_scores), color="red", alpha=0.2)
plt.title("Curvas de Aprendizaje")
plt.xlabel("Tamaño del Conjunto de Entrenamiento")
plt.ylabel("AUC")
plt.legend(loc="best")
plt.grid()
st.pyplot(plt)

# Matriz de Confusión con Threshold Ajustado
st.header("Matriz de Confusión con Threshold Ajustado")
threshold = st.slider("Umbral de decisión para la Clase 1", 0.0, 1.0, 0.5)

# Simulación de predicciones para mostrar la matriz de confusión
y_valid = np.random.randint(0, 2, size=100)  # Datos simulados
y_proba = np.random.rand(100)  # Probabilidades simuladas
y_pred_adjusted = (y_proba >= threshold).astype(int)

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

    # Convertir a DMatrix y hacer predicciones
    dmatrix_data = DMatrix(input_data)  # Reemplazar con `input_data_reduced` si usas PCA
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
