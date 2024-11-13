import streamlit as st
import pickle
import pandas as pd
import numpy as np
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc
from xgboost import DMatrix
import plotly.graph_objects as go
import plotly.express as px
from joblib import load  # Importa load para cargar el modelo joblib
from tensorflow.keras.models import load_model  # Para cargar el modelo LSTM en formato .h5

# Configurar la página en modo amplio
st.set_page_config(layout="wide", page_title="📚 GRUPO 4 - CR 🔎")

# Definir colores de la paleta
colores = {
    "fondo": "#13141A",
    "titulo": "#A90448",
    "entrenamiento": "#FB3640",
    "validacion": "#FDA543",
    "exito": "#17C69B"
}

# Función para cargar el modelo y métricas según el modelo seleccionado
def cargar_modelo_y_métricas(nombre_modelo):
    if nombre_modelo == "rfm_xgboost":
        archivo_modelo = "dashboardData/XGBoostRFM/metricas_rfm_xgboost.pkl"
        with open(archivo_modelo, "rb") as f:
            model_records = pickle.load(f)
        return model_records, model_records['best_instance']
    elif nombre_modelo == "lstm":
        # Cargar archivos específicos para el modelo LSTM
        with open("dashboardData/LSTM/classification_reportLSTM.json", "r") as f:
            classification_report = json.load(f)
        with open("dashboardData/LSTM/training_historyLSTM.json", "r") as f:
            training_history = json.load(f)
        predictions_df = pd.read_csv("dashboardData/LSTM/predictionsLSTM.csv")
        
        # Cargar el modelo LSTM en formato .h5
        lstm_model = load_model("dashboardData/LSTM/model_lstm_optimized.h5")
        
        # Estructurar los datos como se esperan en el código
        model_records = {
            'classification_report': classification_report,
            'training_history': training_history,
            'predictions_df': predictions_df
        }
        return model_records, lstm_model  # Retorna el modelo LSTM cargado
    elif nombre_modelo == "apriori_rf":
        # Cargar datos específicos para Apriori Random Forest
        with open("dashboardData\AprioriRandomForest\metrics_data_AprioriModel.json", "r") as f:
            apriori_rf_data = json.load(f)
        
        accuracy, precision, recall, f1, roc_auc = apriori_rf_data["metrics"].values()
        report = {
            'weighted avg': {
                'precision': precision,
                'recall': recall,
                'f1-score': f1
            }
        }
        model_records = {
            'log_loss': apriori_rf_data["log_loss"],
            'roc_curve': apriori_rf_data["roc_curve"],
            'learning_curve': apriori_rf_data["learning_curve"],
            'confusion_matrix': apriori_rf_data["confusion_matrix"],
            'classification_report': report
        }
        rf_model = load("dashboardData/AprioriRandomForest/model_RandomForest_Loss.joblib")
        return model_records, rf_model
    else:
        return None, None


# Opciones de modelos en el menú desplegable
modelos_disponibles = {
    "Modelo RFM - XGBoost": "rfm_xgboost",
    "Modelo LSTM": "lstm",
    "Modelo Apriori - Random Forest": "apriori_rf"
}

# Dashboard - Título y Autores
st.markdown(f"""
<h1 style='text-align: center; color:{colores['titulo']}'>
📚 DASHBOARD - PREDICTOR DE COMPRADORES RECURRENTES - RESULTADOS 🔎
</h1>
""", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center;'>
    Autores ☝️ Diego Alexander Hernández Silvestre 21270 🛻 |
    Linda Inés Jiménez Vides 21169 🏎️ |
    Mario Antonio Guerra Morales 21008 🚲 |
    David Jonathan Aragón Vasquez 21053 🚁  
</div>
""", unsafe_allow_html=True)

# Sección de Instrucciones de uso
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<h2 style='color:{colores['titulo']}'>Instrucciones de uso</h2>", unsafe_allow_html=True)
    st.write("""
    Este dashboard ha sido diseñado para permitir una visualización y análisis detallados de los resultados obtenidos con un modelo de predicción de compradores recurrentes, desarrollado en el contexto de un proyecto universitario. Su propósito principal es ayudar a los usuarios a entender el comportamiento de los clientes y a anticipar su probabilidad de realizar compras futuras, basándose en técnicas avanzadas de análisis de datos.

    ### Cómo Utilizar el Dashboard

    1. **Resultados Generales**: En esta sección, se presentan los principales indicadores del modelo, como la precisión promedio y el AUC en validación cruzada. Estos resultados permiten obtener una visión global del rendimiento del modelo.

    2. **Métricas y Gráficas**: Visualice y analice gráficas interactivas como la curva ROC, curvas de aprendizaje y la matriz de confusión, las cuales ayudan a comprender la capacidad del modelo para clasificar correctamente a los clientes y a identificar áreas de mejora.

    3. **Predicción de Nuevos Datos**: Para realizar predicciones, suba un archivo CSV con los datos del cliente a la sección de "Predicción". El modelo procesará el archivo y generará probabilidades de recurrencia en las compras, permitiéndole identificar qué clientes tienen una mayor probabilidad de realizar compras futuras.
    """)

# Selección del modelo
with col2:
    st.write("")
    st.write("""
    ### Beneficios del Dashboard

    Este dashboard facilita la toma de decisiones basadas en datos, apoyando a los equipos de marketing y ventas para optimizar sus campañas y esfuerzos de retención de clientes. Mediante la identificación de compradores recurrentes, es posible enfocar los recursos en los clientes más valiosos, maximizando así la efectividad de las estrategias de fidelización.
    """)

    st.markdown(f"<h2 style='color:{colores['exito']}'>Selección del modelo</h2>", unsafe_allow_html=True)

    # Menú desplegable para seleccionar el modelo
    modelo_seleccionado = st.selectbox("Selecciona el modelo para visualizar", list(modelos_disponibles.keys()))

    # Cargar el modelo y métricas según el modelo seleccionado
    nombre_modelo = modelos_disponibles[modelo_seleccionado]
    model_records, modelo = cargar_modelo_y_métricas(nombre_modelo)
    metricas = model_records

    # Mostrar el modelo seleccionado
    st.write(f"**Modelo seleccionado:** {modelo_seleccionado}")

    st.subheader("Métricas de los modelos")
    # Extraer métricas del reporte de clasificación
    report = metricas['classification_report']
    results = {
        modelo_seleccionado: {
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score'],
        }
    }
    
    # Crear DataFrame para mostrar la tabla
    df_results = pd.DataFrame(results).T
    st.table(df_results)

# Curvas de Pérdida durante el Entrenamiento y Validación usando Plotly
with col3:
    st.header("Curvas de Error")
    fig_loss = go.Figure()
    if nombre_modelo == "rfm_xgboost":
        fig_loss.add_trace(go.Scatter(y=metricas['best_evals_result']['train']['logloss'], mode='lines', name='Pérdida en Entrenamiento', line=dict(color=colores['entrenamiento'])))
        fig_loss.add_trace(go.Scatter(y=metricas['best_evals_result']['valid']['logloss'], mode='lines', name='Pérdida en Validación', line=dict(color=colores['validacion'])))
        fig_loss.update_layout(title="Gráfica de Pérdida durante el Entrenamiento", xaxis_title="Iteración", yaxis_title="Log Loss", plot_bgcolor=colores['fondo'])
    elif nombre_modelo == "lstm":
        fig_loss.add_trace(go.Scatter(y=metricas['training_history']['loss'], mode='lines', name='Pérdida en Entrenamiento', line=dict(color=colores['entrenamiento'])))
        fig_loss.add_trace(go.Scatter(y=metricas['training_history']['val_loss'], mode='lines', name='Pérdida en Validación', line=dict(color=colores['validacion'])))
        fig_loss.update_layout(title="Curva de Entrenamiento - LSTM", xaxis_title="Épocas", yaxis_title="Loss", plot_bgcolor=colores['fondo'])
    elif nombre_modelo == "apriori_rf":
        # Convertir range en una lista para evitar errores
        fig_loss.add_trace(go.Scatter(x=list(range(1, metricas['log_loss']["n_estimators"] + 1)), y=metricas['log_loss']["train_losses"], mode='lines', name='Pérdida en Entrenamiento', line=dict(color=colores['entrenamiento'])))
        fig_loss.add_trace(go.Scatter(x=list(range(1, metricas['log_loss']["n_estimators"] + 1)), y=metricas['log_loss']["val_losses"], mode='lines', name='Pérdida en Validación', line=dict(color=colores['validacion'])))
        fig_loss.update_layout(title="Log Loss - Apriori Random Forest", xaxis_title="Número de Árboles", yaxis_title="Log Loss", plot_bgcolor=colores['fondo'])

    st.plotly_chart(fig_loss)

# Sección de curvas y matriz de confusión
col4, col5, col6 = st.columns(3)

# Curva ROC usando Plotly
with col4:
    st.header("Curva ROC")
    if nombre_modelo == "rfm_xgboost":
        fpr, tpr = metricas['best_fpr'], metricas['best_tpr']
        auc_score = metricas['best_score_auc']

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"Clase 1 (AUC = {auc_score:.2f})", line=dict(color=colores['entrenamiento'])))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Línea Aleatoria", line=dict(dash='dash', color=colores['fondo'])))
        fig_roc.update_layout(title="Curva ROC - Apriori Random Forest", xaxis_title="FPR", yaxis_title="TPR", plot_bgcolor=colores['fondo'])

    elif nombre_modelo == "lstm":
        y_test = metricas['predictions_df']['y_test'].values
        y_pred_proba = metricas['predictions_df']['y_pred_proba'].values
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"Clase 1 (AUC = {auc_score:.2f})", line=dict(color=colores['entrenamiento'])))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Línea Aleatoria", line=dict(dash='dash', color=colores['fondo'])))
        fig_roc.update_layout(title="Curva ROC - LSTM", xaxis_title="FPR", yaxis_title="TPR", plot_bgcolor=colores['fondo'])

    elif nombre_modelo == "apriori_rf":
        fpr1, tpr1 = metricas['roc_curve']['fpr1'], metricas['roc_curve']['tpr1']
        roc_auc1 = metricas['roc_curve']['roc_auc1']
        fpr0, tpr0 = metricas['roc_curve']['fpr0'], metricas['roc_curve']['tpr0']
        roc_auc0 = metricas['roc_curve']['roc_auc0']

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr1, y=tpr1, mode='lines', name=f"Clase 1 (AUC = {roc_auc1:.2f})", line=dict(color=colores['entrenamiento'])))
        fig_roc.add_trace(go.Scatter(x=fpr0, y=tpr0, mode='lines', name=f"Clase 0 (AUC = {roc_auc0:.2f})", line=dict(color=colores['validacion'])))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Línea Aleatoria", line=dict(dash='dash', color=colores['fondo'])))
        fig_roc.update_layout(title="Curva ROC - Apriori Random Forest", xaxis_title="FPR", yaxis_title="TPR", plot_bgcolor=colores['fondo'])

    st.plotly_chart(fig_roc)

# Curvas de Aprendizaje usando Plotly
with col5:
    st.header("Curvas de Aprendizaje")
    if nombre_modelo == "rfm_xgboost":
        train_sizes = metricas['learning_curve']['train_sizes']
        train_scores_mean = metricas['learning_curve']['train_scores_mean']
        valid_scores_mean = metricas['learning_curve']['valid_scores_mean']

        fig_learning = go.Figure()
        fig_learning.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, mode='lines+markers', name="AUC en Entrenamiento", line=dict(color=colores['entrenamiento'])))
        fig_learning.add_trace(go.Scatter(x=train_sizes, y=valid_scores_mean, mode='lines+markers', name="AUC en Validación", line=dict(color=colores['validacion'])))
        fig_learning.update_layout(title="Curvas de Aprendizaje - XGBoost", xaxis_title="Tamaño del Conjunto de Entrenamiento", yaxis_title="AUC", plot_bgcolor=colores['fondo'])
        st.plotly_chart(fig_learning)
    elif nombre_modelo == "lstm":
        fig_learning = go.Figure()
        fig_learning.add_trace(go.Scatter(y=metricas['training_history']['accuracy'], mode='lines', name='Exactitud en Entrenamiento', line=dict(color=colores['entrenamiento'])))
        fig_learning.add_trace(go.Scatter(y=metricas['training_history']['val_accuracy'], mode='lines', name='Exactitud en Validación', line=dict(color=colores['validacion'])))
        fig_learning.update_layout(title="Curvas de Aprendizaje - LSTM", xaxis_title="Épocas", yaxis_title="Exactitud", plot_bgcolor=colores['fondo'])
        st.plotly_chart(fig_learning)
    elif nombre_modelo == "apriori_rf":
        fig_learning = go.Figure()
        fig_learning.add_trace(go.Scatter(x=metricas['learning_curve']['train_sizes'], y=metricas['learning_curve']['train_mean'], mode='lines', name='Precisión en Entrenamiento', line=dict(color=colores['entrenamiento'])))
        fig_learning.add_trace(go.Scatter(x=metricas['learning_curve']['train_sizes'], y=metricas['learning_curve']['val_mean'], mode='lines', name='Precisión en Validación', line=dict(color=colores['validacion'])))
        fig_learning.update_layout(title="Curvas de Aprendizaje - Apriori RF", xaxis_title="Tamaño del Conjunto de Entrenamiento", yaxis_title="Precisión", plot_bgcolor=colores['fondo'])
        st.plotly_chart(fig_learning)

# Matriz de Confusión
with col6:
    st.header("Matriz de Confusión")

    if nombre_modelo == "rfm_xgboost":
        threshold = st.slider("Umbral de decisión", 0.0, 0.5, 0.25)
        X_valid_dmatrix = DMatrix(pd.DataFrame(model_records['X_valid'], columns=model_records['feature_names']))
        y_valid = model_records['y_valid']
        y_proba = modelo.predict(X_valid_dmatrix)
        y_pred_adjusted = (y_proba >= threshold).astype(int)
        cm_adjusted = confusion_matrix(y_valid, y_pred_adjusted)
    elif nombre_modelo == "lstm":
        threshold = st.slider("Umbral de decisión", 0.0, 0.5, 0.25)
        y_test = metricas['predictions_df']['y_test'].values
        y_pred_proba = metricas['predictions_df']['y_pred_proba'].values
        y_pred_adjusted = (y_pred_proba >= threshold).astype(int)
        y_valid = y_test
        cm_adjusted = confusion_matrix(y_valid, y_pred_adjusted)
    elif nombre_modelo == "apriori_rf":
        cm_adjusted = metricas['confusion_matrix']  # Usar la matriz de confusión proporcionada

    # Crear heatmap de la matriz de confusión
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_adjusted,
        x=["Predicción Negativa", "Predicción Positiva"],
        y=["Real Negativo", "Real Positivo"],
        colorscale=[[0, colores['fondo']], [1, colores['exito']]],
        texttemplate="%{z}",
        showscale=False
    ))
    fig_cm.update_layout(
        title="Matriz de Confusión Ajustada",
        xaxis_title="Predicciones",
        yaxis_title="Valores Verdaderos",
        plot_bgcolor=colores['fondo']
    )
    st.plotly_chart(fig_cm)

st.header("Predicción")
uploaded_file = st.file_uploader("Selecciona un archivo CSV para predecir", type="csv")
if uploaded_file is not None:
    # Cargar datos del archivo subido
    input_data = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.write(input_data.head())

    # Realizar predicciones según el modelo seleccionado
    if nombre_modelo == "rfm_xgboost":
        input_data = input_data.reindex(columns=model_records['feature_names'], fill_value=0)
        dmatrix_data = DMatrix(input_data)
        pred_proba = modelo.predict(dmatrix_data)
    elif nombre_modelo == "lstm":
        pred_proba = modelo.predict(input_data).flatten()
    elif nombre_modelo == "apriori_rf":
        pred_proba = modelo.predict_proba(input_data)[:, 1]

    # Convertir probabilidades a etiquetas (0 o 1) usando el umbral
    pred_labels = (pred_proba >= threshold).astype(int)

    # Mostrar predicciones en un DataFrame
    st.write("Predicciones detalladas:")
    resultados = pd.DataFrame({
        "ID": input_data.index,  # Usa un identificador si lo tienes, aquí se usa el índice de DataFrame
        "Predicción": pred_labels,
        "Probabilidad Clase 1": pred_proba
    })

    st.write(resultados)

    # Graficar las probabilidades de predicción con Plotly
    fig = px.bar(resultados, x="ID", y="Probabilidad Clase 1", color="Predicción",
                 color_continuous_scale=["blue", "red"], labels={"ID": "ID de muestra"})
    fig.update_layout(
        title="Probabilidades de Predicción para la Clase 1",
        xaxis_title="ID de muestra",
        yaxis_title="Probabilidad de Clase 1",
        coloraxis_colorbar=dict(title="Clase Predicha")
    )
    st.plotly_chart(fig)

    # Botón para descargar las predicciones
    csv = resultados.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar predicciones",
        data=csv,
        file_name="predicciones.csv",
        mime="text/csv"
    )
