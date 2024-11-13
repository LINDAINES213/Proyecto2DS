import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from xgboost import DMatrix
import plotly.graph_objects as go
import plotly.express as px

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
    archivo_modelo = f"metricas_{nombre_modelo}.pkl"
    with open(archivo_modelo, "rb") as f:
        model_records = pickle.load(f)
    return model_records

# Opciones de modelos en el menú desplegable
modelos_disponibles = {
    "Modelo RFM - XGBoost": "rfm_xgboost",
    "Modelo SVM": "svm",
    "Modelo LSTM": "lstm",
    "Modelo CNN": "cnn"
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
    model_records = cargar_modelo_y_métricas(nombre_modelo)
    modelo = model_records['best_instance']
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
    fig_loss.add_trace(go.Scatter(y=metricas['best_evals_result']['train']['logloss'], mode='lines', name='Pérdida en Entrenamiento', line=dict(color=colores['entrenamiento'])))
    fig_loss.add_trace(go.Scatter(y=metricas['best_evals_result']['valid']['logloss'], mode='lines', name='Pérdida en Validación', line=dict(color=colores['validacion'])))
    fig_loss.update_layout(title="Gráfica de Pérdida durante el Entrenamiento", xaxis_title="Iteración", yaxis_title="Log Loss", plot_bgcolor=colores['fondo'])
    st.plotly_chart(fig_loss)

# Sección de curvas y matriz de confusión
col4, col5, col6 = st.columns(3)

# Curva ROC usando Plotly
with col4:
    st.header("Curva ROC")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=metricas['best_fpr'], y=metricas['best_tpr'], mode='lines', name="Curva ROC", line=dict(color=colores['entrenamiento'])))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Línea Aleatoria", line=dict(dash='dash', color=colores['fondo'])))
    fig_roc.update_layout(title=f"Curva ROC (AUC = {metricas['best_score_auc']:.2f})", xaxis_title="Tasa de Falsos Positivos", yaxis_title="Tasa de Verdaderos Positivos", plot_bgcolor=colores['fondo'])
    st.plotly_chart(fig_roc)

# Curvas de Aprendizaje usando Plotly
with col5:
    st.header("Curvas de Aprendizaje")
    train_sizes = metricas['learning_curve']['train_sizes']
    train_scores_mean = metricas['learning_curve']['train_scores_mean']
    valid_scores_mean = metricas['learning_curve']['valid_scores_mean']

    fig_learning = go.Figure()
    fig_learning.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, mode='lines+markers', name="AUC en Entrenamiento", line=dict(color=colores['entrenamiento'])))
    fig_learning.add_trace(go.Scatter(x=train_sizes, y=valid_scores_mean, mode='lines+markers', name="AUC en Validación", line=dict(color=colores['validacion'])))
    fig_learning.update_layout(title="Curvas de Aprendizaje", xaxis_title="Tamaño del Conjunto de Entrenamiento", yaxis_title="AUC", plot_bgcolor=colores['fondo'])
    st.plotly_chart(fig_learning)

# Matriz de Confusión con Plotly Heatmap
with col6:
    st.header("Matriz de Confusión")
    threshold = st.slider("Umbral de decisión", 0.0, 0.5, 0.25)
    X_valid_dmatrix = DMatrix(pd.DataFrame(model_records['X_valid'], columns=model_records['feature_names']))
    y_valid = model_records['y_valid']
    y_proba = modelo.predict(X_valid_dmatrix)
    y_pred_adjusted = (y_proba >= threshold).astype(int)
    cm_adjusted = confusion_matrix(y_valid, y_pred_adjusted)

    # Crear heatmap personalizado para la matriz de confusión
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

# Sección de predicciones de usuario
st.header("Predicción")
uploaded_file = st.file_uploader("Selecciona un archivo CSV para predecir", type="csv")
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.write(input_data.head())

    input_data = input_data.reindex(columns=model_records['feature_names'], fill_value=0)
    dmatrix_data = DMatrix(input_data)
    pred_proba = modelo.predict(dmatrix_data)
    pred_labels = (pred_proba >= threshold).astype(int)

    st.write("Predicciones:")
    st.write(pred_labels)

    resultados = pd.DataFrame({"Predicción": pred_labels, "Probabilidad Clase 1": pred_proba})
    csv = resultados.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar predicciones",
        data=csv,
        file_name="predicciones.csv",
        mime="text/csv"
    )
