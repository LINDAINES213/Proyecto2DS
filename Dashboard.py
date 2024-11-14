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
import joblib

st.set_page_config(layout="wide", page_title="📚 GRUPO 4 - CR 🔎")

colores = {
    "fondo": "#13141A",
    "titulo": "#A90448",
    "entrenamiento": "#FB3640",
    "validacion": "#FDA543",
    "exito": "#17C69B"
}

def cargar_modelo_y_métricas(nombre_modelo):
    if nombre_modelo == "rfm_xgboost":
        archivo_modelo = "dashboardData/XGBoostRFM/metricas_rfm_xgboost.pkl"
        with open(archivo_modelo, "rb") as f:
            model_records = pickle.load(f)
        return model_records, model_records['best_instance']
    elif nombre_modelo == "lstm":
        with open("dashboardData/LSTM/classification_reportLSTM.json", "r") as f:
            classification_report = json.load(f)
        with open("dashboardData/LSTM/training_historyLSTM.json", "r") as f:
            training_history = json.load(f)
        predictions_df = pd.read_csv("dashboardData/LSTM/predictionsLSTM.csv")
        
        lstm_model = load_model("dashboardData/LSTM/model_lstm_optimized.h5")
        
        model_records = {
            'classification_report': classification_report,
            'training_history': training_history,
            'predictions_df': predictions_df
        }
        return model_records, lstm_model 
    elif nombre_modelo == "apriori_rf":
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

    elif nombre_modelo == "logistic_regression":
        archivo_modelo = "dashboardData/RegresionLogistica/modelo_logistico_con_metricas.pkl"
        model_records = joblib.load(archivo_modelo)
        return model_records, model_records['model']
    else:
        return None, None


modelos_disponibles = {
    "Modelo RFM - XGBoost": "rfm_xgboost",
    "Modelo LSTM": "lstm",
    "Modelo Apriori - Random Forest": "apriori_rf",
    "Modelo Regresión Logística": "logistic_regression"
}

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

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<h2 style='color:{colores['titulo']}'>Instrucciones de uso</h2>", unsafe_allow_html=True)
    st.write("""
    Este dashboard permite visualizar y analizar los resultados de un modelo seleccionado de predicción de compradores recurrentes. Puedes explorar las métricas y gráficas asociadas a cada modelo y realizar predicciones cargando nuevos datos de clientes en formato CSV.

    ### Cómo Utilizar el Dashboard

    1. **Selección del Modelo**:
       En el menú "Selección del modelo", puedes elegir entre tres modelos diferentes:
        - **RFM - XGBoost**: Un modelo basado en el análisis RFM (Recencia, Frecuencia, Valor Monetario) usando XGBoost.
        - **LSTM**: Una red neuronal recurrente que utiliza secuencias de comportamiento de clientes.
        - **Apriori - Random Forest**: Un modelo que combina reglas de asociación con Random Forest.
        - **Regresión Logística**: Un modelo de clasificación lineal simple.
       Cada modelo muestra sus métricas y resultados específicos.
             
    2. **Visualización de Métricas y Gráficas**:
       Una vez que seleccionas un modelo, puedes visualizar sus principales métricas de rendimiento, como la precisión y el F1-Score. También podrás ver:
        - **Curvas de Pérdida**: Para evaluar la evolución del error durante el entrenamiento y validación.
        - **Curva ROC**: Muestra la capacidad de discriminación del modelo.
        - **Curvas de Aprendizaje**: Ayuda a entender cómo mejora el modelo con más datos de entrenamiento.
        - **Matriz de Confusión**: Disponible para ajustar y ver la precisión en función de un umbral de decisión.

    """)

# Selección del modelo
with col2:
    st.write("")
    st.write("""
    ### Recomendaciones:
    - **Formato de datos**: Asegúrate de que el archivo CSV cumpla con el formato específico del modelo seleccionado. Los detalles sobre las columnas requeridas se encuentran en la documentación del proyecto.
    - **Umbral de decisión**: Experimenta con diferentes umbrales para ver cómo afectan la matriz de confusión y las predicciones.

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
    if nombre_modelo == "logistic_regression":
        report = model_records['metrics']['classification_report']
    else:
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
    elif nombre_modelo == "logistic_regression":
        fig_loss.add_trace(go.Scatter(
            x=list(range(1, len(metricas['loss_values']) + 1)),  # Épocas en el eje X
            y=metricas['loss_values'],  # Valores de pérdida en el eje Y
            mode='lines',  # Solo líneas para una curva suavizada
            name='Pérdida (Log Loss)',
            line=dict(color=colores['entrenamiento'])
        ))
        fig_loss.update_layout(
            title="Curva de Pérdida - Regresión Logística",
            xaxis_title="Épocas",
            yaxis_title="Log Loss",
            plot_bgcolor=colores['fondo']
        )

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
    
    elif nombre_modelo == "logistic_regression":
        fig_roc = go.Figure()

        color_map = {
            'class_-1': colores['entrenamiento'],  # Clase -1
            'class_0': colores['validacion'],      # Clase 0
            'class_1': "#17C69B"                   # Clase 1 (por ejemplo, verde éxito)
        }
        for class_label, data in metricas['roc_auc_data'].items():
            fpr, tpr, auc_score = data['fpr'], data['tpr'], data['auc']
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"{class_label} (AUC = {auc_score:.2f})",
                line=dict(color=color_map.get(class_label, colores['entrenamiento']))  # Color según clase
            ))

        # Añadir la línea de referencia (clasificación aleatoria)
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name="Línea Aleatoria",
            line=dict(dash='dash', color=colores['exito'])
        ))

        # Configuración de la gráfica
        fig_roc.update_layout(
            title="Curva ROC - Regresión Logística",
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            plot_bgcolor=colores['fondo']
        )

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
    elif nombre_modelo == "logistic_regression":
        # Convertir las listas a arrays de NumPy para realizar operaciones elementales
        train_sizes = np.array(metricas['learning_curve']['train_sizes'])
        train_scores_mean = np.array(metricas['learning_curve']['train_scores_mean'])
        train_scores_std = np.array(metricas['learning_curve']['train_scores_std'])
        valid_scores_mean = np.array(metricas['learning_curve']['valid_scores_mean'])
        valid_scores_std = np.array(metricas['learning_curve']['valid_scores_std'])

        fig_learning = go.Figure()

        # AUC en Entrenamiento con banda de desviación estándar
        fig_learning.add_trace(go.Scatter(
            x=train_sizes, 
            y=train_scores_mean, 
            mode='lines', 
            name="AUC en Entrenamiento", 
            line=dict(color=colores['entrenamiento'])
        ))
        fig_learning.add_trace(go.Scatter(
            x=train_sizes, 
            y=train_scores_mean + train_scores_std, 
            mode='lines', 
            name="Desviación Estándar Entrenamiento (+)", 
            line=dict(color=colores['entrenamiento'], width=0),
            showlegend=False,
            fill='tonexty'  # Relleno hacia abajo hasta la curva de media
        ))
        fig_learning.add_trace(go.Scatter(
            x=train_sizes, 
            y=train_scores_mean - train_scores_std, 
            mode='lines', 
            name="Desviación Estándar Entrenamiento (-)", 
            line=dict(color=colores['entrenamiento'], width=0),
            showlegend=False,
            fill='tonexty'  # Relleno hacia abajo hasta la curva de media
        ))

        # AUC en Validación con banda de desviación estándar
        fig_learning.add_trace(go.Scatter(
            x=train_sizes, 
            y=valid_scores_mean, 
            mode='lines', 
            name="AUC en Validación", 
            line=dict(color=colores['validacion'])
        ))
        fig_learning.add_trace(go.Scatter(
            x=train_sizes, 
            y=valid_scores_mean + valid_scores_std, 
            mode='lines', 
            name="Desviación Estándar Validación (+)", 
            line=dict(color=colores['validacion'], width=0),
            showlegend=False,
            fill='tonexty'  # Relleno hacia abajo hasta la curva de media
        ))
        fig_learning.add_trace(go.Scatter(
            x=train_sizes, 
            y=valid_scores_mean - valid_scores_std, 
            mode='lines', 
            name="Desviación Estándar Validación (-)", 
            line=dict(color=colores['validacion'], width=0),
            showlegend=False,
            fill='tonexty'  # Relleno hacia abajo hasta la curva de media
        ))

        # Configuración de la gráfica
        fig_learning.update_layout(
            title="Curvas de Aprendizaje - Regresión Logística",
            xaxis_title="Tamaño del Conjunto de Entrenamiento",
            yaxis_title="AUC",
            plot_bgcolor=colores['fondo']
        )

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
        x_labels=["Predicción Negativa", "Predicción Positiva"]
        y_labels= ["Real Negativo", "Real Positivo"]
    elif nombre_modelo == "lstm":
        threshold = st.slider("Umbral de decisión", 0.0, 0.5, 0.25)
        y_test = metricas['predictions_df']['y_test'].values
        y_pred_proba = metricas['predictions_df']['y_pred_proba'].values
        y_pred_adjusted = (y_pred_proba >= threshold).astype(int)
        y_valid = y_test
        cm_adjusted = confusion_matrix(y_valid, y_pred_adjusted)
        x_labels=["Predicción Negativa", "Predicción Positiva"]
        y_labels= ["Real Negativo", "Real Positivo"]
    elif nombre_modelo == "apriori_rf":
        cm_adjusted = metricas['confusion_matrix']  # Usar la matriz de confusión proporcionada
        x_labels=["Predicción Negativa", "Predicción Positiva"]
        y_labels= ["Real Negativo", "Real Positivo"]
    elif nombre_modelo == "logistic_regression":
        cm_adjusted = metricas['metrics']['confusion_matrix']  # Matriz de confusión precalculada para regresión logística
        x_labels = ["Predicción -1", "Predicción 0", "Predicción 1"]
        y_labels = ["Real -1", "Real 0", "Real 1"]
    # Crear heatmap de la matriz de confusión
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_adjusted,
        x=x_labels,
        y=y_labels,
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

# Configurar la interfaz de Streamlit
st.header("Predicción")
uploaded_file = st.file_uploader("Selecciona un archivo CSV para predecir", type="csv")
if uploaded_file is not None:
    # Cargar datos desde el archivo subido
    input_data = pd.read_csv(uploaded_file)
    # Título y vista previa centrados
    st.markdown("<h3 style='text-align: center;'>Vista previa de los datos</h3>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center;'>
            {input_data.head().to_html(index=False)}
        </div>
        """,
        unsafe_allow_html=True
    )
    # Preparar los datos según el modelo seleccionado
    if nombre_modelo == "rfm_xgboost":
        input_data = input_data.reindex(columns=model_records['feature_names'], fill_value=0)
        dmatrix_data = DMatrix(input_data)
        pred_proba = modelo.predict(dmatrix_data)
    
    elif nombre_modelo == "lstm":
        # Cargar el scaler para el modelo LSTM
        scaler = joblib.load('dashboardData/LSTM/scalerLSTM.pkl')

        # Parámetros de la secuencia esperados por el modelo LSTM
        sequence_length = 10
        required_columns = [
            'age_range', 'gender', 'clicks', 'add_to_cart', 'purchases', 
            'add_to_favorites', 'total_actions', 'unique_item_count', 
            'user_id', 'merchant_id', 'item_id', 'category_id', 'brand_id'
        ]
        # Seleccionar las columnas requeridas para el modelo LSTM
        input_data = input_data[required_columns]

        # Convertir a array y escalar los datos
        data_array = input_data.to_numpy()
        data_scaled = scaler.transform(data_array)

        # Generar secuencias de longitud adecuada (10 pasos)
        sequences = []
        for i in range(len(data_scaled) - sequence_length + 1):
            sequences.append(data_scaled[i:i + sequence_length])
        
        sequences = np.array(sequences)

        # Verificar si hay suficientes secuencias para realizar predicciones
        if len(sequences) > 0:
            pred_proba = modelo.predict(sequences).flatten()
        else:
            st.error(f"No hay suficientes datos para generar secuencias de longitud {sequence_length}.")
            pred_proba = []

    elif nombre_modelo == "apriori_rf":
        feature_names = modelo.feature_names_in_
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        pred_proba = modelo.predict_proba(input_data)[:, 1]

    elif nombre_modelo == "logistic_regression":
        # Definir manualmente los nombres de las columnas según el entrenamiento del modelo
        feature_names = [
            'user_id', 'age_range', 'gender', 'merchant_id', 'item_id', 'category_id', 
            'brand_id', 'clicks', 'add_to_cart', 'purchases', 'add_to_favorites', 
            'total_actions', 'unique_item_count'
        ]
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Obtener las probabilidades de la clase positiva (1)
        pred_proba = modelo.predict_proba(input_data)
        

    # Convertir probabilidades a etiquetas (0 o 1) usando el umbral si hay predicciones
    if len(pred_proba) > 0:
        if nombre_modelo == "apriori_rf" or nombre_modelo == "lstm":
            pred_labels = (pred_proba).astype(int)
        elif nombre_modelo == "logistic_regression":
            pred_labels = np.argmax(pred_proba, axis=1) - 1
        else:
            pred_labels = (pred_proba >= threshold).astype(int)

        # Crear DataFrame de resultados
        if nombre_modelo == "logistic_regression":
            resultados = pd.DataFrame({
                "ID": range(len(pred_labels)),
                "Predicción": pred_labels,
                "Probabilidad Clase -1": pred_proba[:, 0],
                "Probabilidad Clase 0": pred_proba[:, 1],
                "Probabilidad Clase 1": pred_proba[:, 2]
            })
            y_col = "Probabilidad Clase 1"
        else:
            resultados = pd.DataFrame({
                "ID": range(len(pred_labels)),
                "Predicción": pred_labels,
                "Probabilidad recurrencia": pred_proba
            })
            y_col = "Probabilidad recurrencia"
        # Crear dos columnas en Streamlit
        col1, col2 = st.columns([1, 4])

        # Mostrar predicciones detalladas en la primera columna
        with col1:
            st.write("Predicciones detalladas:")
            st.write(resultados)

            # Botón para descargar las predicciones
            csv = resultados.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar predicciones",
                data=csv,
                file_name="predicciones.csv",
                mime="text/csv"
            )

        # Graficar las probabilidades de predicción en la segunda columna con un efecto degradado en el color de las barras
        with col2:
            fig = px.bar(
                resultados,
                x="ID",
                y=y_col,
                color=y_col,  # Usamos la probabilidad para el degradado
                color_continuous_scale=["#17C69B", "#FB3640"],  # Verde para probabilidad baja, rojo para alta
                labels={"ID": "ID de muestra"}
            )

            # Configurar el diseño del gráfico
            fig.update_layout(
                title="Probabilidades de recurrencia de clientes",
                title_font=dict(size=20, color="#A90448"),  # Usar el color de título definido
                xaxis_title="ID de muestra",
                yaxis_title=y_col,
                plot_bgcolor="#13141A",  # Fondo oscuro
                paper_bgcolor="#13141A",  # Fondo de papel oscuro
                font=dict(color="white"),
                coloraxis_colorbar=dict(title="Probabilidad")
            )

            # Ajustar el color de los ejes y el título para que tengan contraste
            fig.update_yaxes(title_font=dict(color="white"), tickfont=dict(color="white"))
            fig.update_xaxes(title_font=dict(color="white"), tickfont=dict(color="white"))

            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig)
