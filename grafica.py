import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Leer los datos desde el archivo JSON
with open('metrics_data.json', 'r') as f:
    data = json.load(f)

# Desempaquetar los datos
train_losses = data["log_loss"]["train_losses"]
val_losses = data["log_loss"]["val_losses"]
n_estimators = data["log_loss"]["n_estimators"]

fpr1 = data["roc_curve"]["fpr1"]
tpr1 = data["roc_curve"]["tpr1"]
roc_auc1 = data["roc_curve"]["roc_auc1"]
fpr0 = data["roc_curve"]["fpr0"]
tpr0 = data["roc_curve"]["tpr0"]
roc_auc0 = data["roc_curve"]["roc_auc0"]

train_sizes = np.array(data["learning_curve"]["train_sizes"])
train_mean = np.array(data["learning_curve"]["train_mean"])
train_std = np.array(data["learning_curve"]["train_std"])
val_mean = np.array(data["learning_curve"]["val_mean"])
val_std = np.array(data["learning_curve"]["val_std"])

cm = np.array(data["confusion_matrix"])

accuracy = data["metrics"]["accuracy"]
precision = data["metrics"]["precision"]
recall = data["metrics"]["recall"]
f1 = data["metrics"]["f1"]
roc_auc = data["metrics"]["roc_auc"]

# Graficar Log Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_estimators + 1), train_losses, label='Log Loss de Entrenamiento', color='blue')
plt.plot(range(1, n_estimators + 1), val_losses, label='Log Loss de Validación', color='orange')
plt.title('Log Loss durante el entrenamiento y validación')
plt.xlabel('Número de Árboles')
plt.ylabel('Log Loss')
plt.legend()
plt.grid()
plt.show()

# Graficar la curva ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='ROC curve for class 1 (area = %0.2f)' % roc_auc1)
plt.plot(fpr0, tpr0, color='blue', lw=2, label='ROC curve for class 0 (area = %0.2f)' % roc_auc0)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Graficar Curvas de Aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Precisión en Entrenamiento', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.plot(train_sizes, val_mean, label='Precisión en Validación', color='orange')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='orange', alpha=0.2)
plt.title('Curvas de Aprendizaje')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Precisión')
plt.legend()
plt.grid()
plt.show()

# Graficar la matriz de confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
