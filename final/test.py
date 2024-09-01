import pandas as pd
import json
import joblib
# Funciones auxiliares sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_curve, auc  # Metricas
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import optuna
import traceback
from datetime import datetime
import numpy as np
import spacy
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import importlib 
import archivos
import os 
import pickle
from joblib import load



BBDD = "sqlite:///optuna_lightgbm.sqlite3"
TRIALS = 2
SEED = 12345
TEST_SIZE = 0.2


def levantar_modelo(model_path):
    try:
       return load(model_path)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Se produjo un error: {e}")
        print(f"Detalles del error:\n{tb}")


def modelo_base_grafico_metricas():
    
    STUDY_NAME = "modelo_base"
    
       
    studies_names = [
            f'lightgbm_{STUDY_NAME}', 
            f'onevsrest_gb_2_{STUDY_NAME}', 
            f'randomforest_{STUDY_NAME}'
            ]

    # studies = [optuna.load_study(study_name=study_name, storage=BBDD) for study_name in studies_names]
    
    studies = []

    for study_name_template in studies_names:
        study_name = study_name_template.format(STUDY_NAME=STUDY_NAME)
        study = optuna.load_study(study_name=study_name, storage=BBDD)
        studies.append(study)

    
    df_result = pd.DataFrame()
    df_result['study'] = [study.study_name for study in studies]
    df_result['best_params'] = [study.best_params for study in studies]
    df_result['train_score'] = [study.best_trial.value for study in studies]

    kappa_scores = get_test_score_modelo_base()
    
    # Añadir una columna 'test_score' basada en la posición del índice
    df_result['test_score'] = df_result.index.map({
        0: kappa_scores['lgb'],
        1: kappa_scores['ovr'],
        2: kappa_scores['rfc']
    })
   
    df_result.set_index('study', inplace=True   )
    
    df_result[['test_score','train_score']].sort_values('test_score',ascending=False,inplace=True,)

    # Crear el gráfico de barras
    ax = df_result.plot(kind='bar', title='Kappa score')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    
    # Agregar valores encima de las barras, rotados a 45 grados y redondeados a 3 decimales
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                height, 
                f'{height:.3f}',  # Redondear a 3 decimales
                ha='center', 
                va='bottom', 
                rotation=45,  # Rotar el texto a 45 grados
                fontsize=10,  # Opcional: ajustar el tamaño de la fuente
                color='black'  # Opcional: ajustar el color del texto
            )

    # Colocar la leyenda abajo a la derecha
    ax.legend(loc='lower right')
    # Mostrar el gráfico
    plt.tight_layout()  # Ajustar el diseño para que se vean bien las etiquetas
    plt.show()
     
        
def get_test_score_modelo_base():
    # bdd
    importlib.reload(archivos)    
    STUDY_NAME = "modelo_base"
    
    # Leemos
    df = archivos.get_modelo_base()
    
    # Preparar los datos
    final_columns = [elemento for elemento in df.columns if elemento != 'target']
    X = df[final_columns]
    y = df["target"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    
    lgb = levantar_modelo(f"models/lgbm/{STUDY_NAME}/model_{STUDY_NAME}.pkl")
    ovr = levantar_modelo(f"./models/onevsrest/{STUDY_NAME}/model_{STUDY_NAME}.pkl")
    rfc = levantar_modelo(f"./models/randomforest/{STUDY_NAME}/model_{STUDY_NAME}.pkl")
    
    # Hacer predicciones
    preds_lgb = lgb.predict(X_test)
    preds_ovr = ovr.predict(X_test)
    preds_rfc = rfc.predict(X_test)
    
    
    # Crear el DataFrame con las predicciones
    df_preds = {
        'lgb': cohen_kappa_score(y_test, preds_lgb),
        'ovr': cohen_kappa_score(y_test, preds_ovr),
        'rfc': cohen_kappa_score(y_test, preds_rfc)
    }

    
    return df_preds
    

def modelo_base_curva_roc():
    """
    BUSCAR COMO HACER CURVA ROC PARA MULTICLASE
    """
    
    STUDY_NAME = "modelo_base"
    
    #lgb = levantar_modelo(f"models/lgbm/{STUDY_NAME}/model_{STUDY_NAME}.pkl")
    ovr = levantar_modelo(f"models/onevsrest/{STUDY_NAME}/model_{STUDY_NAME}.pkl")
    rfc = levantar_modelo(f"models/randomforest/{STUDY_NAME}/model_{STUDY_NAME}.pkl")
    
    
    importlib.reload(archivos)    
    
    # Leemos
    df = archivos.get_modelo_base()
    
    # Preparar los datos
    final_columns = [elemento for elemento in df.columns if elemento != 'target']
    X = df[final_columns]
    y = df["target"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    
    # hacer las predicciones
    preds = pd.DataFrame()
    #preds["lgb"] = lgb.predict_proba(X_test)[:,1]
    preds["ovr"] = ovr.predict_proba(X_test)[:,1]
    preds["rfc"] = rfc.predict_proba(X_test)[:,1]
    
    # Calculamos rate y roc
    #fpr_lgb, tpr_lgb, _ = roc_curve(y_test, preds["lgb"])
    #roc_auc_lgb = auc(fpr_lgb, tpr_lgb)

    fpr_ovr, tpr_ovr, _ = roc_curve(y_test, preds["ovr"])
    roc_auc_ovr = auc(fpr_ovr, tpr_ovr)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, preds["rfc"])
    roc_auc_rf = auc(fpr_rf, tpr_rf)


    # Graficamos
    plt.figure(figsize=(10, 6))
    #plt.plot(fpr_lgb, tpr_lgb, color='orange', lw=2, label=f'LGB (AUC = {roc_auc_lgb:.2f})')
    plt.plot(fpr_ovr, tpr_ovr, color='orange', lw=2, label=f'OVR (AUC = {roc_auc_ovr:.2f})')
    plt.plot(fpr_rf, tpr_rf, color='purple', lw=2, label=f'RFC (AUC = {roc_auc_rf:.2f})')
    

    # Graficar la línea de referencia
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

    # Configurar detalles del gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    # Mostrar el gráfico
    plt.show()


def curva_roc_multiclase_modelo_base():
    # Parámetros y datos
    STUDY_NAME = "modelo_base"
    ovr = levantar_modelo(f"models/onevsrest/{STUDY_NAME}/model_{STUDY_NAME}.pkl")

     # Leemos
    df = archivos.get_modelo_base()
    
    # Preparar los datos
    final_columns = [elemento for elemento in df.columns if elemento != 'target']
    X = df[final_columns]
    y = df["target"]

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    # Obtener las probabilidades predichas
    y_score = ovr.predict_proba(X_test)

    # Binarizar las etiquetas verdaderas
    clases = np.unique(y)
    y_true_bin = label_binarize(y_test, classes=clases)
    n_classes = y_true_bin.shape[1]

    # Calcular la curva ROC y el área bajo la curva (AUC) para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcular la curva ROC promedio
    fpr["macro"], tpr["macro"], _ = roc_curve(
        y_true_bin.ravel(),
        y_score.ravel()
    )
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Graficar las curvas ROC
    plt.figure()
    plt.figure(figsize=(12, 6))  # Ajusta el tamaño según tus necesidades


    # Curvas ROC de cada clase
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve (class {clases[i]}) (area = {roc_auc[i]:.2f})')

    # Curva ROC promedio
    plt.plot(fpr["macro"], tpr["macro"], color='navy', lw=2, linestyle='--', 
            label=f'ROC curve (macro-average) (area = {roc_auc["macro"]:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    
    plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')  # Guarda el gráfico como un archivo PNG

    plt.show()
    
    
    
    
    
def modelo_base_matriz_de_confusion():
    
    STUDY_NAME = "modelo_base"
    
    #lgb = levantar_modelo(f"models/lgbm/{STUDY_NAME}/model_{STUDY_NAME}.pkl")
    ovr = levantar_modelo(f"models/onevsrest/{STUDY_NAME}/model_{STUDY_NAME}.pkl")
    rfc = levantar_modelo(f"models/randomforest/{STUDY_NAME}/model_{STUDY_NAME}.pkl")
    
    
    importlib.reload(archivos)    
    
    # Leemos
    df = archivos.get_modelo_base()
    
    # Preparar los datos
    final_columns = [elemento for elemento in df.columns if elemento != 'target']
    X = df[final_columns]
    y = df["target"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    
    fig, ax = plt.subplots(3, 3, figsize=(12, 6))
    
    #cm_rf_original= confusion_matrix(y_test, lgb.predict(X_test))
    cm_gb_original= confusion_matrix(y_test, ovr.predict(X_test))
    cm_rf_pesos= confusion_matrix(y_test, rfc.predict(X_test))
    
    #ConfusionMatrixDisplay(confusion_matrix = cm_rf_original, display_labels = [0, 1]).plot(ax=ax[0,0])
    ConfusionMatrixDisplay(confusion_matrix = cm_rf_pesos, display_labels = [0, 1]).plot(ax=ax[0,1])
    ConfusionMatrixDisplay(confusion_matrix = cm_gb_original, display_labels = [0, 1]).plot(ax=ax[0,2])
    


    ax[0,0].set_title('LGM')
    ax[0,1].set_title('OVR')
    ax[0,2].set_title('RFC')
    

    fig.delaxes(ax[1, 0])
    fig.delaxes(ax[1, 1])
    fig.delaxes(ax[1, 2])
    fig.delaxes(ax[2, 0])
    fig.delaxes(ax[2, 1])
    fig.delaxes(ax[2, 2])

    plt.suptitle("Matrices de Confusión")
    plt.tight_layout()

    plt.show()