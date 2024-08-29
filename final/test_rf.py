import pandas as pd
import json
import joblib
# Funciones auxiliares sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score  # Metricas
from sklearn.metrics import confusion_matrix
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

BBDD = "sqlite:///optuna.sqlite3"
TRIALS = 2
SEED = 12345
TEST_SIZE = 0.2




def ploimportance(importance_df,path):
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importancia de la característica')
    plt.ylabel('Características')
    plt.title('Importancia de las Características en el Modelo')
    plt.gca().invert_yaxis()
    plt.savefig(path)
    
    
def plotcm(cm,path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(path)

 


def modelo_base():
    
    # bdd
    importlib.reload(archivos)    
    STUDY_NAME="randomforest1234_modelo_base"
    
    # Leemos
    df = archivos.get_modelo_base()
    
    # Preparar los datos
    final_columns = [elemento for elemento in df.columns if elemento != 'target']
    X = df[final_columns]
    y = df["target"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    
     
    # Levantamos el studio 
    print(f"[{datetime.now()}] - Cargando el modelo optimo \n")
    study = optuna.load_study(study_name=STUDY_NAME, storage=BBDD)
    print(f"[{datetime.now()}] - Mejores hiperparametros: {study.best_params} \n")
    
    
    # model
    best_params = study.best_params
    best_model = RandomForestClassifier(**best_params, random_state=SEED)
    
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, f'models/randomforest/modelo_base/{STUDY_NAME}.pkl')
        
    y_pred = best_model.predict(X_test)
     
    # Obtener la importancia de las características
    importance = best_model.feature_importances_

    # Crear un DataFrame para visualizar mejor
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance
    }).sort_values(by='importance', ascending=False)
       
    # metricas
    test_kappa = cohen_kappa_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    with open(f'models/randomforest/modelo_base/metrics_{STUDY_NAME}.txt', 'w') as f:
            f.write(f'study_name: {STUDY_NAME}\n')
            f.write(f'Fecha y hora: {datetime.now()}\n')
            f.write(f'Kappa: {test_kappa}\n')
            f.write(f'Accuracy: {test_accuracy}\n')
            f.write(f'Dimensióm Test: {X_test.shape}\n')
            f.write(f'Mejores hiperparámetros: {study.best_params}\n')
    
    print(f"Kappa en test: {test_kappa}")
    print(f"Acurracy en test: {test_kappa}")
    
    cm = confusion_matrix(y_test, y_pred)
    plotcm(cm,f'models/randomforest/modelo_base/confusion_matrix_{STUDY_NAME}.png')
    ploimportance(importance_df,f'models/randomforest/modelo_base/importance_{STUDY_NAME}.png')





def get_numeric_columns(df):
  """
  Devuelve una lista de columnas numéricas en el DataFrame df.

  Parameters:
  df (pd.DataFrame): DataFrame del que obtener las columnas numéricas.

  Returns:
  list: Lista de nombres de columnas numéricas.
  """
  # Obtener las columnas numéricas
  numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
  numeric_cols.remove("target")
  return numeric_cols



def get_categorical_columns(df, text_column):
    """
    Devuelve una lista de columnas categóricas en el DataFrame df,
    excluyendo la columna especificada (text_column).

    Parameters:
    df (pd.DataFrame): DataFrame del que obtener las columnas categóricas.
    text_column (str): Nombre de la columna de texto a excluir.

    Returns:
    list: Lista de nombres de columnas categóricas excluyendo text_column.
    """
    # Obtener las columnas categóricas
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Excluir la columna de texto
    for a in text_column:
        if a in categorical_cols:
            categorical_cols.remove(a)
    
    return categorical_cols