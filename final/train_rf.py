import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime
import traceback
import warnings
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import optuna
from sklearn.metrics import make_scorer, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold 
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
import spacy
from sklearn.ensemble import RandomForestClassifier

from nltk.corpus import stopwords

warnings.filterwarnings("ignore")

def train_randomForest(path_df, study_name, ntrials, flag_generar_archivo=True):
  """
  Entrenamamos un modelo de red neuronal secuencial con Keras.
  """
  try:
    
    # leer datos
    # os.makedirs("models", exist_ok=True)
    # os.makedirs("models/rn", exist_ok=True)
    # os.makedirs(f"models/rn/{study_name}", exist_ok=True)    
    bbdd = "sqlite:///optuna.sqlite3"
    
       
    
    # abrimos directamente el archiv
    df = pd.read_excel(path_df)
    print(f"{datetime.now()} - Inicio preprocessing \n")
    df = preprocessing(path_df)
    print(f"{datetime.now()} - Inicio feature engineering \n")
    df = featureEngineering(df)
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df, ['Descripcion'])
    text_colummns = ["Descripcion"]
    pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
    numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
    final_columns = numeric_columns + categorical_columns + text_colummns + pesos_columns
    df["Descripcion"] = df["Descripcion"].fillna('')
    
    # Preparar los datos
    X = df[final_columns]
    y = df["target"].values

    SEED = 12345
    TEST_SIZE = 0.2

    
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=TEST_SIZE, random_state=SEED)


    # Definir los transformadores para el pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_columns),  # No se aplica ningún preprocesamiento a las variables numéricas
            ('pesos', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), pesos_columns),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_columns),
            ('text', TfidfVectorizer(max_features=10000), "Descripcion")
        ],
        remainder='drop'
    )
    
    
    # Función objetivo para Optuna
    def objective(trial):
        # Hiperparámetros a optimizar
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)

        # Crear el modelo con los hiperparámetros sugeridos
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        # Crear pipeline completo
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Entrenar el modelo
        pipeline.fit(X_train, y_train)

        # Evaluar el modelo en el conjunto de test
        preds = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        return accuracy
    
    
    # Crear un estudio y optimizar
    print(f"{datetime.now()} - Inicio optimizacion de hiperparametros \n")
    study = optuna.create_study(direction='maximize', 
                                storage=bbdd,  # Specify the storage URL here.
                                study_name=f"randomforest_{study_name}",
                                load_if_exists=True)
    study.optimize(objective, n_trials=ntrials)

 
    # Imprimir los mejores hiperparámetros encontrados
    print('Mejores hiperparámetros:', study.best_params)
    print('Mejor valor de Kappa:', study.best_value)
        
    
    
  except Exception as e:
    tb = traceback.format_exc()
    print(f"Se produjo un error: {e}")
    print(f"Detalles del error:\n{tb}")





#Genero una metrica para que lightGBM haga la evaluación y pueda hacer early_stopping en el cross validation
def rfc_custom_metric_kappa(dy_pred, dy_true):
    metric_name = 'kappa'
    value = cohen_kappa_score(dy_true.get_label(),dy_pred.argmax(axis=1),weights = 'quadratic')
    is_higher_better = True
    return(metric_name, value, is_higher_better)

#Funcion objetivo a optimizar. En este caso vamos a hacer 5fold cv sobre el conjunto de train. 
# El score de CV es el objetivo a optimizar. Ademas vamos a usar los 5 modelos del CV para estimar el conjunto de test,
# registraremos en optuna las predicciones, matriz de confusion y el score en test.
# CV Score -> Se usa para determinar el rendimiento de los hiperparametros con precision 
# Test Score -> Nos permite testear que esta todo OK, no use (ni debo usar) esos datos para nada en el entrenamiento 
# o la optimizacion de hiperparametros



def holamundo(df_path, study_name, trials):

  bbdd = "sqlite:///optuna.sqlite3"
    
       
    
  # abrimos directamente el archiv
  df = pd.read_excel(df_path)
  print(f"{datetime.now()} - Inicio preprocessing \n")
  df = preprocessing(df_path)
  print(f"{datetime.now()} - Inicio feature engineering \n")
  df = featureEngineering(df)
  numeric_columns = get_numeric_columns(df)
  categorical_columns = get_categorical_columns(df, ['Descripcion'])
  text_colummns = ["Descripcion"]
  pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
  numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
  final_columns = numeric_columns + categorical_columns + text_colummns + pesos_columns
  df["Descripcion"] = df["Descripcion"].fillna('')
  
  # Preparar los datos
  X = df[final_columns]
  y = df["target"]

  SEED = 12345
  TEST_SIZE = 0.2

  
  X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=TEST_SIZE, random_state=SEED)


  # Definir los transformadores para el pipeline
  preprocessor = ColumnTransformer(
      transformers=[
          ('num', 'passthrough', numeric_columns),  # No se aplica ningún preprocesamiento a las variables numéricas
          ('pesos', Pipeline([
              ('imputer', SimpleImputer(strategy='mean')),
              ('scaler', StandardScaler())
          ]), pesos_columns),
          ('cat', Pipeline([
              ('imputer', SimpleImputer(strategy='most_frequent')),
              ('onehot', OneHotEncoder(handle_unknown='ignore'))
          ]), categorical_columns),
          ('text', TfidfVectorizer(max_features=10000), "Descripcion")
      ],
      remainder='drop'
  )

  def cv_es_rfc_objective(trial):

    #Parametros para LightGBM
    rfc_params = {      
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        #'max_depth': trial.suggest_int('max_depth', 3, 4)
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                        } 

    #Voy a generar estimaciones de los 5 modelos del CV sobre los datos test y los acumulo en la matriz scores_ensemble
    scores_ensemble = np.zeros((len(y_test),len(y_train.unique())))

    #Score del 5 fold CV inicializado en 0
    score_folds = 0

    #Numero de splits del CV
    n_splits = 5

    #Objeto para hacer el split estratificado de CV
    skf = StratifiedKFold(n_splits=n_splits)

    for i, (if_index, oof_index) in enumerate(skf.split(X_train, y_train)):
        
        # Dataset in fold (donde entreno)
        X_if, y_if = X_train.iloc[if_index], y_train.iloc[if_index]
        
        # Dataset Out of fold (donde mido la performance del CV)
        X_oof, y_oof = X_train.iloc[oof_index], y_train.iloc[oof_index]

        # Crear el modelo RandomForestClassifier con los parámetros sugeridos
        rfc_model = RandomForestClassifier(**rfc_params, random_state=42)
        
        # Crear pipeline completo
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', rfc_model)
        ])
        
        # Entrenar el modelo
        pipeline.fit(X_if, y_if)
        
        # Acumular los scores (probabilidades) de cada clase para cada uno de los modelos que determino en los folds
        scores_ensemble += pipeline.predict_proba(X_test)
        
        # Score del fold (registros de dataset train que en este fold quedan out of fold)
        score_folds += cohen_kappa_score(y_oof, pipeline.predict(X_oof), weights='quadratic') / n_splits


    #Guardo prediccion del trial sobre el conjunto de test
    # Genero nombre de archivo
    
    

    #Determino score en conjunto de test y asocio como metrica adicional en optuna
    test_score = cohen_kappa_score(y_test,scores_ensemble.argmax(axis=1),weights = 'quadratic')
    trial.set_user_attr("test_score", test_score)

    #Devuelvo score del 5fold cv a optuna para que optimice en base a eso
    return(score_folds)

  

  #Genero estudio
  study = optuna.create_study(direction='maximize', 
                                storage=bbdd,  # Specify the storage URL here.
                                study_name=f"randomforest1234_{study_name}",
                                load_if_exists=True)
    
  #Corro la optimizacion
  study.optimize(cv_es_rfc_objective, n_trials=10)


  
def preprocessing(path_df): 
  """
  Eliminamos columnas que no sirven, Imputamos NANs, convertimos Class a numérico y limpiamos el texto.
  """
  df = pd.read_excel(path_df)
  
  
  df.drop(columns=["DescCuenta","NTesoreria","DescTesoreria","DescEntidad","Beneficiario"], axis=1)
  
  df["Descripcion"] = df["Descripcion"].fillna("")
  df["ClaseReg"] = df["ClaseReg"].fillna("Indefinido")
  
  
  Class = list(df.Class.unique())
  clases = {val:Class.index(val) for val in Class}
  def get_class(val):
      return clases[val]
    
  df['target'] = df['Class'].apply(get_class)
  # df["texto_limpio"] = df["Descripcion"].apply(pre_procesamiento_texto)
  
  return df

def featureEngineering(df):
  """
  Creamos features de texto y features de pesos sobre el texto limpio.
  """
  # features de texto -----------------------------------------------------------------------
  df['Descripcion'] = df['Descripcion'].astype(str)   
  df['text_size'] = df['Descripcion'].str.len()
  df['text_words_count'] = df['Descripcion'].apply(lambda x: len(x.split()))  
  
  # conteo de palabras -----------------------------------------------------------------------
  dictOfWords = {}
  for target in df.target.unique():
    df_target = df[df["target"]==target]
    all_descriptions = ' '.join(df_target['Descripcion'].dropna())  # Concatenar todas las descripciones
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(all_descriptions)

    clean_text = []
    for token in doc:
        if (
            not token.is_stop  # No incluir stopwords
            and not token.is_punct  # No incluir puntuación
            and not token.like_num  # Opción para excluir números si prefieres solo palabras
        ):
            

         
                
            
            #clean_text.append(token.lemma_.lower())
            clean_text.append(str(token.lemma_).lower())


    
    
    # Filtrar stopwords y no palabras alfa
    freq_of_words = pd.Series(clean_text).value_counts()
    
    


    dic_words = freq_of_words.to_dict()
    dictOfWords[str(target)] = dic_words
    df[f'pesos_{target}'] = df['Descripcion'].apply(pesos,dic_words=dic_words)
      
  return df



  
def pesos(texto, dic_words):
  texto = texto.lower()
  palabras = texto.split(' ')
  score = 0
  for palabra in palabras:
      if palabra in dic_words.keys():
          score += dic_words[palabra]
  return score


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


