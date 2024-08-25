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
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin

try:
  STOPWORDS = set(stopwords.words('spanish'))
  nlp = spacy.load("es_core_news_sm")
except Exception as e:
  print(e)
  # !pip install spacy
  # !python -m spacy download es_core_news_sm
  # STOPWORDS = set(stopwords.words('spanish'))
  # nlp = spacy.load("es_core_news_sm")

from nltk.corpus import stopwords

warnings.filterwarnings("ignore")

def train_red_neuronal(path_df, study_name, ntrials, flag_generar_archivo=False):
  """
  Entrenamamos un modelo de red neuronal secuencial con Keras.
  """
  try:
    # leer datos
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/rn", exist_ok=True)
    os.makedirs(f"models/rn/{study_name}", exist_ok=True)    
    bbdd = "sqlite:///optuna.sqlite3"
    
    if( flag_generar_archivo ):
      # df_rnn = preprocessing + feature engineering ------------------------------------
      print(f"{datetime.now()} - Inicio preprocesamiento\n")
      df = preprocessing(path_df)
      print(f"{datetime.now()} - Inicio feature engineering\n")
      df = featureEngineering(df)
      
      df.to_csv(f"../datasets/df_rnn.csv", index=False, sep=";")
    
    
    
    # abrimos directamente el archiv
    df = pd.read_csv(path_df, sep=";")
    
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df, ['Descripcion', 'texto_limpio'])
    text_colummns = ["texto_limpio"]
    pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
    numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
    df[text_colummns] = df[text_colummns].fillna('')
    
    # Definir los transformadores para el pipeline
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
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_columns),
            ('text', TfidfVectorizer(max_features=10000), text_colummns[0])
        ],
        remainder='drop',
        sparse_threshold=0  # Forzamos a que todas las salidas sean matrices densas
    )
    
    final_columns = numeric_columns + categorical_columns + text_colummns + pesos_columns
    X = df[final_columns]
    y = df["target"].values
    
    
    
    SEED = 12345
    TEST_SIZE = 0.2
    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    
    
    
    
    
    # Ajuste para obtener input_dim
    # preprocessor.fit(X_train)  # Aseguramos que el preprocesador esté ajustado
    # X_transformed = preprocessor.transform(X_train)
    # input_dim = X_transformed.shape[1]  # Calcula el número de características después del preprocesamiento
    
    # Definir un clasificador compatible con Scikit-Learn
    class KerasNN(BaseEstimator, ClassifierMixin):
        def __init__(self, input_dim, output_dim, n_layers=1, units=64, activation='relu', dropout=0.2, learning_rate=1e-3, epochs=20, batch_size=32):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.n_layers = n_layers
            self.units = units
            self.activation = activation
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.model = None

        def build_model(self):
            model = Sequential()
            for i in range(self.n_layers):
                if i == 0:
                    model.add(Dense(self.units, input_shape=(self.input_dim,)))
                else:
                    model.add(Dense(self.units))
                model.add(Activation(self.activation))
                model.add(Dropout(self.dropout))
            model.add(Dense(self.output_dim))
            model.add(Activation('softmax'))
            
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(learning_rate=self.learning_rate),
                          metrics=['accuracy'])
            return model

        def fit(self, X, y):
            #y = to_categorical(y)
            self.model = self.build_model()
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            return self

        def predict(self, X):
            return np.argmax(self.model.predict(X), axis=1)

        def predict_proba(self, X):
            return self.model.predict(X)

    def build_nn(trial, input_dim, output_dim):
        n_layers = trial.suggest_int('n_layers', 1, 5)
        units = trial.suggest_int('units', 32, 512)
        activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh'])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        epochs = trial.suggest_int('epochs', 10, 50)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        return KerasNN(
            input_dim=input_dim, 
            output_dim=output_dim, 
            n_layers=n_layers, 
            units=units, 
            activation=activation, 
            dropout=dropout, 
            learning_rate=learning_rate, 
            epochs=epochs, 
            batch_size=batch_size
        )

    
    
    def objective(trial):
      
      preprocessor.fit(X_train)  # Aseguramos que el preprocesador esté ajustado
      X_transformed = preprocessor.transform(X_train)
      input_dim = X_transformed.shape[1]
      
      # Calcular dinámicamente el número de clases
      output_dim = len(np.unique(y))

      # Crear el modelo Keras con los hiperparámetros sugeridos
      nn_model = build_nn(trial, input_dim=input_dim, output_dim=output_dim)

      # Pipeline de preprocesamiento y modelo
      model_pipeline = Pipeline([
          ('preprocessor', preprocessor),
          ('model', nn_model)
      ])
      
      # Validación cruzada
      kappa_scores = cross_val_score(
          model_pipeline, X_train, y_train, 
          cv=3, 
          scoring=make_scorer(cohen_kappa_score, needs_proba=True)
      )
      
      return kappa_scores.mean()
    

    # Crear un estudio y optimizar
    print(f"{datetime.now()} - Inicio optimizacion de hiperparametros \n")
    study = optuna.create_study(direction='maximize', 
                                storage=bbdd,  # Specify the storage URL here.
                                study_name=f"rn_{study_name}",
                                load_if_exists=True)
    study.optimize(objective, n_trials=ntrials)

 
    # Imprimir los mejores hiperparámetros encontrados
    print('Mejores hiperparámetros:', study.best_params)
    print('Mejor valor de Kappa:', study.best_value)
        
    
    
  except Exception as e:
    tb = traceback.format_exc()
    print(f"Se produjo un error: {e}")
    print(f"Detalles del error:\n{tb}")

  



def preprocessing(path_df): 
  """
  Eliminamos columnas que no sirven, Imputamos NANs, convertimos Class a numérico y limpiamos el texto.
  """
  df = pd.read_excel(path_df)
  
  df.drop(columns=["DescCuenta","NTesoreria","DescTesoreri­a","DescEntidad","Beneficiario"], axis=1)
  
  df["Descripcion"] = df["Descripcion"].fillna("")
  df["ClaseReg"] = df["ClaseReg"].fillna("Indefinido")
  
  
  Class = list(df.Class.unique())
  clases = {val:Class.index(val) for val in Class}
  def get_class(val):
      return clases[val]
    
  df['target'] = df['Class'].apply(get_class)
  df["texto_limpio"] = df["Descripcion"].apply(pre_procesamiento_texto)
  
  return df
  

def featureEngineering(df):
  """
  Creamos features de texto y features de pesos sobre el texto limpio.
  """
  # features de texto -----------------------------------------------------------------------
  df['texto_limpio'] = df['texto_limpio'].astype(str)   
  df['text_size'] = df['texto_limpio'].str.len()
  df['text_words_count'] = df['texto_limpio'].apply(lambda x: len(x.split()))  
  
  # conteo de palabras -----------------------------------------------------------------------
  dictOfWords = {}
  for target in df.target.unique():
    df_target = df[df["target"]==target]
    all_descriptions = ' '.join(df_target['texto_limpio'].dropna())  # Concatenar todas las descripciones

    # Tokenización y eliminación de stopwords
    stop_words = set(stopwords.words('spanish')) 
    word_tokens = word_tokenize(all_descriptions.lower())  # Tokenización y convertir a minúsculas
    filtered_words = [word for word in word_tokens if word.isalnum() and word not in stop_words]  # Filtrar stopwords y no palabras alfa
    freq_of_words = pd.Series(filtered_words).value_counts()
    
    
    dic_words = freq_of_words.to_dict()
    dictOfWords[str(target)] = dic_words
    df[f'pesos_{target}'] = df['texto_limpio'].apply(pesos,dic_words=dic_words)
      
  return df


def pre_procesamiento_texto(text):
  # Quito simbolos
  texto = solo_numeros_y_letras(text)

  # tokenizacion
  texto = separar_texto_de_numeros(texto)

  # Elimino espacios de mas
  texto = eliminar_espacios_adicionales(texto)

  # Elimino stopwords
  texto = remove_stopwords(texto)

  # Lematizacion
  texto = lematizacion(texto)

  # Solo palabras y numeros con minimo de 3 caracteres
  texto = filtrar_palabras_numeros(texto)

  # Eliminar palabras frecuentas: RES-CTA- etc.
  # texto = eliminar_palabras(texto)

  return texto


def solo_numeros_y_letras(text):
  # Reemplazar todo lo que no sea letra o número por un espacio
  text_limpio = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ]', ' ', str(text))
  return text_limpio

def separar_texto_de_numeros(texto):
    # Expresión regular para insertar un espacio entre letras y números
    texto = re.sub(r'([a-zA-Z]+)(\d+)', r'\1 \2', texto)
    texto = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', texto)
    return texto

def eliminar_espacios_adicionales(texto):
    # Reemplazar múltiples espacios por un solo espacio
    texto_limpio = ' '.join(texto.split())
    return texto_limpio
  

def remove_stopwords(text):
    """
    Elimino stopwords en español.
    """
    return ' '.join([word for word in text.split() if word.lower() not in STOPWORDS])


def lematizacion(text):
  """
  no stop words + lematizacion
  """
  clean_text = []
  
  for token in nlp(text):
    if (
        not token.is_stop             # Excluir stop words
        and (token.is_alpha or token.is_digit)  # Incluir solo letras o números
        and not token.is_punct        # Excluir signos de puntuación
        and not token.like_url        # Excluir URLs
    ):
        clean_text.append(token.lemma_.upper())  

  return " ".join(clean_text)

def filtrar_palabras_numeros(texto):
    # Expresión regular para encontrar palabras o números con al menos 3 caracteres
    palabras_filtradas = re.findall(r'\b\w{3,}\b', texto)
    return " ".join(palabras_filtradas)
  

def eliminar_palabras(texto):
  texto = texto.upper()
  texto = re.findall(r"(?!CTA)(?!RES)(?!PAGO)(?!AGO)(?!PAG)[A-Z0-9]{3,}", texto)
  texto = list(dict.fromkeys(texto))
  texto = " ".join(texto).strip()
  return texto
  
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


