import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import traceback
from sklearn.preprocessing import StandardScaler


def fe():
  try:
    
    # leemos
    df = pd.read_csv("../datasets/df_preprocessing.csv", sep=';')
    
    # Creamos variables de texto
    df = crear_variablesTexto(df)
    
    # eliminamos variables que no sirven
    df.drop(columns=['Descripcion'], inplace=True)

    # onehot encoding 
    df = aplicar_ohe(df)
        
    # conteo de palabras
    df = asignar_pesos_al_texto(df)    
    
    # Estandarizacion de pesos
    df = estandarizar_pesos(df)
    
    # Guardamos
    df.to_csv("../datasets/df_final.csv", index=False, sep=';')

  except Exception as e:
    tb = traceback.format_exc()
    print(f"Se produjo un error: {e}")
    print(f"Detalles del error:\n{tb}")
    return None
  

def crear_variablesTexto(df):
  """
  """
  df['Descripcion'] = df['Descripcion'].astype(str)
  df['texto_limpio'] = df['texto_limpio'].astype(str)
  
  df['descripcion_size'] = df['Descripcion'].str.len()
  df['descripcion_words_count'] = df['Descripcion'].apply(lambda x: len(x.split()))  

  df['text_size'] = df['texto_limpio'].str.len()
  df['text_words_count'] = df['texto_limpio'].apply(lambda x: len(x.split()))  
  return df



def aplicar_ohe(df):
  """
  """
  categ = ['Tipo_comp','Tipo_Reg','Clase_Reg','Tipo_cta']
  for col in categ:
      df = pd.concat([df,pd.get_dummies(df[col],prefix=col, prefix_sep='_')],axis=1)
      df.drop(col, axis=1, inplace=True)
  return df


def pesos(texto, dic_words):
  """
  """
  texto = texto.lower()
  palabras = texto.split(' ')
  score = 0
  for palabra in palabras:
      if palabra in dic_words.keys():
          score += dic_words[palabra]
  return score



def asignar_pesos_al_texto(df):    
  """
  """  
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
  
  with open('dict_words.json', 'w') as file:
    json.dump(dictOfWords, file, indent=4)
  return df


def estandarizar_pesos(df):
  """
  """
  # Seleccionar las columnas que comienzan con 'pesos_'
  pesos_cols = [col for col in df.columns if col.startswith('pesos_')]
  # Aplicar StandarScaler
  scaler = StandardScaler()
  df[pesos_cols] = scaler.fit_transform(df[pesos_cols])
  return df