# Librerias basicas 
import re
import spacy
from nltk.corpus import stopwords
import pandas as pd
import traceback
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


import warnings
warnings.filterwarnings("ignore")

def preprocess(path_df):
  try:
    # leer datos
    df = pd.read_excel(path_df)
    # limpieza completa del texto
    
    df["Descripcion"] = df["Descripcion"].fillna("")
    Class = list(df.Class.unique())
    clases = {val:Class.index(val) for val in Class}
    def get_class(val):
        return clases[val]

    df['target'] = df['Class'].apply(get_class)
    df["texto_limpio"] = df["Descripcion"].apply(pre_procesamiento_texto)
    df.to_csv("df_preprocesado.csv", index=False)
  except Exception as e:
    tb = traceback.format_exc()
    print(f"Se produjo un error: {e}")
    print(f"Detalles del error:\n{tb}")

  
  
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
