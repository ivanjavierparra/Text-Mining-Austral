import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import traceback

def fe(df):
  try:
    df
    # convertir categoricas a numericas---------------------------------------------------------

    df['texto_limpio'] = df['texto_limpio'].astype(str)
    
    # Creo nuevas variables de texto -----------------------------------------------------------
    df['text_size'] = df['texto_limpio'].str.len()
    df['text_words_count'] = df['texto_limpio'].apply(lambda x: len(x.split()))  
    
    # onehot encoding --------------------------------------------------------------------------
    categ = ['Tipo_comp','Tipo_Reg','Clase_Reg','Tipo_cta']
    for col in categ:
        df = pd.concat([df,pd.get_dummies(df[col],prefix=col, prefix_sep='_')],axis=1)
        df.drop(col, axis=1, inplace=True)
        
    # conteo de palabras -----------------------------------------------------------------------
        
    def pesos(texto, dic_words):
        texto = texto.lower()
        palabras = texto.split(' ')
        score = 0
        for palabra in palabras:
            if palabra in dic_words.keys():
                score += dic_words[palabra]
        return score
    
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

    return df, dictOfWords
    
  except Exception as e:
    tb = traceback.format_exc()
    print(f"Se produjo un error: {e}")
    print(f"Detalles del error:\n{tb}")
    return None