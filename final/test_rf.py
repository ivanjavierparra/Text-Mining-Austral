import pandas as pd
import json
import joblib
# Funciones auxiliares sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score  # Metricas
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
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

 
def pesos(texto, dic_words):
  texto = texto.lower()
  palabras = texto.split(' ')
  score = 0
  for palabra in palabras:
      if palabra in dic_words.keys():
          score += dic_words[palabra]
  return score

def test_fr(path_df, study_name):
    try:
        df = pd.read_csv(path_df)
        bbdd = "sqlite:///optuna.sqlite3"
    
        SEED = 12345
        TEST_SIZE = 0.2
        
        print(f"{datetime.now()} - Inicio preprocessing \n")
        df.drop(columns=["DescCuenta","NTesoreria","DescTesoreria","DescEntidad","Beneficiario"], axis=1)
        df["Descripcion"] = df["Descripcion"].fillna("")
        df["ClaseReg"] = df["ClaseReg"].fillna("Indefinido")
        Class = list(df.Class.unique())
        clases = {val:Class.index(val) for val in Class}
        def get_class(val):
            return clases[val]
        df['target'] = df['Class'].apply(get_class)
        print(f"{datetime.now()} - Inicio feature engineering \n")
        df['Descripcion'] = df['Descripcion'].astype(str)   
        df['text_size'] = df['Descripcion'].str.len()
        df['text_words_count'] = df['Descripcion'].apply(lambda x: len(x.split()))
        print(f"{datetime.now()} - Calculamos pesos \n")
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
        numeric_columns = get_numeric_columns(df)
        categorical_columns = get_categorical_columns(df, ['Descripcion'])
        text_colummns = ["Descripcion"]
        pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
        numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
        final_columns = numeric_columns + categorical_columns + text_colummns + pesos_columns
        df["Descripcion"] = df["Descripcion"].fillna('')


        
        X = df.drop(columns=["target"])
        y = df.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
        
        print("Cargando el modelo optimo")
        study = optuna.load_study(study_name=f"randomforest1234_{study_name}", storage=bbdd)
        # Mostrar los mejores hiperparámetros encontrados
        print("Mejores hiperparámetros:", study.best_params)
        
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


        
        
        


        

        # Mostrar los mejores hiperparámetros encontrados
        best_params = study.best_params
        best_model = RandomForestClassifier(**best_params, random_state=SEED)
        # Crear pipeline completo
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', best_model)
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(best_model, f'models/randomforest/{study_name}/model_randomforest_{study_name}.pkl')
        
        y_pred = best_model.predict(X_test)
        
        # Obtener la importancia de las características
        # importance = best_model.feature_importances_
        # print(importance)

        # Crear un DataFrame para visualizar mejor
        # importance_df = pd.DataFrame({
        #     'feature': X_train.columns,
        #     'importance': importance
        # }).sort_values(by='importance', ascending=False)

        
        # metricas
        test_kappa = cohen_kappa_score(y_test, y_pred)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Kappa en test: {test_kappa}")
        print(f"Acurracy en test: {test_kappa}")
        
        
        # with open(f'models/onevsrest/{study_name}/metrics_{study_name}.txt', 'w') as f:
        #     f.write(f'study_name: onevsrest_{study_name}\n')
        #     f.write(f'Fecha y hora: {datetime.now()}\n')
        #     f.write(f'Kappa: {test_kappa}\n')
        #     f.write(f'Accuracy: {test_accuracy}\n')
        #     f.write(f'dimensióm Train: {X_train.shape}\n')
        #     f.write(f'dimensióm Test: {X_test.shape}\n')
        #     f.write(f'Mejores hiperparámetros: {study.best_params}\n')
            

        # # Crear la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plotcm(cm,f'models/onevsrest/{study_name}/confusion_matrix_{study_name}.png')
        #ploimportance(importance_df,f'models/lgbm/{study_name}/importance_{study_name}.png')

        # Obtener los coeficientes
        # coef = best_model.coef_

        # # La importancia de las características es el valor absoluto de los coeficientes
        # feature_importance = np.abs(coef)

        # # Si quieres una lista ordenada de las características más importantes
        # sorted_indices = np.argsort(feature_importance)[::-1]
        # sorted_features = X_train.columns[sorted_indices]

        # print("Importancia de las características:", feature_importance)
        # print("Características ordenadas por importancia:", sorted_features)


    except Exception as e:
      tb = traceback.format_exc()
      print(f"Se produjo un error: {e}")
      print(f"Detalles del error:\n{tb}")


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