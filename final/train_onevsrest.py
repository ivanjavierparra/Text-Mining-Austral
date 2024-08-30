import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import traceback
import warnings
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import optuna
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import json
import os
import importlib 
import archivos
warnings.filterwarnings("ignore")
BBDD = "sqlite:///optuna_lightgbm.sqlite3"
TRIALS = 50
SEED = 12345
TEST_SIZE = 0.2



def modelo_base():
    """
    Solo variables categoricas y numericas. No incluye pesos ni tf-idf.
    """
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
    
    # Kappa
    kappa_scorer = make_scorer(cohen_kappa_score)
    
    def cv_es_ovr_objective(trial):

         
        # Definir los hiperparámetros a optimizar
        C = trial.suggest_loguniform('C', 1e-5, 100)
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        
        # Elegir entre dos conjuntos de parámetros compatibles
        param_set = trial.suggest_categorical('param_set', ['set1', 'set2'])
        
        if param_set == 'set1':
            penalty = 'l2'
            solver = trial.suggest_categorical('solver', ['lbfgs', 'newton-cg', 'sag'])
        else:  # set2
            penalty = 'l1'
            solver = 'liblinear'

        
        # Crear y entrenar el modelo
        model_ovr = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=max_iter,
                multi_class='ovr',
                random_state=SEED
            )
        )
   
   
        # Realizar validación cruzada usando Kappa como métrica
        kappa = cross_val_score(model_ovr, X_train, y_train, cv=3, scoring=kappa_scorer).mean()
        
        return kappa

  

    #Genero estudio
    study = optuna.create_study(direction='maximize', 
                                    storage=BBDD,  # Specify the storage URL here.
                                    study_name=f"onevsrest_{STUDY_NAME}",
                                    load_if_exists=True)
        
    #Corro la optimizacion
    study.optimize(cv_es_ovr_objective, n_trials=TRIALS)
    
    
    # guardamos mejor modelo
    print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
    best_params = study.best_param
    
    # Extraer los parámetros específicos
    C = best_params['C']
    max_iter = best_params['max_iter']

    # Determinar penalty y solver basado en el mejor param_set
    if best_params['param_set'] == 'set1':
        penalty = 'l2'
        solver = 'lbfgs'  # Asumiendo que usaste 'lbfgs' para el set1 en la optimización
    else:  # set2
        penalty = 'l1'
        solver = 'liblinear'

    # Crear el modelo final con los mejores hiperparámetros
    best_model = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            multi_class='ovr',
            random_state=SEED
        )
    )
    
    print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, f'models/onevsrest/{STUDY_NAME}/model_{STUDY_NAME}.pkl')           
    print(f"[{datetime.now()}] - Se ha guardado el modelo en models/onevsrest/{STUDY_NAME}/model_{STUDY_NAME}.pkl \n")       
    
    
def modelo_text_mining():
    """
    Acá usamos los pesos calculados sobre el texto sin tf-idf
    """    
    try:
        # bdd
        STUDY_NAME = "modelo_text_mining"
        
        # Leemos
        df = archivos.modelo_text_mining()
        
        # Eliminamos columnas que nos nos sirven
        df.drop(columns=['texto_limpio'], inplace=True)
        
        # Preparar los datos
        final_columns = [elemento for elemento in df.columns if elemento != 'target']
        X = df[final_columns]
        y = df["target"]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
        
        # kappa
        kappa_scorer = make_scorer(cohen_kappa_score)
        
        def cv_es_ovr_objective(trial):

            # Definir los hiperparámetros a optimizar
            C = trial.suggest_loguniform('C', 0.1, 1)
            max_iter = trial.suggest_int('max_iter', 100, 120)
            
            # Elegir entre dos conjuntos de parámetros compatibles
            param_set = trial.suggest_categorical('param_set', ['set1', 'set2'])
            
            if param_set == 'set1':
                penalty = 'l2'
                solver = 'lbfgs'  # Fijamos el solver para reducir el espacio de búsqueda
            else:  # set2
                penalty = 'l1'
                solver = 'liblinear'
            
            
            # Crear y entrenar el modelo
            model_ovr = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    penalty=penalty,
                    solver=solver,
                    max_iter=max_iter,
                    multi_class='ovr',
                    random_state=SEED
                )
            )
            
            # Realizar validación cruzada usando Kappa como métrica
            kappa = cross_val_score(model_ovr, X_train, y_train, cv=3, scoring=kappa_scorer).mean()

            return kappa

        #Genero estudio
        study = optuna.create_study(direction='maximize', 
                                        storage=BBDD,  # Specify the storage URL here.
                                        study_name=f"onevsrest_{STUDY_NAME}",
                                        load_if_exists=True)
            
        #Corro la optimizacion
        study.optimize(cv_es_ovr_objective, n_trials=TRIALS)
        
        
        # guardamos mejor modelo
        print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
        best_params = study.best_param
        
        # Extraer los parámetros específicos
        C = best_params['C']
        max_iter = best_params['max_iter']

        # Determinar penalty y solver basado en el mejor param_set
        if best_params['param_set'] == 'set1':
            penalty = 'l2'
            solver = 'lbfgs'  # Asumiendo que usaste 'lbfgs' para el set1 en la optimización
        else:  # set2
            penalty = 'l1'
            solver = 'liblinear'

        # Crear el modelo final con los mejores hiperparámetros
        best_model = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=max_iter,
                multi_class='ovr',
                random_state=SEED
            )
        )
        
        print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
        best_model.fit(X_train, y_train)
        joblib.dump(best_model, f'models/onevsrest/{STUDY_NAME}/model_{STUDY_NAME}.pkl')           
        print(f"[{datetime.now()}] - Se ha guardado el modelo en models/onevsrest/{STUDY_NAME}/model_{STUDY_NAME}.pkl \n")  
    
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Se produjo un error: {e}")
        print(f"Detalles del error:\n{tb}")






def modelo_completo():
    """
    Acá usamos pesos + tf-idf.
    """
    try:
        
        # bdd
        STUDY_NAME = "modelo_completo"
        
        # Leemos
        df = archivos.modelo_text_mining()
        
        # columnas 
        numeric_columns = get_numeric_columns(df)
        categorical_columns = get_categorical_columns(df, ['texto_limpio'])
        text_colummns = ["texto_limpio"]
        pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
        numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
        final_columns = numeric_columns + categorical_columns + text_colummns + pesos_columns       
        df['texto_limpio'] = df['texto_limpio'].fillna('')

 
        # Preparar los datos
        X = df[final_columns]
        y = df["target"]

    
        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


        # Definir los transformadores para el pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_columns),  # No se aplica ningún preprocesamiento a las variables numéricas
                ('pesos', 'passthrough', pesos_columns),
                ('cat', 'passthrough', categorical_columns),
                ('text', TfidfVectorizer(max_features=10000), "texto_limpio")
            ],
            remainder='drop'
        )
        
        # kappa
        kappa_scorer = make_scorer(cohen_kappa_score)

        def cv_es_ovr_objective(trial):

            # Definir los hiperparámetros a optimizar
            C = trial.suggest_loguniform('C', 0.1, 10)
            max_iter = trial.suggest_int('max_iter', 100, 500)
            
            # Elegir entre dos conjuntos de parámetros compatibles
            param_set = trial.suggest_categorical('param_set', ['set1', 'set2'])
            
            if param_set == 'set1':
                penalty = 'l2'
                solver = 'lbfgs'  # Fijamos el solver para reducir el espacio de búsqueda
            else:  # set2
                penalty = 'l1'
                solver = 'liblinear'
            

            # Crear y entrenar el modelo
            model_ovc = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    penalty=penalty,
                    solver=solver,
                    max_iter=max_iter,
                    multi_class='ovr',
                    random_state=SEED
                )
            )
            
            # Crear pipeline completo
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model_ovc)
            ])
            
            # Realizar validación cruzada usando Kappa como métrica
            kappa = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=kappa_scorer).mean()
            
            return kappa

  

        #Genero estudio
        study = optuna.create_study(direction='maximize', 
                                        storage=BBDD,  # Specify the storage URL here.
                                        study_name=f"onevsrest_{STUDY_NAME}",
                                        load_if_exists=True)
            
        #Corro la optimizacion
        study.optimize(cv_es_ovr_objective, n_trials=TRIALS)
        
        
        
        # guardamos mejor modelo
        print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
        best_params = study.best_param
        
        # Extraer los parámetros específicos
        C = best_params['C']
        max_iter = best_params['max_iter']

        # Determinar penalty y solver basado en el mejor param_set
        if best_params['param_set'] == 'set1':
            penalty = 'l2'
            solver = 'lbfgs'  # Asumiendo que usaste 'lbfgs' para el set1 en la optimización
        else:  # set2
            penalty = 'l1'
            solver = 'liblinear'

        # Crear el modelo final con los mejores hiperparámetros
        best_model = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=max_iter,
                multi_class='ovr',
                random_state=SEED
            )
        )
        
        # Crear pipeline completo
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', best_model)
        ])
        
        print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
        pipeline.fit(X_train, y_train)
        joblib.dump(best_model, f'models/onevsrest/{STUDY_NAME}/model_{STUDY_NAME}.pkl')           
        print(f"[{datetime.now()}] - Se ha guardado el modelo en models/onevsrest/{STUDY_NAME}/model_{STUDY_NAME}.pkl \n")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Se produjo un error: {e}")
        print(f"Detalles del error:\n{tb}")
  



def modelo_tfidf():
    """
    Acá usamos solo tf-idf.
    """   
    # bdd
    STUDY_NAME = "modelo_tfidf"
    
    # Leemos
    df = archivos.modelo_text_mining()
    
    # columnas      
    df['texto_limpio'] = df['texto_limpio'].fillna('hola')
    
    # Preparar los datos
    y = df["target"]

    # Convertir texto a TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['texto_limpio'])

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    
    # kappa
    kappa_scorer = make_scorer(cohen_kappa_score)
    
    def cv_es_ovr_objective(trial):
       
        # Definir los hiperparámetros a optimizar
        # Definir los hiperparámetros a optimizar
        C = trial.suggest_loguniform('C', 0.1, 10)
        max_iter = trial.suggest_int('max_iter', 100, 500)
        
        # Elegir entre dos conjuntos de parámetros compatibles
        param_set = trial.suggest_categorical('param_set', ['set1', 'set2'])
        
        if param_set == 'set1':
            penalty = 'l2'
            solver = 'lbfgs'  # Fijamos el solver para reducir el espacio de búsqueda
        else:  # set2
            penalty = 'l1'
            solver = 'liblinear'
            
        # Crear y entrenar el modelo
        model_ovc = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=max_iter,
                multi_class='ovr',
                random_state=SEED
            )
        )

        # Realizar validación cruzada usando Kappa como métrica
        kappa = cross_val_score(model_ovc, X_train, y_train, cv=3, scoring=kappa_scorer).mean()
        
        return kappa

    #Genero estudio
    study = optuna.create_study(direction='maximize', 
                                    storage=BBDD,  # Specify the storage URL here.
                                    study_name=f"onevsrest_{STUDY_NAME}",
                                    load_if_exists=True)
        
    #Corro la optimizacion
    study.optimize(cv_es_ovr_objective, n_trials=TRIALS)

    # Obtener los mejores hiperparámetros
    best_params = study.best_params
    print("Mejores hiperparámetros:", best_params)

     # guardamos mejor modelo
    print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
    best_params = study.best_param
    
    # Extraer los parámetros específicos
    C = best_params['C']
    max_iter = best_params['max_iter']

    # Determinar penalty y solver basado en el mejor param_set
    if best_params['param_set'] == 'set1':
        penalty = 'l2'
        solver = 'lbfgs'  # Asumiendo que usaste 'lbfgs' para el set1 en la optimización
    else:  # set2
        penalty = 'l1'
        solver = 'liblinear'

    # Crear el modelo final con los mejores hiperparámetros
    best_model = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            multi_class='ovr',
            random_state=SEED
        )
    )
    
    
    print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, f'models/onevsrest/{STUDY_NAME}/model_{STUDY_NAME}.pkl')           
    print(f"[{datetime.now()}] - Se ha guardado el modelo en models/onevsrest/{STUDY_NAME}/model_{STUDY_NAME}.pkl \n")




  


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