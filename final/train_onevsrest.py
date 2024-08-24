import pandas as pd
import traceback
import warnings
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import optuna
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.model_selection import train_test_split, cross_val_score
import json
import os

warnings.filterwarnings("ignore")

def model_onevsrest(path_df, study_name, ntrials):
  try:
    # leer datos
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/onevsrest", exist_ok=True)
    os.makedirs(f"models/onevsrest/{study_name}", exist_ok=True)    
    df = pd.read_csv(path_df)
    bbdd = "sqlite:///optuna.sqlite3"
    
    # convertir columnas a nro
    for col in df.columns:
      if df[col].dtype == 'object':
       df[col] = pd.to_numeric(df[col], errors='ignore')
    
    # filtrar features
    features = []
    for x in enumerate(df.dtypes):
        if x[1] in ["float64","int64","bool"]:
            features.append(df.columns[x[0]])
    
    params = {
        "study_name": study_name,
        "features": features
    }
    
    with open(f'models/onevsrest/{study_name}/params_{study_name}.json','w') as file:
      json.dump(params,file,indent=4)
      
    
    X = df.drop(columns=["target"])
    y = df["target"]
    
    
    SEED = 12345
    TEST_SIZE = 0.2
    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

   
   
    # Definir la función objetivo para Optuna
    def objective(trial):
        
        # Hiperparámetros a optimizar
        C = trial.suggest_loguniform('C', 1e-5, 100.0)
        solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear'])
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        
        # Crear y ajustar el modelo
        model = OneVsRestClassifier(LogisticRegression(C=C, solver=solver, max_iter=max_iter, multi_class='ovr'))


        # Realizar validación cruzada usando Kappa como métrica
        kappa_scorer = make_scorer(cohen_kappa_score)
        kappa = cross_val_score(model, X_train, y_train, cv=3, scoring=kappa_scorer).mean()

        return kappa


    # Crear un estudio y optimizar
    study = optuna.create_study(direction='maximize', 
                                storage=bbdd,  # Specify the storage URL here.
                                study_name=study_name,
                                load_if_exists=True)
    study.optimize(objective, n_trials=ntrials)

 

    
    
    
  except Exception as e:
    tb = traceback.format_exc()
    print(f"Se produjo un error: {e}")
    print(f"Detalles del error:\n{tb}")

  
