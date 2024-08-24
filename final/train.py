import pandas as pd
import traceback
import warnings
import lightgbm as lgb
import optuna
from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


warnings.filterwarnings("ignore")

def model_lightgbm(study_name,ntrials):
  try:
    # leer datos 
    df = pd.read_csv(f"models/lgbm/{study_name}/df_train.csv")
    bbdd = "sqlite:///optuna.sqlite3"
    
    # convertir columnas a nro
    for col in df.columns:
      if df[col].dtype == 'object':
       df[col] = pd.to_numeric(df[col], errors='ignore')
    

    X = df.drop(columns=["target"])
    y = df["target"]
    
    
    SEED = 12345
    TEST_SIZE = 0.2
    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

   
    kappa_scorer = make_scorer(cohen_kappa_score)
    def objective(trial):
        param = {
            'objective': 'multiclass',
            'num_class': len(set(y)),  # Número de clases
            'metric': 'multi_logloss',  # Esto es solo para LightGBM; la métrica de optimización será accuracy
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
            'verbose': -1,
        }
        
        # Crear el dataset de LightGBM
        model = lgb.LGBMClassifier(**param,verbose_eval=False)
        
        # Realizar validación cruzada usando Kappa como métrica
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

  
