import pandas as pd
import json
import joblib
# Funciones auxiliares sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score  # Metricas
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
import traceback
from datetime import datetime

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


def test(path_df,study_name):
    try:
        df = pd.read_csv(path_df)
        bbdd = "sqlite:///optuna.sqlite3"
    
        SEED = 12345
        TEST_SIZE = 0.2
        params_work = json.load(open(f"models/lgbm/{study_name}/params_{study_name}.json")) 
        features = params_work["features"]
        
        X = df[features]
        y = df.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
        
        print("Cargando el modelo optimo")
        study = optuna.load_study(study_name=study_name, storage=bbdd)
        # Mostrar los mejores hiperparámetros encontrados
        print("Mejores hiperparámetros:", study.best_params)
        
        best_model = lgb.LGBMClassifier(**study.best_params)
        best_model.fit(X_train, y_train)
        joblib.dump(best_model, f'models/lgbm/{study_name}/model_{study_name}.pkl')
        
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
        
        with open(f'models/lgbm/{study_name}/metrics_{study_name}.txt', 'w') as f:
            f.write(f'study_name: {study_name}\n')
            f.write(f'Fecha y hora: {datetime.now()}\n')
            f.write(f'Kappa: {test_kappa}\n')
            f.write(f'Accuracy: {test_accuracy}\n')
            f.write(f'dimensióm Test: {X_test.shape}\n')
            f.write(f'Mejores hiperparámetros: {study.best_params}\n')
            

        # Crear la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plotcm(cm,f'models/lgbm/{study_name}/confusion_matrix_{study_name}.png')
        ploimportance(importance_df,f'models/lgbm/{study_name}/importance_{study_name}.png')

    except Exception as e:
      tb = traceback.format_exc()
      print(f"Se produjo un error: {e}")
      print(f"Detalles del error:\n{tb}")