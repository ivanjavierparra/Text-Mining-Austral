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


def test_onevsrest(path_df, study_name):
    try:
        df = pd.read_csv(path_df)
        bbdd = "sqlite:///optuna.sqlite3"
    
        SEED = 12345
        TEST_SIZE = 0.2
        params_work = json.load(open(f"models/onevsrest/{study_name}/params_{study_name}.json")) 
        features = params_work["features"]
        df = df[features]
        X = df.drop(columns=["target"])
        y = df.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
        
        print("Cargando el modelo optimo")
        study = optuna.load_study(study_name=f"onevsrest_{study_name}", storage=bbdd)
        # Mostrar los mejores hiperparámetros encontrados
        print("Mejores hiperparámetros:", study.best_params)
        
        # Mostrar los mejores hiperparámetros encontrados
        best_params = study.best_params
        best_model = OneVsRestClassifier(LogisticRegression(**best_params, multi_class='ovr'))
        best_model.fit(X_train, y_train)
        joblib.dump(best_model, f'models/onevsrest/{study_name}/model_onevsrest_{study_name}.pkl')
        
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
        
        
        with open(f'models/onevsrest/{study_name}/metrics_{study_name}.txt', 'w') as f:
            f.write(f'study_name: onevsrest_{study_name}\n')
            f.write(f'Fecha y hora: {datetime.now()}\n')
            f.write(f'Kappa: {test_kappa}\n')
            f.write(f'Accuracy: {test_accuracy}\n')
            f.write(f'dimensióm Train: {X_train.shape}\n')
            f.write(f'dimensióm Test: {X_test.shape}\n')
            f.write(f'Mejores hiperparámetros: {study.best_params}\n')
            

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