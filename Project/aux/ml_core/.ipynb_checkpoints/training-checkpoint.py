import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import optuna
from scipy import stats
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, get_scorer, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from kneed import KneeLocator

def train_with_best_params(original_estimator, best_params, X_train, y_train, rng_seed, folds=None):
    """
    Entrena un modelo usando los mejores hiperparámetros encontrados previamente
    y evalúa su rendimiento mediante validación cruzada emparejada.

    Esta función:
        - Si se le pasa una lista de folds:
            - Reentrena el modelo en cada fold usando los mejores hiperparámetros.
            - Calcula el mejor F1-score por fold (optimizando el umbral).
            - Extrae importancias de características por fold.
            - Genera rankings de características por fold.
        - Entrena un modelo final usando TODO el conjunto de entrenamiento.

    Parámetros
    ----------
    original_estimator : sklearn estimator
        Modelo base (LogisticRegression, RandomForestClassifier, XGBClassifier, CatBoostClassifier, etc.)
    best_params : dict
        Hiperparámetros óptimos obtenidos mediante Optuna.
    X_train : pandas.DataFrame
        Matriz de entrenamiento.
    y_train : array-like
        Etiquetas de entrenamiento.
    folds : list of tuples
        Lista de folds precomputados (train_idx, valid_idx), compartidos entre modelos
        para garantizar comparabilidad y reproducibilidad.

    Return
    ------
    results : dict
        Diccionario con:
            - cv_scores: lista de F1 por fold
            - mean_f1: F1 medio
            - interval: (min_f1, max_f1)
            - fold_importances: importancias por fold
            - fold_rankings: ranking de features por fold
            - final_model: modelo final entrenado con todos los datos
    """
    # Si se le pasa una lista de folds, hace validación cruzada
    if folds:
        # Listas para almacenar métricas e importancias por fold
        fold_importances = []
        fold_rankings = []
        fold_scores = []
    
        # Recorrer todos los folds de validación cruzada
        for fold_idx, (train_idx, valid_idx) in enumerate(folds):
    
            # Separar datos de entrenamiento y validación del fold actual
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train[train_idx], y_train[valid_idx]
    
            # Clonar el estimador original para evitar contaminación entre folds
            model = clone(original_estimator)
            model.set_params(**best_params, random_state=rng_seed)
    
            # Entrenamiento con soporte de early stopping cuando el modelo lo permite
            if isinstance(model, XGBClassifier):
                # XGBoost requiere pasar early_stopping_rounds como parámetro del modelo
                model.set_params(early_stopping_rounds=50)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
            elif isinstance(model, CatBoostClassifier):
                # CatBoost permite early stopping directamente en el fit
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                          early_stopping_rounds=50, verbose=False)
    
            else:
                # Modelos sin early stopping
                model.fit(X_tr, y_tr)
    
            # ============================
            # Evaluación del F1 por fold
            # ============================
    
            # Obtener probabilidades del modelo
            y_proba = model.predict_proba(X_val)[:, 1]
    
            # Evaluar F1 para 101 umbrales entre 0 y 1
            thresholds = np.linspace(0, 1, 101)
            f1_scores = [f1_score(y_val, (y_proba >= t).astype(int)) for t in thresholds]
    
            # Guardar el mejor F1 del fold
            fold_scores.append(max(f1_scores))
    
            # ============================
            # Importancias de características
            # ============================
    
            # Modelos basados en árboles
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
    
            # Modelos lineales (coeficientes)
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).flatten()
    
            else:
                raise ValueError("Modelo no soporta extracción de importancias")
    
            # Normalizar importancias para comparabilidad entre folds
            importances = importances / (importances.max() + 1e-12)
    
            # Ranking descendente de importancia
            ranking = np.argsort(importances)[::-1]
    
            # Guardar resultados del fold
            fold_importances.append(importances)
            fold_rankings.append(ranking)

    # ============================
    # Entrenamiento final del modelo
    # ============================

    # Entrenar el modelo final con TODOS los datos de entrenamiento
    final_model = clone(original_estimator)
    final_model.set_params(**best_params, random_state=rng_seed)
    final_model.fit(X_train, y_train)

    # ============================
    # Construir diccionario de resultados
    # ============================
    results = {"final_model": final_model}
    
    # Incluir información dependiente de la validación cruzada si la hubo
    if folds:
        results.update({
            "cv_scores": fold_scores,
            "mean_f1": np.mean(fold_scores),
            "interval": (np.min(fold_scores), np.max(fold_scores)),
            "fold_importances": fold_importances,
            "fold_rankings": fold_rankings
        })
    
    return results