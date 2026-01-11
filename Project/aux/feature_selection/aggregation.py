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

def aggregate_importances_across_folds(fold_importances, top_k=50):
    """
    Agrega importancias de características obtenidas en validación cruzada.

    Esta función combina las importancias de cada fold para obtener:
        - Un recuento de "votos" de las features más relevantes.
        - La importancia media de cada característica a lo largo de los folds.

    El objetivo es identificar qué variables son consistentemente relevantes
    independientemente del fold, lo cual aporta estabilidad y robustez al análisis.

    Parámetros
    ----------
    fold_importances : list of arrays
        Lista donde cada elemento es un vector de importancias normalizadas
        correspondiente a un fold de validación cruzada.
    top_k : int
        Número de características más importantes a considerar en cada fold
        para el recuento de votos.

    Return
    ------
    results : dict
        Diccionario con:
            - votes: número de veces que cada feature aparece en el top_k
            - mean_importances: importancia media de cada feature
    """

    # Número total de folds y número de características
    n_folds = len(fold_importances)
    n_features = len(fold_importances[0])

    # Inicializar contadores:
    # votes -> cuántas veces aparece cada feature en el top_k
    # mean_importances -> suma acumulada de importancias para luego promediar
    votes = np.zeros(n_features)
    mean_importances = np.zeros(n_features)

    # Recorrer importancias de cada fold
    for imp in fold_importances:

        # Ranking descendente de importancias (de mayor a menor)
        ranking = np.argsort(imp)[::-1]

        # Seleccionar las top_k características más importantes del fold
        top_features = ranking[:top_k]

        # Incrementar votos para las características seleccionadas
        votes[top_features] += 1

        # Acumular importancias para calcular la media posteriormente
        mean_importances += imp

    # Calcular la importancia media dividiendo entre el número de folds
    mean_importances /= n_folds

    # Devolver resultados agregados
    return {
        "votes": votes,
        "mean_importances": mean_importances
    }


def aggregate_importances_across_models(model_importance_dicts):
    """
    Agrega importancias de características procedentes de múltiples modelos.

    Esta función combina los resultados obtenidos tras la validación cruzada
    de distintos modelos (por ejemplo: RF, XGB, CatBoost, LR) para obtener:

        - total_votes:
            Número total de veces que cada característica aparece entre las más
            relevantes (según el recuento de votos de cada modelo).
        - total_importances:
            Importancia media acumulada de cada característica a través de todos
            los modelos.
        - final_ranking:
            Ranking global de características, ordenado primero por votos y,
            en caso de empate, por importancia media.

    El objetivo es identificar qué variables son consistentemente relevantes
    no solo dentro de un modelo, sino a través de varios algoritmos distintos,
    lo cual aporta robustez y estabilidad al análisis de importancia.

    Parámetros
    ----------
    model_importance_dicts : list of dict
        Lista de diccionarios, uno por modelo, donde cada diccionario contiene:
            - "votes": array con votos por característica
            - "mean_importances": array con importancias medias por característica

    Return
    ------
    results : dict
        Diccionario con:
            - total_votes: votos agregados entre modelos
            - total_importances: importancias agregadas entre modelos
            - final_ranking: ranking global de características
    """

    # Número de características (se asume que todos los modelos tienen el mismo número)
    n_features = len(model_importance_dicts[0]["votes"])

    # Inicializar acumuladores globales
    total_votes = np.zeros(n_features)
    total_importances = np.zeros(n_features)

    # Sumar votos e importancias de cada modelo
    for d in model_importance_dicts:
        total_votes += d["votes"]
        total_importances += d["mean_importances"]

    # ============================
    # Ranking final de características
    # ============================
    # Se usa lexsort con dos claves:
    #   1) -total_votes        → primero ordenar por votos (descendente)
    #   2) -total_importances  → en caso de empate, ordenar por importancia media
    #
    # NOTA: lexsort ordena de derecha a izquierda, por eso el orden de argumentos
    final_ranking = np.lexsort((-total_importances, -total_votes))

    # Devolver resultados agregados
    return {
        "total_votes": total_votes,
        "total_importances": total_importances,
        "final_ranking": final_ranking
    }
