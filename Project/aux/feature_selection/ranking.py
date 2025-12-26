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

def build_feature_ranking_dataframe(feature_names, total_votes, total_importances, final_ranking):
    """
    Construye un DataFrame ordenado con:
    - nombre de la feature
    - votos agregados
    - importancia agregada
    """

    df = pd.DataFrame({
        "feature": feature_names,
        "votes": total_votes,
        "mean_importance": total_importances
    })

    # Ordenar según el ranking final (que ya está de mayor a menor)
    df = df.iloc[final_ranking].reset_index(drop=True)

    return df

def build_final_feature_dataframe(final_importances_ranking_data, feature_names, a=0.7, b=0.3):
    """
    Construye un DataFrame con:
    - feature
    - total_votes
    - total_importances
    - score = a * votos_norm + b * importancias_norm
    """

    total_votes = final_importances_ranking_data["total_votes"]
    total_importances = final_importances_ranking_data["total_importances"]

    # Normalizar votos e importancias a [0,1]
    votes_norm = (total_votes - total_votes.min()) / (total_votes.max() - total_votes.min() + 1e-12)
    importances_norm = (total_importances - total_importances.min()) / (total_importances.max() - total_importances.min() + 1e-12)

    # Score combinado
    score = a * votes_norm + b * importances_norm

    # Construir DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "total_votes": total_votes,
        "total_importances": total_importances,
        "score": score
    })

    # Ordenar por score descendente
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

    return df
