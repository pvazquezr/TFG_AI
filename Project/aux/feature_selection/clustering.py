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

def run_elbow_on_scores(df_final, save_path="./data/figures/elbow_features_score.pdf"):
    """
    Aplica el método del codo (elbow) usando la columna 'score' del dataframe
    generado por build_final_feature_dataframe.
    """

    # Ordenar los scores de mayor a menor
    scores_sorted = np.sort(df_final["score"].values)[::-1]

    # Índices de las features (1..N)
    x = range(1, len(scores_sorted) + 1)

    # Detectar el codo
    knee = KneeLocator(
        x,
        scores_sorted,
        curve="convex",
        direction="decreasing"
    )

    elbow_point = knee.knee

    # Dibujar el gráfico
    plt.figure(figsize=(8, 6))
    plt.plot(x, scores_sorted, marker="o")
    if elbow_point is not None:
        plt.axvline(elbow_point, color="red", linestyle="--", label=f"Codo en {elbow_point}")
    plt.xlabel("Features ordenadas por score", fontsize=16)
    plt.ylabel("Score combinado (votos + importancia)", fontsize=16)
    plt.title("Curva de scores y punto de corte (codo)", fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()

    # Guardar
    plt.savefig(save_path, format="pdf", bbox_inches="tight")

    # Mostrar
    plt.show()

    print(f"El punto de corte sugerido por KneeLocator es: {elbow_point} features")

    # Seleccionar las características según el KneeLocator
    if elbow_point is not None:
        elbow_selected_features = df_final.nlargest(elbow_point, "score")["feature"].tolist()
    else:
        elbow_selected_features = []

    print(f"\n{len(elbow_selected_features)} features seleccionadas:", elbow_selected_features)

    return elbow_point, elbow_selected_features


def compute_silhouette_scores(df_final, rng_seed=1, k_min=2, k_max=6):
    """
    Calcula los silhouette scores para k-means usando la columna 'score'
    del dataframe generado por build_final_feature_dataframe.
    """

    # Matriz X para clustering: score en 1D
    X = df_final[["score"]].values

    silhouette_scores = {}

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=rng_seed)
        labels = km.fit_predict(X)
        silhouette_scores[k] = silhouette_score(X, labels)

    print("Silhouette scores por número de clusters:")
    for k, v in silhouette_scores.items():
        print(f"k={k}: {v:.4f}")

    return silhouette_scores



def run_kmeans_on_scores(df_final, n_clusters=2, rng_seed=1, save_path="./data/figures/kmeans_features_score.pdf"):
    """
    Aplica KMeans sobre la columna 'score' del dataframe final,
    añade la columna 'cluster', dibuja el gráfico con un recuadro
    mostrando el silhouette score, y devuelve el dataframe y las
    features seleccionadas del cluster superior.
    """

    # Extraer el score como matriz 1D
    X = df_final[["score"]].values

    # Ajustar KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=rng_seed)
    labels = kmeans.fit_predict(X)
    df_final["cluster"] = labels

    # Calcular silhouette
    silhouette = silhouette_score(X, labels)

    # Reordenar clusters según la media del score
    cluster_order = df_final.groupby("cluster")["score"].mean().sort_values().index
    mapping = {old: new for new, old in enumerate(cluster_order)}
    df_final["cluster"] = df_final["cluster"].map(mapping)

    # Dibujar gráfico
    plt.figure(figsize=(8, 6))
    plt.scatter(
        range(len(df_final)),
        df_final["score"],
        c=df_final["cluster"],
        cmap="viridis",
        s=50
    )
    plt.xlabel("Features (ordenadas)", fontsize=14)
    plt.ylabel("Score combinado (votes + importance)", fontsize=14)
    plt.title("Clusterización de scores (KMeans)", fontsize=16)

    # Añadir recuadro con silhouette
    plt.text(
        0.96, 0.96,
        f"Silhouette = {silhouette:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    if n_clusters == 2: 
        label = "Clúster (0=bajo, 1=alto)" 
    elif n_clusters == 3: 
        label = "Clúster (0=bajo, 1=medio, 2=alto)" 
    else: 
        label = f"Clúster (0=bajo, ..., {n_clusters-1}=alto)"
    plt.colorbar(label=label)
    plt.tight_layout()

    # Guardar
    plt.savefig(save_path, format="pdf", bbox_inches="tight")

    # Mostrar
    plt.show()

    # Seleccionar las features del cluster más alto
    selected_features = df_final[df_final["cluster"] == (n_clusters - 1)]["feature"].tolist()

    print(f"Silhouette score: {silhouette:.4f}")
    print(f"{len(selected_features)} features seleccionadas:", selected_features)

    return df_final, selected_features, silhouette


