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

def optimize_hyperparameters(original_estimator, param_grid, X_train, y_train, folds, rng_seed,
                             n_trials=30, study_name="", store_study=False):
    """
    Realiza una de búsqueda de hiperparámetros basada en un grid de rangos a explorar
    
    NOTA: Usa siempre: F1-score y selecciona el mejor umbral que le permita maximizarlo

    Parámetros
    ----------
    original_estimator : sklearn estimator
        Modelo base (LogisticRegression, RandomForestClassifier, XGBClassifier, CatBoost)
    param_grid : dict
        Diccionario con los hiperparámetros a explorar.
    X_train : array-like
        Matriz de entrenamiento.
    y_train : array-like
        Etiquetas de entrenamiento.
    folds: array-like
        Folds para la validación cruzada (se pasan para garantizar comparabilidad entre modelos)
    n_trials: int
        Número de trials para optimizar hiperparámetros
    study_name: str
        Nombre del estudio de Optuna
    store_study: bool
        Indica si se desea persistir el estudio en una BD SQLite (./data/optuna_studies/studies.db)

    Return
    ------
    study_results : dict
        Diccionario con: Objeto de estudio, mejores parámetros, mejor puntuación de métrica, 
        mejor umbral de f1, intervalo de confianza 95% y modelo entrenado con los mejores parámetros hallados
    """
    def objective(trial):
        # Construir kwargs del estimador a partir de la rejilla de parámetros
        # Basándose en el primer elemento de la rejilla para cada parámetro:
        #    "I": Espacio de enteros
        #    "F": Espacio de floats
        #    En cualquier otro caso: Espacio de categorías
        params = {}
        for name, space in param_grid.items():
            match space[0]:
                case "I":
                    params[name] = trial.suggest_int(name, min(space[1:-1]), max(space[1:-1]), **space[-1])
                case "F":
                    params[name] = trial.suggest_float(name, min(space[1:-1]), max(space[1:-1]), **space[-1])
                case _:
                    params[name] = trial.suggest_categorical(name, space[1:-1], **space[-1])

        # Clonar el estimador origen y actualizarlo con los hiperparámetros en cada iteración
        model = clone(original_estimator)
        model.set_params(**params, random_state=rng_seed)

        # Recorrer todos los folds de validación cruzada
        #
        # NOTA: Se tratan por separado los conjuntos de train y val de cada fold
        # para poder aprovecharse de la evaluación con early_stopping que 
        # XGBoost y CatBoost permiten definir para ahorrar creaciones de árboles 
        # innecesarias si no hay mejoras
        scores = []
        for fold_idx, (train_idx, valid_idx) in enumerate(folds):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train[train_idx], y_train[valid_idx]

            # Ajuste con soporte de eval_set y early stopping
            # Si los estimadores lo permiten, se usa early stopping
            #
            # NOTA: XGBClassifier da problemas si se le pasa el parámetro
            #       early_stopping_rounds en el fit, así que hay que 
            #       tratarlo a parte
            if isinstance(model, XGBClassifier):
                # XGBoost
                model.set_params(early_stopping_rounds=50)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            elif isinstance(model, CatBoostClassifier):
                # Catboost
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            else:
                # Modelos que no soportan early stopping
                model.fit(X_tr, y_tr)

            # Evaluar y devolver el mejor valor de F1 según umbral óptimo en cada caso
            # Se obtienen las "probabilidades" del modelo sobre el set de validación
            y_proba = model.predict_proba(X_val)[:, 1]
            # Se calcula el F1 del modelo para todos los umbrales (de 0 a 100)
            thresholds = np.linspace(0, 1, 101)
            f1_scores = [f1_score(y_val, (y_proba >= t).astype(int)) for t in thresholds]
            best_f1 = max(f1_scores)
            # Se acumula el mejor F1 usando el umbral óptimo sobre el modelo
            scores.append(best_f1)

            # Prunning: k-folds con mal resultado se cortan antes de tiempo
            # Reportar progreso a Optuna
            trial.report(best_f1, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Intervalo
        trial.set_user_attr("min_score", np.min(scores))
        trial.set_user_attr("max_score", np.max(scores))

        # Devolver F1-Score medio que Optuna trata de optimizar
        return np.mean(scores)

    # Crear estudio y optimizarlo
    if store_study:
        storage_path = "sqlite:///./data/optuna_studies/studies.db"
        study = optuna.create_study(
            study_name=study_name, direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
            storage=storage_path
        )
    else:
        study = optuna.create_study(
            study_name=study_name, direction="maximize",
            pruner=optuna.pruners.MedianPruner()
        )

    study.optimize(objective, n_trials=n_trials)

    # Organizar resultados a devolver
    study_results = {
        "study": study,
        "best_params": study.best_params,
        "best_score": study.best_value,
        "best_score_interval": (
            study.best_trial.user_attrs["min_score"],
            study.best_trial.user_attrs["max_score"]
        )
    }
    return study_results