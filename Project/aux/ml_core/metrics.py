import numpy as np
import pandas as pd
from dcurves import dca, plot_graphs
from sklearn.metrics import f1_score, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt


def best_f1_threshold(y_true, y_score):
    # Asegurar que y_true es 1D
    y_true = np.array(y_true).ravel()

    # Convertir y_score a 1D si viene como matriz
    y_score = np.array(y_score)
    if y_score.ndim > 1:
        y_score = y_score[:, -1]  # clase positiva

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)

    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1

def compute_ece(y_true, y_proba, n_bins=10):
    """
    Implementación de ECE a partir de la definición que viene en el paper:
        "Guo et al. (2017) — On Calibration of Modern Neural Networks" https://arxiv.org/abs/1706.04599
        ECE = Σ_b (n_b / N) * | acc(b) - conf(b) |
            - N     = número total de muestras
            - n_b   = número de muestras en el bin b
            - accuracy(b)  = proporción de muestras positivas en el bin b
            - confidence(b) = probabilidad media predicha en el bin b
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Bin boundaries
        start, end = bins[i], bins[i+1]

        # Samples in this bin
        idx = (y_proba >= start) & (y_proba < end)
        if np.sum(idx) == 0:
            continue

        # Mean predicted probability
        conf = np.mean(y_proba[idx])
        
        # Empirical accuracy
        acc = np.mean(y_true[idx])

        # Weighted absolute difference
        ece += np.abs(acc - conf) * np.sum(idx) / len(y_true)

    return ece


def decision_curve_analysis(base_model, va_model, platt_model, isotonic_model, X_test, y_test, X_cal, y_cal, save_path=None, **kwargs):    
    # Probabilidades
    proba_base     = base_model.predict_proba(X_test)[:, 1]
    proba_platt    = platt_model.predict_proba(X_test)[:, 1]
    proba_isotonic = isotonic_model.predict_proba(X_test)[:, 1]
    p_prime, _     = va_model.predict_proba(base_model.predict_proba(X_test))
    proba_va       = p_prime[:, 1]
    
    # DataFrame para DCA
    df = pd.DataFrame({
        "y": y_test,
        "base": proba_base,
        "platt": proba_platt,
        "isotonic": proba_isotonic,
        "va": proba_va
    })
    
    # Decision Curve Analysis
    df_dca = dca(
        data=df,
        outcome="y",
        modelnames=["base", "platt", "isotonic", "va"],
        thresholds=np.arange(0.01, 0.99, 0.01),  # 1% a 99% en escala 0–1
    )
    
    # Plot estándar de net benefit
    plot_graphs(
        plot_df=df_dca,
        graph_type="net_benefit",
        file_name=save_path,
        linewidths=[1.5, 1.5, 1.5, 1.5, 2.0, 2.0], 
        color_names=['limegreen', 'deepskyblue', 'darkorange', 'rebeccapurple', 'red', 'blue'],
        **kwargs
    )

def get_importances_logreg(model, feature_names):
    coefs = np.abs(model.coef_[0])
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": coefs
    }).sort_values("importance", ascending=False)
    return df

def get_importances_rf(model, feature_names):
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    return df

def get_importances_xgb(model):
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")
    df = pd.DataFrame(scores.items(), columns=["feature", "importance"])
    df = df.sort_values("importance", ascending=False)
    return df

def get_importances_catboost(model, feature_names):
    importances = model.get_feature_importance()
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    return df

def plot_importances(df, title):
    plt.figure(figsize=(6, 6))
    sns.barplot(data=df, x="importance", y="feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()
