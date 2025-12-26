import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from venn_abers import VennAbers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_curve
from .metrics import best_f1_threshold, compute_ece


def calibrate_models(model, X_cal, y_cal):
    """
    Devuelve tres calibradores:
    - Venn–ABERS
    - Platt Scaling
    - Isotonic Regression
    """

    # --- 1) Venn–ABERS ---
    # VennAbers espera una matriz (n_samples, 2)
    p_cal = model.predict_proba(X_cal)

    va = VennAbers()
    va.fit(p_cal, y_cal)

    # --- 2) Platt Scaling ---
    platt = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method="sigmoid"
    )
    platt.fit(X_cal, y_cal)

    # --- 3) Isotonic Regression ---
    isotonic = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method="isotonic"
    )
    isotonic.fit(X_cal, y_cal)

    return va, platt, isotonic

def evaluate_calibrated_models(base_model, calibrated_models, X_test, y_test):
    """
    Se le pasa una lista de calibradores y devuelve métricas de relevancia para el modelo base y los modelos calibrados
    """
    
    results = {}

    # --- Probabilidades del modelo original ---
    p_base = base_model.predict_proba(X_test)
    p_base = p_base[:, 1]  # FORZAR 1D

    thr_base, f1_base = best_f1_threshold(y_test, p_base)

    results["base"] = {
        "brier": brier_score_loss(y_test, p_base),
        "logloss": log_loss(y_test, p_base),
        "ece": compute_ece(y_test, p_base),
        "f1": f1_base,
        "threshold": thr_base,
        "probs_info": {"p_base": p_base}
    }

    # --- Calibradores ---
    for nombre, cal in calibrated_models.items():
        probs_info = {}
        if nombre == "va":
            p_test = base_model.predict_proba(X_test)
            # NOTA: predict_proba de Venn-ABERS devuelve (p_prime, p0_p1)
            #
            # p_prime (n,2) : Probabilidades calibradas finales 
            #                 (probabilidad_clase_0, probabilidad_clase_1) ---> Para cálculo de métricas
            #                 No es el punto medio tal cual como se ve en la fórmula:
            #                     p_prime[:, 1] = p0_p1[:, 1] / (1 - p0_p1[:, 0] + p0_p1[:, 1])
            #                     p_prime[:, 0] = 1 - p_prime[:, 1]
            # p0_p1 (n,2) : Intervalo [p_low, p_high] ----> Para medición de incertidumbre
            #
            # Es decir, que para los cálculos de F1, Brier, LogLoss se puede obtener la probabilidad 
            # de la clase positiva así: p_prime[:, 1]
            p_prime, p0_p1 = cal.predict_proba(base_model.predict_proba(X_test))
            
            # Probabilidad calibrada para Venn-ABERS
            p_cal = p_prime[:, 1] # Probabilidad de la clase 1 (positiva)

            # Se guarda toda la información de la predicción
            probs_info = {
                "p_prime": p_prime,
                "p0_p1": p0_p1,
                "p_cal": p_cal
            }
        else:
            # Probabilidad calibrada para Platt e Isotonic
            p_cal = cal.predict_proba(X_test)[:, 1]
            probs_info = {
                "p_cal": p_cal
            }

        thr, f1_val = best_f1_threshold(y_test, p_cal)

        results[nombre] = {
            "brier": brier_score_loss(y_test, p_cal),
            "logloss": log_loss(y_test, p_cal),
            "ece": compute_ece(y_test, p_cal),
            "f1": f1_val,
            "threshold": thr,
            "intervals_info": probs_info
        }

    return results


def venn_abers_intervals(model, X_test, y_test, X_cal, y_cal,
        decision_threshold=0.5,
        confidence_level=0.05,
        min_interval_width_filter=0.0,
        figsize=(12, 6),
        point_size=30,
        title=None,
        save_path=""
    ):
    """
    Genera un gráfico de intervalos conformales (p0, p1) + probabilidad unida p,
    comparando probabilidades calibradas y no calibradas.

    Parámetros:
    -----------
    model : modelo sklearn con predict_proba
    X_cal, y_cal : datos de calibración
    X_test, y_test : datos de test
    ScoresToMultiProbs : función que devuelve (p0, p1)
    decision_threshold : umbral para dibujar línea horizontal
    figsize : tamaño del gráfico
    point_size : tamaño base de los puntos
    title : título opcional
    """

    # --- Probabilidades del modelo base ---
    y_pred_test = model.predict_proba(X_test)
    y_pred_cal = model.predict_proba(X_cal)

    # --- 1) Venn–ABERS ---
    # VennAbers espera una matriz (n_samples, 2)
    va = VennAbers()
    va.fit(y_pred_cal, y_cal)
    
    # --- Obtener intervalos conformales ---
    p, p0_p1 = va.predict_proba(y_pred_test)
    p0 = p0_p1[:, 0]
    p1 = p0_p1[:, 1]

    # --- Construir DataFrame de resultados ---
    predictions = pd.DataFrame({
        "y_true": y_test,
        "raw_scores": y_pred_test[:, 1],
        "p0": p0,
        "p1": p1,
        "p": p[:, 1],
    })

    predictions["width"] = predictions["p1"] - predictions["p0"]
    predictions = predictions.sort_values(by="p").reset_index(drop=True)

    predictions_to_draw = predictions[predictions["width"] > min_interval_width_filter]

    # --- Gráfico ---
    with plt.style.context("fivethirtyeight"):
        fig, ax = plt.subplots(figsize=figsize)
    
        s = point_size
    
        plt.scatter(predictions_to_draw.index, predictions_to_draw["raw_scores"],
                    label="sin calibrar", marker='o', s=s+70, color='c')
    
        plt.scatter(predictions_to_draw.index, predictions_to_draw["p1"],
                    label="probabilidad superior (p1)", s=s+50)
    
        plt.scatter(predictions_to_draw.index, predictions_to_draw["p0"],
                    label="probabilidad inferior (p0)", s=s+50)
    
        plt.scatter(predictions_to_draw.index, predictions_to_draw["p"],
                    label="probabilidad conforme", s=s+10, marker='o')
    
        plt.scatter(predictions_to_draw.index, predictions_to_draw["y_true"],
                    label="predicción real\n(0=no respondedor, 1=respondedor)", s=s, marker='o', color="k")
    
        plt.plot(predictions_to_draw.index, predictions_to_draw["width"],
                 label="ancho del intervalo (p1-p0)", lw=3.5, color="pink")
    
        # --- Líneas horizontales ---
        # Sólo pintar el umbral de decisión si no está a cero
        if decision_threshold:
            plt.hlines(y=decision_threshold, xmin=0, xmax=len(predictions),
                       lw=0.5, color='green')
            plt.text(1, decision_threshold + 0.05,
                     f'y={decision_threshold}', ha='right', va='center', color='green')
    
        # Ejemplo de línea de confianza (p.ej. 0.05)
        plt.hlines(y=confidence_level, xmin=0, xmax=len(predictions),
                   lw=0.5, color='red')
        plt.text(1, 0.10, f'y={confidence_level}', ha='right', va='center', color='red')
    
        plt.legend(fontsize=14, scatterpoints=1, markerscale=1, loc="upper left", bbox_to_anchor=(0.04, 0.96))
        plt.ylabel("probabilidad", fontsize=20)
        plt.xlabel("orden de menor a mayor probabilidad conforme", fontsize=20)
        plt.xticks([])
        plt.rc('ytick', labelsize=15)
        plt.rc('xtick', labelsize=15)
    
        if title:
            plt.title(title, fontsize=22)
    
        # Guardar si se pasa el path
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
    
        # Mostrar
        plt.show()

    return predictions


def plot_calibration_curve(ax, y_true, p_base, p_platt, p_isotonic, p_va, title, strategy="quantile"):
    # Línea ideal
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")

    # Modelo base
    prob_true, prob_pred = calibration_curve(y_true, p_base, n_bins=10, strategy=strategy)
    ax.plot(prob_pred, prob_true, marker="o", label="Modelo base")

    # Platt
    prob_true, prob_pred = calibration_curve(y_true, p_platt, n_bins=10, strategy=strategy)
    ax.plot(prob_pred, prob_true, marker="o", label="Platt")

    # Isotonic
    prob_true, prob_pred = calibration_curve(y_true, p_isotonic, n_bins=10, strategy=strategy)
    ax.plot(prob_pred, prob_true, marker="o", label="Isotonic")

    # Venn–ABERS
    prob_true, prob_pred = calibration_curve(y_true, p_va, n_bins=10, strategy=strategy)
    ax.plot(prob_pred, prob_true, marker="o", label="Venn–ABERS")

    ax.set_title(title)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")


def build_calibration_summary(metrics_by_model, decimals=3):
    """
    metrics_by_model debe tener esta estructura:
    {
        "LogReg": logreg_calibration_metrics,
        "RandomForest": rf_calibration_metrics,
        "XGBoost": xgb_calibration_metrics,
        "CatBoost": catboost_calibration_metrics
    }
    """

    # Orden deseado
    model_order = ["LogReg", "RandomForest", "XGBoost", "CatBoost"]
    calibrator_order = ["base", "platt", "isotonic", "va"]

    rows = []

    for model_name, calib_metrics in metrics_by_model.items():
        for calibrator_name, metrics in calib_metrics.items():
            rows.append({
                "Modelo": model_name,
                "Calibrador": calibrator_name,
                "Brier": metrics["brier"],
                "LogLoss": metrics["logloss"],
                "ECE": metrics["ece"],
                "F1": metrics["f1"],
                "Umbral F1": metrics["threshold"]
            })

    df = pd.DataFrame(rows)

    # Aplicar orden categórico
    df["Modelo"] = pd.Categorical(df["Modelo"], categories=model_order, ordered=True)
    df["Calibrador"] = pd.Categorical(df["Calibrador"], categories=calibrator_order, ordered=True)

    # Ordenar
    df = df.sort_values(["Modelo", "Calibrador"])

    # Redondeo configurable
    numeric_cols = ["Brier", "LogLoss", "ECE", "F1", "Umbral F1"]
    df[numeric_cols] = df[numeric_cols].round(decimals)

    return df



# Wrapper para VennAbers
# 
# NOTA: Se creó este wrapper ya que VennAbersCalibrator del propio paquete
#       no admite set de calibración aparte, sólo se le puede especificar 
#       la proporción del set de train que usará como set de calibración
#       pero eso no me permitía hacer comparaciones justas con los demás 
#       calibradores
class VennAbersClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, va_model, threshold=0.5):
        self.base_model = base_model
        self.va_model = va_model
        self.threshold = threshold        

    def fit(self, X, y):
        """
        Entrena el modelo base y luego ajusta Venn–Abers
        usando los scores del modelo base.
        """
        # 1. Entrenar modelo base
        self.base_model.fit(X, y)

        # 2. Obtener scores del modelo base (n_samples x 2)
        base_scores = self.base_model.predict_proba(X)

        # 3. Ajustar Venn–Abers
        self.va_model.fit(base_scores, y)

        return self

    def predict_proba(self, X):
        """
        Devuelve probabilidades calibradas con Venn–Abers.
        """
        # Scores del modelo base (n_samples x 2)
        base_scores = self.base_model.predict_proba(X)
        
        # Salida original de VA
        p_prime, p0_p1 = self.va_model.predict_proba(base_scores) 
        
        return p_prime, p0_p1


    def predict(self, X):
        """
        Predicción binaria usando umbral sobre la clase positiva (1).
        Compatible con sklearn en el sentido de que la firma es predict(X),
        y threshold es opcional para llamadas manuales.
        """
        p_prime, _ = self.predict_proba(X)          # (n, 2)
        p_pos = p_prime[:, 1]                   # prob clase positiva

        return (p_pos >= self.threshold).astype(int)

