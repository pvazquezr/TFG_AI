from IPython.display import display, Markdown
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import shap
import dice_ml
from dice_ml import Dice
from collections import Counter
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import f1_score, average_precision_score
from sklearn.inspection import permutation_importance


# Función auxiliar para formatear el dataframe de counterfactuals
# NOTA: Esto no haría falta si el método: visualize_as_dataframe(show_only_changes=True) funcionase correctamente
def format_cf_with_placeholders(dice_exp, features_to_vary=[], decimals=2, placeholder='-', outcome_col="Response"):
    idx=0
    # Original (fila única) y CFs
    orig = dice_exp.cf_examples_list[idx].test_instance_df.iloc[0]
    cf_df = dice_exp.cf_examples_list[idx].final_cfs_df.copy()

    # Mismas orden de columnas que el original
    orig_df = orig.to_frame().T
    cf_df = cf_df[orig_df.columns]

    # Alinear columnas y orden al original
    cf_df = cf_df.reindex(columns=orig.index)

    # Separar columnas numéricas vs. otras
    num_cols = cf_df.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in cf_df.columns if c not in num_cols]

    # Crear máscara de "sin cambio" por fila vs original
    mask_equal = pd.DataFrame(False, index=cf_df.index, columns=cf_df.columns)

    # Numéricos con tolerancia
    if num_cols:
        a = cf_df[num_cols].to_numpy(dtype=float)
        b = np.tile(orig[num_cols].to_numpy(dtype=float), (len(cf_df), 1))
        cmp_num = np.isclose(a, b, atol=1e-6, rtol=0)
        nan_equal = np.isnan(a) & np.isnan(b)
        mask_equal[num_cols] = cmp_num | nan_equal

    # No numéricos (exactos + NaNs)
    if other_cols:
        eq = cf_df[other_cols].eq(orig[other_cols])
        nan_eq = cf_df[other_cols].isna() & pd.isna(orig[other_cols])
        mask_equal[other_cols] = eq | nan_eq

    # Construir vista en object y colocar placeholder donde NO hay cambio
    cf_view = cf_df.astype(object)
    cf_view = cf_view.mask(mask_equal, other=placeholder)

    # Función auxiliar para redondeos en dataframes con guiones
    def round_mixed_dataframe(df, decimals=2):
        df_copy = pd.DataFrame(index=df.index, columns=df.columns)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                if isinstance(val, (int, float, np.number)):
                    # Redondear decimales
                    df_copy.iat[i, j] = round(val, 2)
                else:
                    df_copy.iat[i, j] = val
        return df_copy

    # Construir vista con solo las columnas necesarias y redondeos necesarios
    cols_to_show = features_to_vary + [outcome_col]
    orig_view_final_format = orig_df[cols_to_show].round(decimals)
    cf_view_final_format = round_mixed_dataframe(cf_view, decimals=2)[cols_to_show]

    # Muestra los dataframes
    return orig_view_final_format, cf_view_final_format


# Función auxiliar para calcular DiCE y visualizar el cambio concreto
def calculate_dice(X, y, features_to_vary, poi_patient_data, gene_columns, model, rng_seed):
    # 1. Preparar datos y modelo
    X["Response"] = y  # DiCE necesita la respuesta
    data = dice_ml.Data(dataframe=X, continuous_features=gene_columns.to_list(), outcome_name="Response")
    model_dice = dice_ml.Model(
        model=model, model_type="classifier", backend="sklearn")
    
    # 2. Crear objeto DiCE
    exp = Dice(data, model_dice, method="random")
    
    # 3. Seleccionar paciente concreto
    idx, sample_id, patient_type, patient_prob = poi_patient_data
    x_patient = X.iloc[idx:idx+1]
    
    # x_patient_dice no puede contener la columna Response sólo las de genes
    x_patient_dice = x_patient[gene_columns]
    
    # 4. Generar contrafactuales
    dice_exp = exp.generate_counterfactuals(
        x_patient_dice, total_CFs=5, features_to_vary=features_to_vary, 
        desired_class="opposite", random_seed=rng_seed)

    # 5. Mostrar la explicación
    #dice_exp.visualize_as_dataframe(show_only_changes=True)
    
    orig_view, cf_view = format_cf_with_placeholders(dice_exp, features_to_vary=features_to_vary)
    print()
    display(Markdown(f"**Original** | Paciente: **{patient_type}** ({sample_id})"))
    display(orig_view)
    display(Markdown(f"**Contrafactuales** | Paciente: **{patient_type}** ({sample_id})"))
    display(cf_view)


# Función auxiliar para calcular DiCE y mostrar histograma
def calculate_dice_bootsrap(X, y, features_to_vary, poi_patient_data, gene_columns, model, rng_seed, save_path=''):
    # 1. Preparar datos y modelo
    X["Response"] = y  # DiCE necesita la respuesta
    data = dice_ml.Data(dataframe=X, continuous_features=gene_columns.to_list(), outcome_name="Response")
    model_dice = dice_ml.Model(
        model=model, model_type="classifier", backend="sklearn")
    
    # 2. Crear objeto DiCE
    exp = Dice(data, model_dice, method="random")
    
    # 3. Seleccionar paciente concreto
    idx, sample_id, patient_type, patient_prob = poi_patient_data
    x_patient = X.iloc[idx:idx+1]
    
    # x_patient_dice no puede contener la columna Response sólo las de genes
    x_patient_dice = x_patient[gene_columns]
    
    # 4. Generar contrafactuales en bootstrap (varias remuestras)
    n_bootstrap = 50
    cf_results = []
    
    for i in range(n_bootstrap):
        # Generar contrafactuales diversos
        dice_exp = exp.generate_counterfactuals(
            x_patient_dice, total_CFs=5, features_to_vary=features_to_vary, 
            desired_class="opposite", random_seed=i)
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        cf_results.append(cf_df)
    
    # 5. Concatenar todos los contrafactuales
    cf_all = pd.concat(cf_results, ignore_index=True)
    
    # 6. Contar frecuencia de aparición de genes "palanca"
    # Palanca = variable que cambia respecto al paciente original
    changed_features = []
    for cf_df in cf_results:
        # Excluir columna Response
        feature_cols = [c for c in cf_df.columns if c != "Response"]
        for _, row in cf_df.iterrows():
            base_row = x_patient.iloc[0][feature_cols]
            diff = (row[feature_cols] != base_row)
            changed_features.extend(diff.index[diff].tolist())
    
    freq_counter = Counter(changed_features)
    
    # 7. Convertir a DataFrame para visualizar
    freq_df = pd.DataFrame(freq_counter.items(), columns=["Gen", "Frequency"]).sort_values("Frequency", ascending=False)

    display(Markdown(f"Paciente: {patient_type} ({sample_id})"))
    display(freq_df)
    
    # 8. Gráfico de barras de frecuencia
    plt.figure(figsize=(8,6))
    plt.bar(freq_df["Gen"], freq_df["Frequency"], color="steelblue")
    plt.xlabel("Gen", fontsize=16)
    plt.ylabel("Frecuencia de aparición como palanca", fontsize=16)
    plt.title(f"Genes palanca en contrafactuales\nPaciente: {patient_type} ({sample_id})", fontsize=18)
        
    # Guardar primero
    plt.savefig(save_path, format="pdf", bbox_inches="tight")

    # Mostrar después
    plt.show()

