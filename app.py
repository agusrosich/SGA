"""
Análisis de Supervivencia en Cáncer de Glándula Salival
Modelos: Random Forest, GBM, Redes Neuronales
Análisis especial: Quimio y Radioterapia en estadios T3/T4
VERSIÓN CORREGIDA: Grupos de riesgo basados en características basales
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from sklearn.utils import resample
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    import missingno as msno
except ImportError:
    print("⚠️ ADVERTENCIA: instalar 'missingno' para visualizar patrones de datos faltantes (pip install missingno)")
    msno = None

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None
    print("⚠️ ADVERTENCIA: instalar 'statsmodels' para aplicar corrección FDR en subgrupos (pip install statsmodels)")

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================================

print("="*60)
print("ANÁLISIS DE SUPERVIVENCIA - CÁNCER DE GLÁNDULA SALIVAL")
print("="*60)

# Cargar datos
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'ExportadaSEER_Estandarizada.csv'
df = pd.read_csv(DATA_PATH)
print(f"\n✓ Datos cargados: {len(df)} pacientes")

# Crear variable de evento (muerte por cáncer)
df['Event'] = (df['COD to site recode'] != 'Alive').astype(int)
df['Survival_years'] = df['Survival months'] / 12
MAX_FOLLOW_UP_MONTHS = 120
df['Survival_months_capped'] = df['Survival months'].clip(upper=MAX_FOLLOW_UP_MONTHS)
df['Event_capped'] = np.where(df['Survival months'] > MAX_FOLLOW_UP_MONTHS, 0, df['Event'])

# Crear estadios TNM agrupados
def assign_stage(row):
    t_raw = str(row['T_Unified']).upper()
    n_raw = str(row['N_Unified']).upper()

    t = t_raw if isinstance(t_raw, str) else ''
    n = n_raw if isinstance(n_raw, str) else ''

    if any(prefix in t for prefix in ['TX', 'T88', 'T0']) or any(prefix in n for prefix in ['NX', 'N88']):
        return 'Unknown'

    if t.startswith('T1') and n.startswith('N0'):
        return 'Stage I'
    if (t.startswith('T2') and n.startswith('N0')) or (t.startswith('T1') and n.startswith('N1')):
        return 'Stage II'
    if (t.startswith('T3') and n.startswith('N0')) or (t.startswith(('T1', 'T2')) and n.startswith(('N1', 'N2'))):
        return 'Stage III'
    if t.startswith('T4') or n.startswith(('N2', 'N3')) or (t.startswith('T3') and n.startswith(('N2', 'N3'))):
        return 'Stage IV'

    return 'Stage III'

df['Stage'] = df.apply(assign_stage, axis=1)
df_unknown_stage = df[df['Stage'] == 'Unknown'].copy()
df = df[df['Stage'] != 'Unknown'].copy()
if len(df_unknown_stage) > 0:
    print(f"V Pacientes excluidos por estadio desconocido: {len(df_unknown_stage)}")

# Grupos de T avanzados
df['T_Advanced'] = df['T_Unified'].apply(lambda x: 'T3-T4' if str(x).startswith(('T3', 'T4')) else 'T1-T2' if str(x).startswith(('T1', 'T2')) else 'Unknown')

# Grupos de tratamiento
df['Treatment_Group'] = df.apply(lambda x: 
    'Chemo+Radio' if x['Chemotherapy_Binary']==1 and x['Radiation_Binary']==1 
    else 'Radio Only' if x['Radiation_Binary']==1 
    else 'Chemo Only' if x['Chemotherapy_Binary']==1 
    else 'No Treatment', axis=1)

# Indicador binario para combinación de quimio + radioterapia
df['Combined_Treatment'] = ((df['Chemotherapy_Binary'] == 1) & (df['Radiation_Binary'] == 1)).astype(int)

def _prepare_covariates_for_balance(df_in, covariates):
    """One-hot encode covariates to compute balance metrics."""
    cov_df = pd.get_dummies(df_in[covariates].copy(), drop_first=False)
    cov_df = cov_df.astype(float)
    return cov_df

def compute_smd(df_in, treatment_col, covariates):
    """Compute standardized mean differences for covariates."""
    cov_df = _prepare_covariates_for_balance(df_in, covariates)
    treated_mask = df_in[treatment_col] == 1
    control_mask = df_in[treatment_col] == 0
    smd_results = []

    for col in cov_df.columns:
        treated_vals = cov_df.loc[treated_mask, col]
        control_vals = cov_df.loc[control_mask, col]

        if treated_vals.empty or control_vals.empty:
            smd = np.nan
        else:
            var_t = treated_vals.var(ddof=1)
            var_c = control_vals.var(ddof=1)
            pooled_sd = np.sqrt((var_t + var_c) / 2)
            if pooled_sd == 0 or np.isnan(pooled_sd):
                smd = np.nan
            else:
                smd = (treated_vals.mean() - control_vals.mean()) / pooled_sd

        smd_results.append({'Covariate': col, 'SMD': smd})

    return pd.DataFrame(smd_results)

def plot_love_plot(balance_before, balance_after, title):
    """Generate love plot comparing balance before and after matching."""
    merged = balance_before.merge(balance_after, on='Covariate', suffixes=('_before', '_after'))
    merged = merged.sort_values('SMD_before', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(merged) * 0.2)))
    ax.scatter(merged['SMD_before'].abs(), merged.index, color='tab:red', label='Antes')
    ax.scatter(merged['SMD_after'].abs(), merged.index, color='tab:blue', label='Despues')
    ax.axvline(0.1, color='gray', linestyle='--', linewidth=1, label='Limite 0.1')
    ax.set_xlabel('|SMD|')
    ax.set_yticks(merged.index)
    ax.set_yticklabels(merged['Covariate'])
    ax.set_title(title)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig

def propensity_score_match(df_in, treatment_col, covariates):
    """Run logistic propensity model and nearest neighbor matching (1:1, sin reemplazo)."""
    working_df = df_in.copy()
    cov_matrix = _prepare_covariates_for_balance(working_df, covariates)
    ps_model = LogisticRegression(max_iter=2000, solver='lbfgs')
    ps_model.fit(cov_matrix, working_df[treatment_col])
    working_df['propensity_score'] = ps_model.predict_proba(cov_matrix)[:, 1]

    treated = working_df[working_df[treatment_col] == 1].copy()
    control = working_df[working_df[treatment_col] == 0].copy()

    if treated.empty or control.empty:
        return working_df, working_df, None

    control_pool = control.copy()
    matched_indices_treated = []
    matched_indices_control = []

    for t_idx, t_row in treated.iterrows():
        if control_pool.empty:
            break
        closest_idx = (control_pool['propensity_score'] - t_row['propensity_score']).abs().idxmin()
        matched_indices_treated.append(t_idx)
        matched_indices_control.append(closest_idx)
        control_pool = control_pool.drop(index=closest_idx)

    matched_treated = treated.loc[matched_indices_treated]
    matched_control = control.loc[matched_indices_control]
    matched_df = pd.concat([matched_treated, matched_control], axis=0).sort_index()
    matched_df['match_weight'] = 1
    return working_df, matched_df, ps_model

print(f"✓ Variables creadas: Stage, T_Advanced, Treatment_Group")

# ============================================================
# 2. ANÁLISIS DESCRIPTIVO
# ============================================================

print("\n" + "="*60)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*60)

desc_stats = pd.DataFrame({
    'Total Pacientes': [len(df)],
    'Edad Media (años)': [df['Age_Median'].mean()],
    'Supervivencia Media (meses)': [df['Survival_months_capped'].mean()],
    'Eventos (muertes)': [df['Event'].sum()],
    'Tasa Mortalidad (%)': [df['Event'].mean()*100]
})
print(desc_stats.to_string(index=False))

print("\n--- Distribución por Estadio TNM ---")
print(df['Stage'].value_counts())

print("\n--- Distribución por T ---")
print(df['T_Unified'].value_counts())

print("\n--- Distribución de Tratamientos ---")
print(df['Treatment_Group'].value_counts())

# ============================================================
# 3. ANÁLISIS ESPECÍFICO T3/T4: QUIMIO Y RADIOTERAPIA
# ============================================================

print("\n" + "="*60)
print("ANÁLISIS T3/T4: EFECTO DE QUIMIO Y RADIOTERAPIA")
print("="*60)

# Filtrar pacientes T3/T4
df_advanced = df[df['T_Advanced'] == 'T3-T4'].copy()
print(f"\nPacientes T3/T4: {len(df_advanced)}")

# Análisis por tratamiento en T3/T4
print("\n--- Supervivencia Media por Tratamiento (T3/T4) ---")
surv_by_treat = df_advanced.groupby('Treatment_Group').agg({
    'Survival_months_capped': ['mean', 'median', 'count'],
    'Event_capped': 'mean'
}).round(2)
surv_by_treat.columns = ['Media (meses)', 'Mediana (meses)', 'N', 'Tasa Mortalidad']
print(surv_by_treat)

print("\n--- Propensity Score Matching: Chemo+Radio vs otros ---")
psm_covariates = ['Age_Median', 'Sex', 'N_Unified', 'Histology_Unified', 'Stage']
psm_ready = df_advanced.dropna(subset=psm_covariates + ['Survival_months_capped', 'Event_capped']).copy()

psm_all, df_advanced_matched, ps_model = propensity_score_match(
    psm_ready,
    treatment_col='Combined_Treatment',
    covariates=psm_covariates
)

love_plot_fig = None
psm_balance_table = pd.DataFrame()
df_t34_matched = df_advanced_matched.copy()

if df_advanced_matched.empty:
    print("No se logró emparejamiento adecuado; revisar los datos disponibles.")
else:
    balance_before = compute_smd(psm_all, 'Combined_Treatment', psm_covariates)
    balance_after = compute_smd(df_advanced_matched, 'Combined_Treatment', psm_covariates)
    balance_table = balance_before.merge(balance_after, on='Covariate', suffixes=('_Antes', '_Despues'))
    balance_table['Cumple_<0.1'] = balance_table['SMD_Despues'].abs() < 0.1

    print("\nBalance de covariables (|SMD|):")
    print(balance_table.to_string(index=False))

    love_plot_fig = plot_love_plot(balance_before, balance_after, 'Balance covariables PSM T3/T4')
    psm_balance_table = balance_table

    treated_matched = df_advanced_matched[df_advanced_matched['Combined_Treatment'] == 1]
    control_matched = df_advanced_matched[df_advanced_matched['Combined_Treatment'] == 0]

    if len(treated_matched) >= 5 and len(control_matched) >= 5:
        result_lr_psm = logrank_test(
            treated_matched['Survival_months_capped'],
            control_matched['Survival_months_capped'],
            treated_matched['Event_capped'],
            control_matched['Event_capped']
        )
        print(f"\nLog-rank en cohorte emparejada: p-value = {result_lr_psm.p_value:.4f}")
    else:
        print("\nAdvertencia: cohorte emparejada insuficiente para log-rank robusto.")

# ============================================================
# 4. PREPARACIÓN DE DATOS PARA MODELOS ML
# ============================================================

print("\n" + "="*60)
print("PREPARACIÓN DE DATOS PARA MODELOS ML")
print("="*60)

# Seleccionar características
feature_cols = ['Age_Median', 'Sex', 'T_Unified', 'N_Unified', 
                'Radiation_Binary', 'Chemotherapy_Binary', 'Histology_Unified']

df_ml = df[feature_cols + ['Event_capped', 'Survival_months_capped']].copy()
df_ml = df_ml.rename(columns={'Event_capped': 'Event'})

# Eliminar filas con valores desconocidos en T o N
df_ml = df_ml[~df_ml['T_Unified'].isin(['TX', 'T88'])].copy()
df_ml = df_ml[~df_ml['N_Unified'].isin(['NX', 'N88'])].copy()

missingness_fig = None
missing_report = df_ml[feature_cols + ['Event']].isnull().sum()
if missing_report.sum() > 0:
    missing_df = pd.DataFrame({
        'Variable': missing_report.index,
        'Missing': missing_report.values,
        '%': (missing_report.values / len(df_ml) * 100).round(2)
    })
    print("\nResumen de valores faltantes (antes de imputación):")
    print(missing_df[missing_df['Missing'] > 0].to_string(index=False))
    if msno is not None:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        msno.matrix(df_ml[feature_cols + ['Event']], ax=ax)
        ax.set_title('Patrón de datos faltantes (previo a imputación)')
        plt.tight_layout()
        missingness_fig = fig
    else:
        print("missingno no disponible; omitiendo visualización gráfica de missingness.")
else:
    print("No se detectaron valores faltantes en las variables seleccionadas.")

# Sensibilidad: guardar versión de casos completos
df_ml_complete_case = df_ml.dropna().copy()
print(f"✓ Casos completos disponibles: {len(df_ml_complete_case)}")

# Imputación múltiple para variables numéricas
categorical_cols = ['Sex', 'T_Unified', 'N_Unified', 'Histology_Unified']
numeric_cols = ['Age_Median', 'Radiation_Binary', 'Chemotherapy_Binary']

for col in categorical_cols:
    df_ml[col] = df_ml[col].fillna('Desconocido').astype(str)
    if not df_ml_complete_case.empty:
        df_ml_complete_case[col] = df_ml_complete_case[col].astype(str)

if df_ml[numeric_cols].isnull().any().any():
    imputer = IterativeImputer(random_state=42, max_iter=10)
    df_ml[numeric_cols] = imputer.fit_transform(df_ml[numeric_cols])
else:
    imputer = None

print(f"✓ Dataset ML tras imputación: {len(df_ml)} pacientes")

# Codificar variables categóricas
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    le_dict[col] = le
    if not df_ml_complete_case.empty:
        df_ml_complete_case[col] = le.transform(df_ml_complete_case[col].astype(str))

# Separar features y target
X = df_ml[feature_cols]
y = df_ml['Event']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Escalar features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# 5. MODELOS DE MACHINE LEARNING
# ============================================================

print("\n" + "="*60)
print("ENTRENAMIENTO DE MODELOS")
print("="*60)

models = {}
results = {}
cv_metrics = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
gbm_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)

model_configs = [
    ('Random Forest', rf_model, False, clone(rf_model)),
    ('GBM', gbm_model, False, clone(gbm_model)),
    ('Neural Network', nn_model, True, Pipeline([('scaler', StandardScaler()), ('model', clone(nn_model))]))
]

print("\n--- Rendimiento de Modelos ---")
for name, estimator, needs_scaling, cv_estimator in model_configs:
    print(f"\nEntrenando {name}...")
    if needs_scaling:
        estimator.fit(X_train_scaled, y_train)
        y_pred = estimator.predict(X_test_scaled)
        y_proba = estimator.predict_proba(X_test_scaled)[:, 1]
    else:
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        y_proba = estimator.predict_proba(X_test)[:, 1]
    
    models[name] = estimator
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    results[name] = {
        'predictions': y_pred,
        'probabilities': y_proba,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }
    
    print(f"  AUC (hold-out): {roc_auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Vivo', 'Fallecido']))
    
    cv_scores = cross_val_score(cv_estimator, X, y, cv=cv, scoring='roc_auc')
    cv_metrics[name] = cv_scores
    print(f"  AUC CV (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

if not df_ml_complete_case.empty:
    rf_cc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    cv_scores_cc = cross_val_score(rf_cc, df_ml_complete_case[feature_cols], df_ml_complete_case['Event'],
                                   cv=cv, scoring='roc_auc')
    print(f"\nSensibilidad (Random Forest - casos completos) AUC: {cv_scores_cc.mean():.3f} ± {cv_scores_cc.std():.3f}")

print("\n✓ Modelos entrenados y validados")

calibration_fig = None
if 'Random Forest' in results:
    prob_true, prob_pred = calibration_curve(y_test, results['Random Forest']['probabilities'], n_bins=10)
    calibration_brier = brier_score_loss(y_test, results['Random Forest']['probabilities'])
    calibration_fig = plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Random Forest')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfecto')
    plt.xlabel('Probabilidad pronosticada')
    plt.ylabel('Probabilidad observada')
    plt.title('Curva de calibración - Random Forest')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    print(f"Brier Score (Random Forest): {calibration_brier:.3f}")
    print("  Interpretación: menor es mejor (0 = perfecto, 0.25 = azar)")
    if calibration_brier < 0.10:
        print("  ✓ Excelente calibración")
    elif calibration_brier < 0.15:
        print("  ✓ Buena calibración")
    else:
        print("  ⚠ Considerar recalibración")
else:
    print("No se pudo generar curva de calibración (Random Forest no disponible).")

# Importancia de características (Random Forest)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

importance_boots = []
for idx in range(100):
    X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=42 + idx)
    rf_boot = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42 + idx, class_weight='balanced')
    rf_boot.fit(X_boot, y_boot)
    importance_boots.append(rf_boot.feature_importances_)

importance_boots = np.array(importance_boots)
feature_importance['CI_2.5%'] = np.percentile(importance_boots, 2.5, axis=0)
feature_importance['CI_97.5%'] = np.percentile(importance_boots, 97.5, axis=0)

print("\n--- Importancia de Variables (Random Forest) ---")
print(feature_importance.to_string(index=False))

# ============================================================
# 6. ODDS RATIOS Y ANÁLISIS ESTADÍSTICO
# ============================================================

print("\n" + "="*60)
print("COX UNIVARIANTE - FACTORES DE RIESGO")
print("="*60)

cox_inputs = []

df_cox_radiation = df[['Survival_months_capped', 'Event_capped', 'Radiation_Binary']].dropna().rename(columns={'Radiation_Binary': 'factor'})
cox_inputs.append(('Radioterapia (sí vs no)', df_cox_radiation))

df_cox_chemo = df[['Survival_months_capped', 'Event_capped', 'Chemotherapy_Binary']].dropna().rename(columns={'Chemotherapy_Binary': 'factor'})
cox_inputs.append(('Quimioterapia (sí vs no)', df_cox_chemo))

df_t_comparison = df[df['T_Advanced'].isin(['T1-T2', 'T3-T4'])].copy()
df_t_comparison['factor'] = (df_t_comparison['T_Advanced'] == 'T3-T4').astype(int)
cox_inputs.append(('T3-T4 vs T1-T2', df_t_comparison[['Survival_months_capped', 'Event_capped', 'factor']].dropna()))

df_n_comparison = df[df['N_Unified'].isin(['N0', 'N1', 'N2', 'N3'])].copy()
df_n_comparison['factor'] = (df_n_comparison['N_Unified'] != 'N0').astype(int)
cox_inputs.append(('N+ vs N0', df_n_comparison[['Survival_months_capped', 'Event_capped', 'factor']].dropna()))

df_stage = df.copy()
df_stage['factor'] = (df_stage['Stage'] == 'Stage IV').astype(int)
cox_inputs.append(('Stage IV vs I-III', df_stage[['Survival_months_capped', 'Event_capped', 'factor']].dropna()))

cox_results = []
for label, data_subset in cox_inputs:
    if data_subset['factor'].nunique() < 2:
        continue
    cph = CoxPHFitter()
    cph.fit(data_subset, duration_col='Survival_months_capped', event_col='Event_capped')
    hr = np.exp(cph.params_['factor'])
    ci_lower, ci_upper = np.exp(cph.confidence_intervals_.loc['factor'])
    p_value = cph.summary.loc['factor', 'p']
    cox_results.append([label, hr, ci_lower, ci_upper, p_value])

cox_df = pd.DataFrame(cox_results, columns=['Factor', 'HR', 'IC 95% Inf', 'IC 95% Sup', 'p-value'])
cox_df['Significancia'] = cox_df['p-value'].apply(lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS')
print(cox_df.to_string(index=False))

# ============================================================
# 7. ANÁLISIS DE SUBGRUPOS: BENEFICIO DE QUIMIO-RADIO
# ============================================================

print("\n" + "="*60)
print("ANÁLISIS DE SUBGRUPOS: BENEFICIO DE TRATAMIENTO")
print("="*60)

subgroups = []

if df_advanced_matched.empty:
    print("\n⚠️ ADVERTENCIA: Subgrupos generados sin ajuste por PSM (interpretación descriptiva).")
    subgroup_source = df[df['T_Advanced'] == 'T3-T4'].copy()
else:
    print("\n--- Subgrupos (cohorte emparejada por PSM) ---")
    subgroup_source = df_advanced_matched.copy()

for t_stage in ['T3', 'T4A', 'T4B']:
    df_sub = subgroup_source[subgroup_source['T_Unified'] == t_stage].copy()
    if len(df_sub) > 30:
        with_both = df_sub[(df_sub['Chemotherapy_Binary']==1) & (df_sub['Radiation_Binary']==1)]
        without_both = df_sub[(df_sub['Chemotherapy_Binary']==0) | (df_sub['Radiation_Binary']==0)]
        
        if len(with_both) > 5 and len(without_both) > 5:
            surv_with = with_both['Survival_months_capped'].median()
            surv_without = without_both['Survival_months_capped'].median()
            mort_with = with_both['Event_capped'].mean()
            mort_without = without_both['Event_capped'].mean()
            
            subgroups.append([
                f"T: {t_stage}",
                len(with_both),
                len(without_both),
                surv_with,
                surv_without,
                mort_with * 100,
                mort_without * 100
            ])

top_histologies = subgroup_source['Histology_Unified'].value_counts().head(5).index
for hist in top_histologies:
    df_sub = subgroup_source[subgroup_source['Histology_Unified'] == hist].copy()
    df_sub = df_sub[df_sub['T_Advanced'] == 'T3-T4']
    
    if len(df_sub) > 30:
        with_both = df_sub[(df_sub['Chemotherapy_Binary']==1) & (df_sub['Radiation_Binary']==1)]
        without_both = df_sub[(df_sub['Chemotherapy_Binary']==0) | (df_sub['Radiation_Binary']==0)]
        
        if len(with_both) > 5 and len(without_both) > 5:
            surv_with = with_both['Survival_months_capped'].median()
            surv_without = without_both['Survival_months_capped'].median()
            mort_with = with_both['Event_capped'].mean()
            mort_without = without_both['Event_capped'].mean()
            
            subgroups.append([
                f"Hist: {hist[:20]}",
                len(with_both),
                len(without_both),
                surv_with,
                surv_without,
                mort_with * 100,
                mort_without * 100
            ])

subgroup_df = pd.DataFrame(subgroups, columns=[
    'Subgrupo', 'N Con Tx', 'N Sin Tx', 
    'Surv Med Con (m)', 'Surv Med Sin (m)',
    'Mort Con (%)', 'Mort Sin (%)'
])
if subgroup_df.empty:
    print("\nNo se identificaron subgrupos con tamaño suficiente tras el emparejamiento/filtros.")
else:
    print("\n" + subgroup_df.to_string(index=False))
    print("\nNota: Resultados sin ajuste causal; interpretar únicamente como descriptivos.")

# ============================================================
# 8. ANÁLISIS ML: GRUPOS DE RIESGO BASALES (CORREGIDO)
# ============================================================

print("\n" + "="*60)
print("ANÁLISIS ML: PERFIL DE RIESGO BASAL")
print("="*60)

if not df_t34_matched.empty:
    df_t34_ml = df_t34_matched.copy()
    print("Cohorte utilizada: pacientes T3/T4 emparejados por puntaje de propensión.")
else:
    df_t34_ml = df[df['T_Advanced'] == 'T3-T4'].copy()
    print("⚠️ Advertencia: no se logró emparejamiento global; se usa la cohorte completa T3/T4 con matching específico por grupo de riesgo.")

if 'Combined_Treatment' not in df_t34_ml.columns:
    df_t34_ml['Combined_Treatment'] = ((df_t34_ml['Chemotherapy_Binary'] == 1) &
                                       (df_t34_ml['Radiation_Binary'] == 1)).astype(int)

print(f"\nPacientes T3/T4 en análisis: {len(df_t34_ml)}")
print(f"Con Quimio+Radio: {df_t34_ml['Combined_Treatment'].sum()}")
print(f"Sin Quimio+Radio: {len(df_t34_ml) - df_t34_ml['Combined_Treatment'].sum()}")

# MODELO DE RIESGO BASAL (sin incluir tratamiento)
print("\n--- Modelo de riesgo basal (predicción de mortalidad) ---")
feature_cols_basal = ['Age_Median', 'Sex', 'T_Unified', 'N_Unified', 'Histology_Unified']

df_t34_encoded = df_t34_ml.copy()

le_dict_t34 = {}
for col in ['Sex', 'T_Unified', 'N_Unified', 'Histology_Unified']:
    le = LabelEncoder()
    df_t34_encoded[col] = le.fit_transform(df_t34_encoded[col].astype(str))
    le_dict_t34[col] = le

X_basal = df_t34_encoded[feature_cols_basal]
y_mortality = df_t34_encoded['Event_capped'] if 'Event_capped' in df_t34_encoded.columns else df_t34_encoded['Event']

# Random Forest para riesgo basal
rf_risk = RandomForestClassifier(n_estimators=100, max_depth=8, 
                                 random_state=42, min_samples_leaf=10,
                                 class_weight='balanced')
rf_risk.fit(X_basal, y_mortality)

df_t34_ml['Predicted_Mortality_RF'] = rf_risk.predict_proba(X_basal)[:, 1]

# GBM para riesgo basal
gbm_risk = GradientBoostingClassifier(n_estimators=100, max_depth=4, 
                                      learning_rate=0.1, random_state=42)
gbm_risk.fit(X_basal, y_mortality)
df_t34_ml['Predicted_Mortality_GBM'] = gbm_risk.predict_proba(X_basal)[:, 1]

print("✓ Modelos de riesgo entrenados")
print(f"  Prob. mortalidad predicha (RF): {df_t34_ml['Predicted_Mortality_RF'].mean():.3f}")
print(f"  Prob. mortalidad predicha (GBM): {df_t34_ml['Predicted_Mortality_GBM'].mean():.3f}")

# Crear grupos de riesgo (terciles)
df_t34_ml['Risk_Group_RF'] = pd.qcut(df_t34_ml['Predicted_Mortality_RF'], 
                                      q=3, 
                                      labels=['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo'])

df_t34_ml['Risk_Group_GBM'] = pd.qcut(df_t34_ml['Predicted_Mortality_GBM'], 
                                       q=3, 
                                       labels=['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo'])

# Mostrar rangos de probabilidad
print("\n--- Rangos de probabilidad de mortalidad (Random Forest) ---")
for risk_group in ['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo']:
    mask = df_t34_ml['Risk_Group_RF'] == risk_group
    probs = df_t34_ml.loc[mask, 'Predicted_Mortality_RF']
    print(f"{risk_group}: {probs.min():.3f} - {probs.max():.3f} (media: {probs.mean():.3f})")

# Características clínicas por grupo de riesgo
print("\n--- Características por Grupo de Riesgo (RF) ---")
for risk_group in ['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo']:
    df_risk = df_t34_ml[df_t34_ml['Risk_Group_RF'] == risk_group]
    print(f"\n{risk_group} (n={len(df_risk)}):")
    print(f"  Edad media: {df_risk['Age_Median'].mean():.1f} años")
    event_col = 'Event_capped' if 'Event_capped' in df_risk.columns else 'Event'
    duration_col = 'Survival_months_capped' if 'Survival_months_capped' in df_risk.columns else 'Survival months'
    print(f"  Mortalidad observada: {df_risk[event_col].mean()*100:.1f}%")
    print(f"  Supervivencia mediana: {df_risk[duration_col].median():.1f} meses")
    print(f"  T más común: {df_risk['T_Unified'].mode().values[0]}")
    print(f"  N más común: {df_risk['N_Unified'].mode().values[0]}")

# ANÁLISIS DE COMPARACIÓN por grupo de riesgo (exploratorio)
print("\n" + "="*60)
print("TRATAMIENTO VS NO TRATAMIENTO POR GRUPO DE RIESGO (EXPLORATORIO)")
print("="*60)

duration_col = 'Survival_months_capped' if 'Survival_months_capped' in df_t34_ml.columns else 'Survival months'
event_col = 'Event_capped' if 'Event_capped' in df_t34_ml.columns else 'Event'

comparison_by_risk_rf = []
risk_psm_balance_rf = {}
risk_covariates = ['Age_Median', 'Sex', 'N_Unified', 'Histology_Unified', 'Stage']

for risk in ['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo']:
    df_risk = df_t34_ml[df_t34_ml['Risk_Group_RF'] == risk].copy()
    if df_risk.empty:
        continue

    treated_n = (df_risk['Combined_Treatment'] == 1).sum()
    control_n = (df_risk['Combined_Treatment'] == 0).sum()
    if treated_n < 5 or control_n < 5:
        print(f"  ⚠️ {risk}: muestras insuficientes para matching (Con Tx={treated_n}, Sin Tx={control_n})")
        continue

    risk_psm_all, risk_matched, _ = propensity_score_match(
        df_risk,
        treatment_col='Combined_Treatment',
        covariates=risk_covariates
    )

    if risk_matched.empty:
        print(f"  ⚠️ {risk}: matching no disponible; se omite análisis.")
        continue

    balance_before_risk = compute_smd(risk_psm_all, 'Combined_Treatment', risk_covariates)
    balance_after_risk = compute_smd(risk_matched, 'Combined_Treatment', risk_covariates)
    balance_table_risk = balance_before_risk.merge(balance_after_risk, on='Covariate', suffixes=('_Antes', '_Despues'))
    balance_table_risk['Cumple_<0.1'] = balance_table_risk['SMD_Despues'].abs() < 0.1
    risk_psm_balance_rf[risk] = balance_table_risk
    print(f"  {risk}: máximo |SMD| post-matching = {balance_table_risk['SMD_Despues'].abs().max():.3f}")

    with_tx = risk_matched[risk_matched['Combined_Treatment'] == 1]
    without_tx = risk_matched[risk_matched['Combined_Treatment'] == 0]

    if len(with_tx) >= 5 and len(without_tx) >= 5:
        surv_with = with_tx[duration_col].median()
        surv_without = without_tx[duration_col].median()
        mort_with = with_tx[event_col].mean()
        mort_without = without_tx[event_col].mean()

        result_lr = logrank_test(with_tx[duration_col], without_tx[duration_col],
                                 with_tx[event_col], without_tx[event_col])

        comparison_by_risk_rf.append({
            'Grupo de Riesgo': risk,
            'N Con Tx (matched)': len(with_tx),
            'N Sin Tx (matched)': len(without_tx),
            'Surv Con Tx (m)': surv_with,
            'Surv Sin Tx (m)': surv_without,
            'Delta_Supervivencia_m': surv_with - surv_without,
            'Mort Con Tx (%)': mort_with * 100,
            'Mort Sin Tx (%)': mort_without * 100,
            'Delta_Mortalidad_pct': (mort_without - mort_with) * 100,
            'p-value': result_lr.p_value,
            'p<0.05': 'SI' if result_lr.p_value < 0.05 else 'NO'
        })
    else:
        print(f"  ⚠️ {risk}: tamaño insuficiente tras matching para prueba log-rank.")

comparison_rf_df = pd.DataFrame(comparison_by_risk_rf)
print("\n--- Comparación por grupo de riesgo (Random Forest) ---")
if comparison_rf_df.empty:
    print("No se pudo realizar análisis con matching por grupo; interpretar con cautela.")
else:
    print(comparison_rf_df.to_string(index=False))

# Mismo análisis con GBM
comparison_by_risk_gbm = []
risk_psm_balance_gbm = {}

for risk in ['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo']:
    df_risk = df_t34_ml[df_t34_ml['Risk_Group_GBM'] == risk].copy()
    if df_risk.empty:
        continue

    treated_n = (df_risk['Combined_Treatment'] == 1).sum()
    control_n = (df_risk['Combined_Treatment'] == 0).sum()
    if treated_n < 5 or control_n < 5:
        print(f"  ⚠️ {risk} (GBM): muestras insuficientes para matching (Con Tx={treated_n}, Sin Tx={control_n})")
        continue

    risk_psm_all, risk_matched, _ = propensity_score_match(
        df_risk,
        treatment_col='Combined_Treatment',
        covariates=risk_covariates
    )

    if risk_matched.empty:
        print(f"  ⚠️ {risk} (GBM): matching no disponible; se omite análisis.")
        continue

    balance_before_risk = compute_smd(risk_psm_all, 'Combined_Treatment', risk_covariates)
    balance_after_risk = compute_smd(risk_matched, 'Combined_Treatment', risk_covariates)
    balance_table_risk = balance_before_risk.merge(balance_after_risk, on='Covariate', suffixes=('_Antes', '_Despues'))
    balance_table_risk['Cumple_<0.1'] = balance_table_risk['SMD_Despues'].abs() < 0.1
    risk_psm_balance_gbm[risk] = balance_table_risk
    print(f"  {risk} (GBM): máximo |SMD| post-matching = {balance_table_risk['SMD_Despues'].abs().max():.3f}")

    with_tx = risk_matched[risk_matched['Combined_Treatment'] == 1]
    without_tx = risk_matched[risk_matched['Combined_Treatment'] == 0]

    if len(with_tx) >= 5 and len(without_tx) >= 5:
        surv_with = with_tx[duration_col].median()
        surv_without = without_tx[duration_col].median()
        mort_with = with_tx[event_col].mean()
        mort_without = without_tx[event_col].mean()

        result_lr = logrank_test(with_tx[duration_col], without_tx[duration_col],
                                 with_tx[event_col], without_tx[event_col])

        comparison_by_risk_gbm.append({
            'Grupo de Riesgo': risk,
            'N Con Tx (matched)': len(with_tx),
            'N Sin Tx (matched)': len(without_tx),
            'Surv Con Tx (m)': surv_with,
            'Surv Sin Tx (m)': surv_without,
            'Delta_Supervivencia_m': surv_with - surv_without,
            'Mort Con Tx (%)': mort_with * 100,
            'Mort Sin Tx (%)': mort_without * 100,
            'Delta_Mortalidad_pct': (mort_without - mort_with) * 100,
            'p-value': result_lr.p_value,
            'p<0.05': 'SI' if result_lr.p_value < 0.05 else 'NO'
        })
    else:
        print(f"  ⚠️ {risk} (GBM): tamaño insuficiente tras matching para prueba log-rank.")

comparison_gbm_df = pd.DataFrame(comparison_by_risk_gbm)
print("\n--- Comparación por grupo de riesgo (GBM) ---")
if comparison_gbm_df.empty:
    print("Celdas insuficientes tras el emparejamiento; interpretar con cautela.")
else:
    print(comparison_gbm_df.to_string(index=False))

print("\n--- Modelo de Cox con interacción Tratamiento x Grupo de Riesgo (RF) ---")
cox_df = df_t34_ml[['Survival_months_capped', 'Event_capped', 'Combined_Treatment', 'Risk_Group_RF']].dropna()
if cox_df.empty or cox_df['Risk_Group_RF'].nunique() < 2:
    print("Datos insuficientes para ajustar el modelo de Cox con interacción.")
else:
    cph = CoxPHFitter()
    try:
        cph.fit(
            cox_df,
            duration_col='Survival_months_capped',
            event_col='Event_capped',
            formula="Combined_Treatment + C(Risk_Group_RF) + Combined_Treatment:C(Risk_Group_RF)"
        )
        print(cph.summary[['coef', 'exp(coef)', 'p']])
    except Exception as exc:
        print(f"No fue posible ajustar el modelo de Cox con interacción: {exc}")

# Análisis adicional por características específicas
print("\n--- Análisis por Características Específicas ---")

df_t34_ml['Age_Group'] = pd.cut(df_t34_ml['Age_Median'], 
                                bins=[0, 50, 65, 100], 
                                labels=['<50', '50-65', '>65'])

print("\nBeneficio por Grupo de Edad:")
for age_group in ['<50', '50-65', '>65']:
    df_age = df_t34_ml[df_t34_ml['Age_Group'] == age_group]
    with_tx = df_age[df_age['Combined_Treatment'] == 1]
    without_tx = df_age[df_age['Combined_Treatment'] == 0]
    
    if len(with_tx) >= 5 and len(without_tx) >= 5:
        diff = with_tx[duration_col].median() - without_tx[duration_col].median()
        result_lr = logrank_test(with_tx[duration_col], without_tx[duration_col],
                                with_tx[event_col], without_tx[event_col])
        sig = '*' if result_lr.p_value < 0.05 else 'NS'
        print(f"  {age_group}: Delta={diff:+.1f} meses, p={result_lr.p_value:.3f} {sig}")

print("\nBeneficio por Estadio N:")
for n_stage in ['N0', 'N1', 'N2', 'N3']:
    df_n = df_t34_ml[df_t34_ml['N_Unified'] == n_stage]
    with_tx = df_n[df_n['Combined_Treatment'] == 1]
    without_tx = df_n[df_n['Combined_Treatment'] == 0]
    
    if len(with_tx) >= 5 and len(without_tx) >= 5:
        diff = with_tx[duration_col].median() - without_tx[duration_col].median()
        result_lr = logrank_test(with_tx[duration_col], without_tx[duration_col],
                                with_tx[event_col], without_tx[event_col])
        sig = '*' if result_lr.p_value < 0.05 else 'NS'
        print(f"  {n_stage}: Delta={diff:+.1f} meses, p={result_lr.p_value:.3f} {sig}")

# ============================================================
# 9. ANÁLISIS TRADICIONAL: SUBGRUPOS TNM
# ============================================================

print("\n" + "="*60)
print("ANÁLISIS TRADICIONAL: BENEFICIO POR COMBINACIÓN T-N")
print("="*60)

beneficio_results = []

for t_stage in ['T3', 'T4A', 'T4B']:
    for n_stage in ['N0', 'N1', 'N2']:
        df_sub = subgroup_source[(subgroup_source['T_Unified'] == t_stage) & (subgroup_source['N_Unified'] == n_stage)].copy()
        
        if len(df_sub) >= 20:
            with_both = df_sub[(df_sub['Chemotherapy_Binary']==1) & (df_sub['Radiation_Binary']==1)]
            without_both = df_sub[(df_sub['Chemotherapy_Binary']==0) | (df_sub['Radiation_Binary']==0)]
            
            if len(with_both) >= 5 and len(without_both) >= 5:
                result_lr = logrank_test(
                    with_both['Survival_months_capped'],
                    without_both['Survival_months_capped'],
                    with_both['Event_capped'],
                    without_both['Event_capped']
                )
                
                surv_diff = with_both['Survival_months_capped'].median() - without_both['Survival_months_capped'].median()
                mort_diff = without_both['Event_capped'].mean() - with_both['Event_capped'].mean()
                
                beneficio_results.append({
                    'Subgrupo': f"{t_stage} {n_stage}",
                    'N Con Tx': len(with_both),
                    'N Sin Tx': len(without_both),
                    'Dif Surv (m)': surv_diff,
                    'Dif Mort (%)': mort_diff * 100,
                    'p-value': result_lr.p_value,
                    'Significativo': 'Sí' if result_lr.p_value < 0.05 else 'No'
                })

beneficio_df = pd.DataFrame(beneficio_results)
if not beneficio_df.empty:
    beneficio_df = beneficio_df.sort_values('Dif Surv (m)', ascending=False)
    print("\n--- Subgrupos con Beneficio Significativo ---")
    print(beneficio_df[beneficio_df['Significativo'] == 'Sí'].to_string(index=False))
    print("\n--- Todos los Subgrupos Analizados ---")
    print(beneficio_df.to_string(index=False))
    print("\nNota: Comparaciones descriptivas sin ajuste adicional; interpretar con cautela.")
else:
    beneficio_df = pd.DataFrame(columns=['Subgrupo', 'N Con Tx', 'N Sin Tx',
                                         'Dif Surv (m)', 'Dif Mort (%)', 'p-value', 'Significativo'])
    print("No se encontraron subgrupos suficientemente grandes")

# ============================================================
# 10. GENERAR PDF CON GRÁFICAS
# ============================================================

print("\n" + "="*60)
print("GENERANDO REPORTE PDF")
print("="*60)

# ============================================================
# 10. SWEET SPOTS DE QUIMIO+RADIO
# ============================================================

print("\n" + "="*60)
print("IDENTIFICACIÓN DE SUBGRUPOS CON MAYOR BENEFICIO (SWEET SPOTS)")
print("Metodología: Pre-especificación + Corrección FDR + PSM opcional")
print("="*60)

# --------------------------------------------------------------
# CONFIGURACIÓN: Subgrupos pre-especificados con justificación clínica
# --------------------------------------------------------------

PRESPECIFIED_SEGMENTS = [
    ('Edad x N', ['Age_Group', 'N_Unified']),              # Edad y nodos determinan pronóstico
    ('Stage x Histologia', ['Stage', 'Histology_Group']),  # Interacción biológica conocida
    ('T x N', ['T_Unified', 'N_Unified']),                 # Estadificación estándar
    ('Edad x Histologia', ['Age_Group', 'Histology_Group'])
]

print(f"\nSubgrupos pre-especificados: {len(PRESPECIFIED_SEGMENTS)}")
print("Justificación: literatura previa y plausibilidad biológica.")

# --------------------------------------------------------------
# PREPARAR DATOS
# --------------------------------------------------------------

df_sweet = df.copy()

df_sweet['Age_Group'] = pd.cut(
    df_sweet['Age_Median'],
    bins=[0, 50, 65, 120],
    labels=['<50', '50-65', '>65']
)

top_histologies = df_sweet['Histology_Unified'].value_counts().head(6).index
df_sweet['Histology_Group'] = df_sweet['Histology_Unified'].where(
    df_sweet['Histology_Unified'].isin(top_histologies),
    'Otros'
)

df_sweet['Combined_Treatment'] = (
    (df_sweet['Chemotherapy_Binary'] == 1) &
    (df_sweet['Radiation_Binary'] == 1)
).astype(int)

# --------------------------------------------------------------
# CRITERIOS DE INCLUSIÓN RIGUROSOS
# --------------------------------------------------------------

MIN_TOTAL = 40
MIN_TREATED = 15
MIN_CONTROL = 15
MIN_TREATMENT_RATE = 0.15

print("\nCriterios de inclusión:")
print(f"  • N total ≥ {MIN_TOTAL}")
print(f"  • N tratados ≥ {MIN_TREATED}")
print(f"  • N controles ≥ {MIN_CONTROL}")
print(f"  • Tasa de tratamiento ≥ {MIN_TREATMENT_RATE*100:.0f}%")

# --------------------------------------------------------------
# ANÁLISIS MULTINIVEL
# --------------------------------------------------------------

sweet_spots_results = []

print("\n--- Analizando subgrupos ---")

for segment_name, cols in PRESPECIFIED_SEGMENTS:
    print(f"\n{segment_name}:")
    for values, subset in df_sweet.groupby(cols, dropna=False):
        if subset.shape[0] < MIN_TOTAL:
            continue

        with_tx = subset[subset['Combined_Treatment'] == 1]
        without_tx = subset[subset['Combined_Treatment'] == 0]

        if len(with_tx) < MIN_TREATED or len(without_tx) < MIN_CONTROL:
            continue

        treatment_rate = len(with_tx) / len(subset)
        if treatment_rate < MIN_TREATMENT_RATE:
            continue

        value_labels = values if isinstance(values, tuple) else (values,)
        value_labels = [
            str(v) if (v is not None and v == v) else 'Desconocido'
            for v in value_labels
        ]
        segment_label = " | ".join(value_labels)

        # Nivel 1: análisis descriptivo (log-rank sin ajuste)
        try:
            lr_naive = logrank_test(
                with_tx['Survival_months_capped'],
                without_tx['Survival_months_capped'],
                with_tx['Event_capped'],
                without_tx['Event_capped']
            )
            p_naive = lr_naive.p_value
        except Exception:
            continue

        surv_with = with_tx['Survival_months_capped'].median()
        surv_without = without_tx['Survival_months_capped'].median()
        surv_gain = surv_with - surv_without

        mort_with = with_tx['Event_capped'].mean()
        mort_without = without_tx['Event_capped'].mean()
        mort_reduction = mort_without - mort_with

        # Nivel 2: PSM dentro del subgrupo
        p_psm = np.nan
        surv_gain_psm = np.nan
        mort_reduction_psm = np.nan
        psm_n = 0

        base_covariates = ['Age_Median', 'Sex', 'N_Unified', 'Histology_Unified']
        available_covariates = [c for c in base_covariates if c not in cols]

        if len(subset) >= 60 and len(available_covariates) >= 2:
            try:
                subset_clean = subset.dropna(subset=available_covariates + ['Combined_Treatment'])
                if len(subset_clean) >= 40:
                    _, subset_matched, _ = propensity_score_match(
                        subset_clean,
                        treatment_col='Combined_Treatment',
                        covariates=available_covariates
                    )

                    if not subset_matched.empty and len(subset_matched) >= 20:
                        with_tx_psm = subset_matched[subset_matched['Combined_Treatment'] == 1]
                        without_tx_psm = subset_matched[subset_matched['Combined_Treatment'] == 0]

                        if len(with_tx_psm) >= 5 and len(without_tx_psm) >= 5:
                            lr_psm = logrank_test(
                                with_tx_psm['Survival_months_capped'],
                                without_tx_psm['Survival_months_capped'],
                                with_tx_psm['Event_capped'],
                                without_tx_psm['Event_capped']
                            )
                            p_psm = lr_psm.p_value
                            surv_gain_psm = with_tx_psm['Survival_months_capped'].median() - without_tx_psm['Survival_months_capped'].median()
                            mort_reduction_psm = without_tx_psm['Event_capped'].mean() - with_tx_psm['Event_capped'].mean()
                            psm_n = len(subset_matched)
            except Exception:
                pass

        # Nivel 3: Cox univariante ajustado
        hr_adjusted = np.nan
        hr_ci_low = np.nan
        hr_ci_high = np.nan
        p_cox = np.nan

        try:
            cox_data = subset[['Survival_months_capped', 'Event_capped', 'Combined_Treatment']].dropna()
            if len(cox_data) >= 30 and cox_data['Event_capped'].sum() >= 10:
                cph = CoxPHFitter()
                cph.fit(cox_data, duration_col='Survival_months_capped', event_col='Event_capped')
                hr_adjusted = np.exp(cph.params_['Combined_Treatment'])
                hr_ci_low, hr_ci_high = np.exp(cph.confidence_intervals_.loc['Combined_Treatment'])
                p_cox = cph.summary.loc['Combined_Treatment', 'p']
        except Exception:
            pass

        print(f"  {segment_label}: n={len(subset)}, Delta={surv_gain:+.1f} meses, p={p_naive:.3f}")

        sweet_spots_results.append({
            'Segmento': segment_name,
            'Detalle': segment_label,
            'N Total': len(subset),
            'N Con Tx': len(with_tx),
            'N Sin Tx': len(without_tx),
            'Tx (%)': treatment_rate * 100,
            'Surv Con Tx (m)': surv_with,
            'Surv Sin Tx (m)': surv_without,
            'Dif Surv (m)': surv_gain,
            'Mort Con Tx (%)': mort_with * 100,
            'Mort Sin Tx (%)': mort_without * 100,
            'Reduccion Mort (%)': mort_reduction * 100,
            'p_logrank_naive': p_naive,
            'N PSM': psm_n,
            'Dif Surv PSM (m)': surv_gain_psm,
            'Reduccion Mort PSM (%)': mort_reduction_psm * 100 if not np.isnan(mort_reduction_psm) else np.nan,
            'p_logrank_PSM': p_psm,
            'HR Cox': hr_adjusted,
            'HR CI Low': hr_ci_low,
            'HR CI High': hr_ci_high,
            'p_Cox': p_cox
        })

# --------------------------------------------------------------
# CORRECCIÓN POR MÚLTIPLES COMPARACIONES (FDR)
# --------------------------------------------------------------

sweet_spots_df = pd.DataFrame(sweet_spots_results)

if sweet_spots_df.empty:
    print("\n⚠️ No se encontraron subgrupos que cumplan los criterios.")
    sweets_summary = {'top': pd.DataFrame(), 'sig': pd.DataFrame(), 'psm': pd.DataFrame()}
else:
    print(f"\n✓ {len(sweet_spots_df)} subgrupos analizados.")

    if multipletests is None:
        print("\n⚠️ Corrección FDR omitida: instalar 'statsmodels' para habilitarla (pip install statsmodels).")
        sweet_spots_df['p_FDR_naive'] = sweet_spots_df['p_logrank_naive']
        sweet_spots_df['Sig_FDR_naive'] = False
        sweet_spots_df['p_FDR_PSM'] = sweet_spots_df['p_logrank_PSM']
        sweet_spots_df['Sig_FDR_PSM'] = False
    else:
        p_values_naive = sweet_spots_df['p_logrank_naive'].dropna()
        if len(p_values_naive) > 0:
            _, p_adjusted_naive, _, _ = multipletests(p_values_naive, method='fdr_bh', alpha=0.05)
            sweet_spots_df.loc[p_values_naive.index, 'p_FDR_naive'] = p_adjusted_naive
            sweet_spots_df['Sig_FDR_naive'] = sweet_spots_df['p_FDR_naive'] < 0.05
        else:
            sweet_spots_df['p_FDR_naive'] = np.nan
            sweet_spots_df['Sig_FDR_naive'] = False

        p_values_psm = sweet_spots_df['p_logrank_PSM'].dropna()
        if len(p_values_psm) >= 2:
            _, p_adjusted_psm, _, _ = multipletests(p_values_psm, method='fdr_bh', alpha=0.05)
            sweet_spots_df.loc[p_values_psm.index, 'p_FDR_PSM'] = p_adjusted_psm
            sweet_spots_df['Sig_FDR_PSM'] = sweet_spots_df['p_FDR_PSM'] < 0.05
        else:
            sweet_spots_df['p_FDR_PSM'] = np.nan
            sweet_spots_df['Sig_FDR_PSM'] = False

    sweet_spots_df['p_logrank_naive'] = sweet_spots_df['p_logrank_naive'].fillna(1.0)
    sweet_spots_df['p_logrank_PSM'] = sweet_spots_df['p_logrank_PSM'].fillna(1.0)
    sweet_spots_df['Best_p'] = sweet_spots_df[['p_logrank_PSM', 'p_logrank_naive']].min(axis=1)
    sweet_spots_df['Sig_PSM_rank'] = sweet_spots_df['Sig_FDR_PSM'].astype(int)
    sweet_spots_df['Sig_naive_rank'] = sweet_spots_df['Sig_FDR_naive'].astype(int)

    def _interpret_sweet(row):
        best_p = row['Best_p'] if not pd.isna(row['Best_p']) else 1.0
        if row['Sig_FDR_PSM']:
            return f"Benefit confirmed (PSM FDR<0.05, p≈{best_p:.3f})"
        if row['Sig_FDR_naive']:
            return f"Benefit suggested (naïve FDR<0.05, p≈{best_p:.3f})"
        if row['Dif Surv (m)'] > 0:
            return f"No clear benefit evidence (p≈{best_p:.3f})"
        return f"No benefit (p≈{best_p:.3f})"

    sweet_spots_df['Interpretation'] = sweet_spots_df.apply(_interpret_sweet, axis=1)

    sweet_spots_df = sweet_spots_df.sort_values(
        ['Sig_PSM_rank', 'Sig_naive_rank', 'Best_p', 'Dif Surv (m)'],
        ascending=[False, False, True, False]
    ).reset_index(drop=True)
    sweet_spots_df = sweet_spots_df.drop(columns=['Sig_PSM_rank', 'Sig_naive_rank'])

    sweets_top = sweet_spots_df.head(15).copy()
    sweets_sig_naive = sweet_spots_df[
        (sweet_spots_df['Sig_FDR_naive']) &
        (sweet_spots_df['Dif Surv (m)'] > 0)
    ].copy()
    sweets_psm_confirmed = sweet_spots_df[
        (sweet_spots_df['Sig_FDR_PSM']) &
        (sweet_spots_df['Dif Surv PSM (m)'] > 0)
    ].copy()

    sweets_summary = {
        'top': sweets_top,
        'sig': sweets_sig_naive,
        'psm': sweets_psm_confirmed
    }

    print("\n" + "="*60)
    print("RESULTADOS SWEET SPOTS (con corrección FDR)")
    print("="*60)

    print("\n--- TIER 1: Confirmados con PSM ---")
    if len(sweets_psm_confirmed) > 0:
        print(f"✓ {len(sweets_psm_confirmed)} subgrupos con beneficio confirmado tras PSM.")
        cols_psm = ['Segmento', 'Detalle', 'N Total', 'N PSM', 'Dif Surv PSM (m)', 'Reduccion Mort PSM (%)', 'p_FDR_PSM']
        print(sweets_psm_confirmed[cols_psm].to_string(index=False))
    else:
        print("✗ Ningún subgrupo alcanza significancia tras PSM y FDR (muestras pequeñas).")

    print("\n--- TIER 2: Significativos sin ajuste causal ---")
    if len(sweets_sig_naive) > 0:
        print(f"✓ {len(sweets_sig_naive)} subgrupos con p_FDR < 0.05 (análisis descriptivo).")
        cols_naive = ['Segmento', 'Detalle', 'N Total', 'Dif Surv (m)', 'Reduccion Mort (%)', 'p_FDR_naive']
        print(sweets_sig_naive[cols_naive].head(10).to_string(index=False))
        print("\n⚠️ Advertencia: Interpretar como exploratorio; requiere validación adicional.")
    else:
        print("✗ Ningún subgrupo mantiene significancia tras corrección FDR en análisis naive.")

    print("\n--- TIER 3: Top beneficios descriptivos ---")
    cols_top = ['Segmento', 'Detalle', 'N Total', 'Dif Surv (m)', 'p_logrank_naive', 'p_FDR_naive']
    print(sweets_top.head(10)[cols_top].to_string(index=False))
    print("\n⚠️ Interpretación descriptiva: requiere confirmación externa.")

    print("\n" + "="*60)
    print("ESTADÍSTICAS RESUMEN")
    print("="*60)

    n_total = len(sweet_spots_df)
    n_positive = (sweet_spots_df['Dif Surv (m)'] > 0).sum()
    n_naive_sig = (sweet_spots_df['p_logrank_naive'] < 0.05).sum()
    n_fdr_sig = sweet_spots_df['Sig_FDR_naive'].sum()
    n_psm_sig = sweet_spots_df['Sig_FDR_PSM'].sum()
    n_psm_attempts = (sweet_spots_df['N PSM'] > 0).sum()

    print(f"Total subgrupos evaluados: {n_total}")
    print(f"Con efecto positivo (Delta > 0): {n_positive} ({n_positive/n_total*100:.1f}%)")
    print(f"p < 0.05 sin corrección: {n_naive_sig} ({n_naive_sig/n_total*100:.1f}%)")
    print(f"p_FDR < 0.05 (naive): {n_fdr_sig} ({n_fdr_sig/n_total*100:.1f}%)")
    print(f"p_FDR < 0.05 (PSM): {n_psm_sig} ({n_psm_sig/n_total*100:.1f}%)")
    print(f"Subgrupos con PSM aplicado: {n_psm_attempts} ({n_psm_attempts/n_total*100:.1f}%)")

pdf_filename = str(BASE_DIR / 'Analisis_Cancer_Glandula_Salival.pdf')

with PdfPages(pdf_filename) as pdf:
    
    # PÁGINA 1: Descripción general
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Análisis de Supervivencia - Cáncer de Glándula Salival', fontsize=16, fontweight='bold')
    
    ax1 = plt.subplot(2, 2, 1)
    stage_counts = df['Stage'].value_counts()
    stage_counts.plot(kind='bar', color='steelblue', ax=ax1)
    ax1.set_title('Distribución por Estadio TNM')
    ax1.set_xlabel('Estadio')
    ax1.set_ylabel('Número de Pacientes')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = plt.subplot(2, 2, 2)
    t_counts = df['T_Unified'].value_counts().head(8)
    t_counts.plot(kind='bar', color='coral', ax=ax2)
    ax2.set_title('Distribución por Estadio T')
    ax2.set_xlabel('Estadio T')
    ax2.set_ylabel('Número de Pacientes')
    ax2.tick_params(axis='x', rotation=45)
    
    ax3 = plt.subplot(2, 2, 3)
    treat_counts = df['Treatment_Group'].value_counts()
    treat_counts.plot(kind='bar', color='mediumseagreen', ax=ax3)
    ax3.set_title('Distribución de Tratamientos')
    ax3.set_xlabel('Tipo de Tratamiento')
    ax3.set_ylabel('Número de Pacientes')
    ax3.tick_params(axis='x', rotation=45)
    
    ax4 = plt.subplot(2, 2, 4)
    surv_by_stage = df.groupby('Stage')['Survival_months_capped'].median().sort_values()
    surv_by_stage.plot(kind='barh', color='plum', ax=ax4)
    ax4.set_title('Supervivencia Mediana por Estadio')
    ax4.set_xlabel('Meses')
    ax4.set_ylabel('Estadio')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    if calibration_fig is not None:
        pdf.savefig(calibration_fig, bbox_inches='tight')
        plt.close(calibration_fig)
    
    if love_plot_fig is not None:
        pdf.savefig(love_plot_fig, bbox_inches='tight')
        plt.close(love_plot_fig)
    
    if not psm_balance_table.empty:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Balance de Covariables Post-PSM (SMD)', fontsize=16, fontweight='bold', loc='left', pad=20)
        table_display = psm_balance_table[['Covariate', 'SMD_Antes', 'SMD_Despues', 'Cumple_<0.1']].copy()
        table_display['SMD_Antes'] = table_display['SMD_Antes'].map(lambda x: f"{x:.3f}")
        table_display['SMD_Despues'] = table_display['SMD_Despues'].map(lambda x: f"{x:.3f}")
        table_display['Cumple_<0.1'] = table_display['Cumple_<0.1'].map(lambda x: 'Sí' if x else 'No')
        table = ax.table(cellText=table_display.values,
                         colLabels=['Covariable', 'SMD Antes', 'SMD Después', 'Cumple <0.1'],
                         loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    if missingness_fig is not None:
        pdf.savefig(missingness_fig, bbox_inches='tight')
        plt.close(missingness_fig)
    
    # PÁGINA 2: Curvas ROC
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Curvas ROC - Modelos Predictivos', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for idx, (name, result) in enumerate(results.items()):
        plt.plot(result['fpr'], result['tpr'], 
                color=colors[idx], lw=2,
                label=f'{name} (AUC = {result["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Azar (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Comparación de Modelos Predictivos de Mortalidad', fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # PÁGINA 3: Importancia de variables
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Importancia de Variables (Random Forest)', fontsize=16, fontweight='bold')
    
    ax = plt.subplot(1, 1, 1)
    y_pos = np.arange(len(feature_importance))
    ax.barh(y_pos, feature_importance['Importance'], color='teal')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_importance['Feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importancia', fontsize=12)
    ax.set_title('Variables con Mayor Impacto en la Predicción', fontsize=14, pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # PÁGINA 4: Kaplan-Meier por Estadio
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Curvas de Kaplan-Meier por Estadio TNM', fontsize=16, fontweight='bold')
    
    ax = plt.subplot(1, 1, 1)
    kmf = KaplanMeierFitter()
    
    stages_to_plot = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    colors_km = ['green', 'blue', 'orange', 'red']
    
    for stage, color in zip(stages_to_plot, colors_km):
        mask = df['Stage'] == stage
        if mask.sum() > 10:
            kmf.fit(df.loc[mask, 'Survival_months_capped'], 
                   df.loc[mask, 'Event_capped'], 
                   label=f'{stage} (n={mask.sum()})')
            kmf.plot_survival_function(ax=ax, color=color, linewidth=2)
    
    ax.set_xlabel('Tiempo (meses)', fontsize=12)
    ax.set_ylabel('Probabilidad de Supervivencia', fontsize=12)
    ax.set_title('Supervivencia Global por Estadio TNM', fontsize=14, pad=20)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # PÁGINA 5: Kaplan-Meier T3/T4 por tratamiento
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Kaplan-Meier: Estadios T3/T4 por Tratamiento', fontsize=16, fontweight='bold')
    
    ax = plt.subplot(1, 1, 1)
    kmf = KaplanMeierFitter()
    
    df_t34 = df[df['T_Advanced'] == 'T3-T4'].copy()
    
    treatments = ['Chemo+Radio', 'Radio Only', 'No Treatment']
    colors_tx = ['purple', 'orange', 'gray']
    
    for tx, color in zip(treatments, colors_tx):
        mask = df_t34['Treatment_Group'] == tx
        if mask.sum() > 10:
            kmf.fit(df_t34.loc[mask, 'Survival_months_capped'],
                    df_t34.loc[mask, 'Event_capped'],
                    label=f'{tx} (n={mask.sum()})')
            kmf.plot_survival_function(ax=ax, color=color, linewidth=2)
    
    ax.set_xlabel('Tiempo (meses)', fontsize=12)
    ax.set_ylabel('Probabilidad de Supervivencia', fontsize=12)
    ax.set_title('Efecto del Tratamiento en Tumores Avanzados (T3/T4)', fontsize=14, pad=20)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # PÁGINA 6: Hazard Ratios
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Hazard Ratios - Factores de Riesgo', fontsize=16, fontweight='bold')
    
    ax = plt.subplot(1, 1, 1)
    
    y_pos = np.arange(len(cox_df))
    
    p_col = 'p-value' if 'p-value' in cox_df.columns else ('p_value' if 'p_value' in cox_df.columns else None)
    hr_col_candidates = [col for col in cox_df.columns if col.lower() in ['hr', 'hazard ratio'] or 'exp' in col.lower()]
    hr_col = hr_col_candidates[0] if hr_col_candidates else None
    lower_candidates = [col for col in cox_df.columns if any(word in col.lower() for word in ['inf', 'low', 'lower', '2.5'])]
    upper_candidates = [col for col in cox_df.columns if any(word in col.lower() for word in ['sup', 'high', 'upper', '97.5'])]
    ci_low_col = lower_candidates[0] if lower_candidates else None
    ci_high_col = upper_candidates[0] if upper_candidates else None

    if hr_col is None:
        print("⚠️ No se encontró columna de hazard ratios para graficar; forest plot omitido.")
    else:
        for i, row in cox_df.iterrows():
            hr_val = row[hr_col] if hr_col in row else np.nan
            ci_low = row[ci_low_col] if ci_low_col and ci_low_col in row else np.nan
            ci_high = row[ci_high_col] if ci_high_col and ci_high_col in row else np.nan
            p_val_row = row[p_col] if p_col and p_col in row else np.nan
            color = 'darkred' if pd.notnull(p_val_row) and p_val_row < 0.05 else 'gray'
            ax.plot([ci_low, ci_high], [i, i], 'k-', linewidth=2, color=color)
            ax.plot(hr_val, i, 'o', markersize=10, color=color)
        
        ax.axvline(x=1, color='blue', linestyle='--', linewidth=1.5, label='HR = 1 (Sin efecto)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cox_df['Factor'])
        ax.set_xlabel('Hazard Ratio (IC 95%)', fontsize=12)
        ax.set_title('Forest Plot: Factores Asociados a Mortalidad', fontsize=14, pad=20)
        ax.grid(axis='x', alpha=0.3)
        ax.legend(fontsize=10)
        
        for i, row in cox_df.iterrows():
            hr_val = row[hr_col] if hr_col in row else np.nan
            ci_low = row[ci_low_col] if ci_low_col and ci_low_col in row else np.nan
            ci_high = row[ci_high_col] if ci_high_col and ci_high_col in row else np.nan
            p_val_row = row[p_col] if p_col and p_col in row else np.nan
            p_text = f"{p_val_row:.4f}" if pd.notnull(p_val_row) else "NA"
            if pd.notnull(ci_low) and pd.notnull(ci_high):
                text_str = f"HR={hr_val:.2f}\n[{ci_low:.2f}-{ci_high:.2f}]\np={p_text}"
                ax.text(ci_high + 0.1, i, text_str, va='center', fontsize=8)
            else:
                text_str = f"HR={hr_val:.2f}\np={p_text}"
                ax.text(hr_val + 0.1, i, text_str, va='center', fontsize=8)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # PÁGINA 7: Grupos de Riesgo ML - Características
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Modelo ML: Grupos de Riesgo Basales (T3/T4)', fontsize=16, fontweight='bold')
    
    # Gráfica 1: Distribución de probabilidades
    ax1 = plt.subplot(2, 2, 1)
    for risk, color in zip(['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo'], ['green', 'orange', 'red']):
        mask = df_t34_ml['Risk_Group_RF'] == risk
        ax1.hist(df_t34_ml.loc[mask, 'Predicted_Mortality_RF'], alpha=0.6, label=risk, color=color, bins=15)
    ax1.set_xlabel('Probabilidad de Mortalidad (RF)')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Probabilidades por Grupo')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Gráfica 2: Mortalidad observada por grupo
    ax2 = plt.subplot(2, 2, 2)
    mort_col_plot = 'Event_capped' if 'Event_capped' in df_t34_ml.columns else 'Event'
    mort_by_risk = df_t34_ml.groupby('Risk_Group_RF')[mort_col_plot].mean() * 100
    mort_by_risk.plot(kind='bar', ax=ax2, color=['green', 'orange', 'red'])
    ax2.set_xlabel('Grupo de Riesgo')
    ax2.set_ylabel('Mortalidad Observada (%)')
    ax2.set_title('Mortalidad Real por Grupo de Riesgo')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Gráfica 3: Supervivencia mediana por grupo
    ax3 = plt.subplot(2, 2, 3)
    duration_col_plot = 'Survival_months_capped' if 'Survival_months_capped' in df_t34_ml.columns else 'Survival months'
    surv_by_risk = df_t34_ml.groupby('Risk_Group_RF')[duration_col_plot].median()
    surv_by_risk.plot(kind='bar', ax=ax3, color=['green', 'orange', 'red'])
    ax3.set_xlabel('Grupo de Riesgo')
    ax3.set_ylabel('Supervivencia Mediana (meses)')
    ax3.set_title('Supervivencia por Grupo de Riesgo')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Gráfica 4: Edad promedio por grupo
    ax4 = plt.subplot(2, 2, 4)
    age_by_risk = df_t34_ml.groupby('Risk_Group_RF')['Age_Median'].mean()
    age_by_risk.plot(kind='bar', ax=ax4, color=['green', 'orange', 'red'])
    ax4.set_xlabel('Grupo de Riesgo')
    ax4.set_ylabel('Edad Promedio (años)')
    ax4.set_title('Edad por Grupo de Riesgo')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # PÁGINA 8: Tratamiento vs no tratamiento por grupo de riesgo (exploratorio)
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Quimio+Radio vs Otros por Grupo de Riesgo (Exploratorio)', fontsize=16, fontweight='bold')
    
    kmf = KaplanMeierFitter()
    
    if not comparison_rf_df.empty:
        ax1 = plt.subplot(2, 2, 1)
        colors_bar = ['green' if x > 0 else 'red' for x in comparison_rf_df['Delta_Supervivencia_m']]
        ax1.barh(comparison_rf_df['Grupo de Riesgo'], comparison_rf_df['Delta_Supervivencia_m'], 
                 color=colors_bar, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Delta supervivencia (meses)')
        ax1.set_title('Mediana (RF)')
        ax1.grid(axis='x', alpha=0.3)
        for i, row in comparison_rf_df.iterrows():
            sig = '*' if row['p-value'] < 0.05 else ''
            ax1.text(row['Delta_Supervivencia_m'], i, f" {row['Delta_Supervivencia_m']:.1f}m {sig}", 
                     va='center', fontsize=9)
    else:
        ax1 = plt.subplot(2, 2, 1)
        ax1.axis('off')
        ax1.text(0.5, 0.5, 'RF: datos insuficientes', ha='center', va='center', fontsize=12)
    
    if not comparison_gbm_df.empty:
        ax2 = plt.subplot(2, 2, 2)
        colors_bar2 = ['green' if x > 0 else 'red' for x in comparison_gbm_df['Delta_Supervivencia_m']]
        ax2.barh(comparison_gbm_df['Grupo de Riesgo'], comparison_gbm_df['Delta_Supervivencia_m'], 
                 color=colors_bar2, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Delta supervivencia (meses)')
        ax2.set_title('Mediana (GBM)')
        ax2.grid(axis='x', alpha=0.3)
        for i, row in comparison_gbm_df.iterrows():
            sig = '*' if row['p-value'] < 0.05 else ''
            ax2.text(row['Delta_Supervivencia_m'], i, f" {row['Delta_Supervivencia_m']:.1f}m {sig}", 
                     va='center', fontsize=9)
    else:
        ax2 = plt.subplot(2, 2, 2)
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'GBM: datos insuficientes', ha='center', va='center', fontsize=12)
    
    ax3 = plt.subplot(2, 2, 3)
    df_high = df_t34_ml[df_t34_ml['Risk_Group_RF'] == 'Alto Riesgo']
    if len(df_high) > 10:
        for tx, color in [(1, 'purple'), (0, 'gray')]:
            mask = df_high['Combined_Treatment'] == tx
            if mask.sum() > 5:
                label = 'Con Quimio+Radio' if tx == 1 else 'Sin Quimio+Radio'
                kmf.fit(df_high.loc[mask, duration_col],
                        df_high.loc[mask, event_col],
                        label=f'{label} (n={mask.sum()})')
                kmf.plot_survival_function(ax=ax3, color=color, linewidth=2)
        ax3.set_xlabel('Meses')
        ax3.set_ylabel('Prob. Supervivencia')
        ax3.set_title('Alto Riesgo: Kaplan-Meier')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'Datos insuficientes (Alto Riesgo)', ha='center', va='center', fontsize=10)
    
    ax4 = plt.subplot(2, 2, 4)
    df_low = df_t34_ml[df_t34_ml['Risk_Group_RF'] == 'Bajo Riesgo']
    if len(df_low) > 10:
        for tx, color in [(1, 'purple'), (0, 'gray')]:
            mask = df_low['Combined_Treatment'] == tx
            if mask.sum() > 5:
                label = 'Con Quimio+Radio' if tx == 1 else 'Sin Quimio+Radio'
                kmf.fit(df_low.loc[mask, duration_col],
                        df_low.loc[mask, event_col],
                        label=f'{label} (n={mask.sum()})')
                kmf.plot_survival_function(ax=ax4, color=color, linewidth=2)
        ax4.set_xlabel('Meses')
        ax4.set_ylabel('Prob. Supervivencia')
        ax4.set_title('Bajo Riesgo: Kaplan-Meier')
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'Datos insuficientes (Bajo Riesgo)', ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    if not sweets_summary['top'].empty:
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Sweet Spots: Ganancia con Quimio+Radio", fontsize=16, fontweight='bold')

        top_plot = sweets_summary['top'].head(10).iloc[::-1]  # mostrar top 10 en orden ascendente en el eje y
        labels = top_plot.apply(lambda row: f"{row['Segmento']} | {row['Detalle']}", axis=1)

        ax1 = plt.subplot(2, 1, 1)
        ax1.barh(labels, top_plot['Dif Surv (m)'], color='darkslateblue')
        ax1.axvline(x=0, color='black', linewidth=1)
        ax1.set_xlabel('Diferencia en Supervivencia (meses)', fontsize=11)
        ax1.set_title('Top 10 segmentos por ganancia en supervivencia', fontsize=13, pad=15)
        ax1.grid(axis='x', alpha=0.3)

        ax2 = plt.subplot(2, 1, 2)
        colors = ['darkgreen' if val > 0 else 'firebrick' for val in top_plot['Reduccion Mort (%)']]
        ax2.barh(labels, top_plot['Reduccion Mort (%)'], color=colors)
        ax2.axvline(x=0, color='black', linewidth=1)
        ax2.set_xlabel('Reducción de Mortalidad (%)', fontsize=11)
        ax2.set_title('Reducción de mortalidad con tratamiento combinado', fontsize=13, pad=15)
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    if not sweet_spots_df.empty:
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Sweet Spots: Tabla Resumen", fontsize=16, fontweight='bold')

        ax = plt.subplot(1, 1, 1)
        ax.axis('off')

        if len(sweets_summary['psm']) > 0:
            table_source = sweets_summary['psm'].head(10)
            table_display = table_source[['Segmento', 'Detalle', 'N Total', 'N PSM', 'Dif Surv PSM (m)', 'Reduccion Mort PSM (%)', 'p_FDR_PSM', 'Interpretation']].copy()
            table_display = table_display.rename(columns={
                'N Total': 'N total',
                'N PSM': 'N PSM',
                'Dif Surv PSM (m)': 'Delta surv PSM (m)',
                'Reduccion Mort PSM (%)': 'Reduccion mort PSM (%)',
                'p_FDR_PSM': 'p_FDR PSM'
            })
            table_display['Interpretation'] = table_display['Interpretation'].astype(str)
        elif len(sweets_summary['sig']) > 0:
            table_source = sweets_summary['sig'].head(10)
            table_display = table_source[['Segmento', 'Detalle', 'N Total', 'Dif Surv (m)', 'Reduccion Mort (%)', 'p_FDR_naive', 'Interpretation']].copy()
            table_display = table_display.rename(columns={
                'N Total': 'N total',
                'Dif Surv (m)': 'Delta surv (m)',
                'Reduccion Mort (%)': 'Reduccion mort (%)',
                'p_FDR_naive': 'p_FDR'
            })
            table_display['Interpretation'] = table_display['Interpretation'].astype(str)
        else:
            table_source = sweet_spots_df.head(10)
            table_display = table_source[['Segmento', 'Detalle', 'N Total', 'Tx (%)', 'Dif Surv (m)', 'Reduccion Mort (%)', 'p_logrank_naive', 'Interpretation']].copy()
            table_display = table_display.rename(columns={
                'N Total': 'N total',
                'Tx (%)': 'Tx (%)',
                'Dif Surv (m)': 'Delta surv (m)',
                'Reduccion Mort (%)': 'Reduccion mort (%)',
                'p_logrank_naive': 'p log-rank'
            })
            table_display['Interpretation'] = table_display['Interpretation'].astype(str)

        numeric_cols = table_display.select_dtypes(include=[float, int]).columns
        for col in numeric_cols:
            if 'p' in col.lower():
                table_display[col] = table_display[col].map(lambda v: f"{v:.4f}" if pd.notnull(v) else "NA")
            else:
                table_display[col] = table_display[col].map(lambda v: f"{v:.1f}" if pd.notnull(v) else "NA")

        table = ax.table(
            cellText=table_display.values,
            colLabels=table_display.columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        ax.text(
            0.5, 0.05,
            "Se listan los principales segmentos con mayor beneficio estimado.\nResultados requieren confirmación y deben interpretarse como exploratorios.",
            ha='center',
            va='center',
            fontsize=10,
            transform=ax.transAxes
        )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # PÁGINA 9: Resumen final
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Resumen del Análisis', fontsize=16, fontweight='bold')
    
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    
    summary_text = f"""
ANÁLISIS DE SUPERVIVENCIA - CÁNCER DE GLÁNDULA SALIVAL
{'='*80}

COHORTE:
• Total pacientes: {len(df):,}
• Edad media: {df['Age_Median'].mean():.1f} años
• Supervivencia mediana (censurada a 10 años): {df['Survival_months_capped'].median():.1f} meses
• Eventos (muertes): {df['Event_capped'].sum():,} ({df['Event_capped'].mean()*100:.1f}%)

PACIENTES T3/T4:
• Total: {len(df_advanced):,}
• Con Quimio+Radio: {((df_advanced['Chemotherapy_Binary']==1) & (df_advanced['Radiation_Binary']==1)).sum():,}
• Sin tratamiento completo: {len(df_advanced) - ((df_advanced['Chemotherapy_Binary']==1) & (df_advanced['Radiation_Binary']==1)).sum():,}

MODELOS PREDICTIVOS:
• Random Forest: AUC = {results['Random Forest']['auc']:.3f}
• Gradient Boosting: AUC = {results['GBM']['auc']:.3f}
• Red Neuronal: AUC = {results['Neural Network']['auc']:.3f}

GRUPOS DE RIESGO ML (T3/T4):
Los pacientes fueron estratificados en 3 grupos usando características BASALES
(edad, T, N, histología) SIN incluir tratamiento:

"""
    
    for risk_group in ['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo']:
        mask = df_t34_ml['Risk_Group_RF'] == risk_group
        probs = df_t34_ml.loc[mask, 'Predicted_Mortality_RF']
        surv = df_t34_ml.loc[mask, duration_col].median()
        mort = df_t34_ml.loc[mask, event_col].mean() * 100
        summary_text += f"""
• {risk_group}:
  - Prob. mortalidad predicha: {probs.mean():.3f} (rango: {probs.min():.3f}-{probs.max():.3f})
  - Supervivencia mediana: {surv:.1f} meses
  - Mortalidad observada: {mort:.1f}%
"""
    
    summary_text += "\nCOMPARACIÓN QUIMIO+RADIO VS OTROS (RF):\n"
    
    if not comparison_rf_df.empty:
        for _, row in comparison_rf_df.iterrows():
            sig_mark = '✓ p<0.05' if row['p-value'] < 0.05 else '✗ p≥0.05'
            summary_text += f"""
• {row['Grupo de Riesgo']}:
  - Delta supervivencia mediana: {row['Delta_Supervivencia_m']:+.1f} meses
  - Delta mortalidad: {row['Delta_Mortalidad_pct']:+.1f}%
  - p-value: {row['p-value']:.4f} ({sig_mark})
"""
    else:
        summary_text += "• Sin resultados robustos tras el emparejamiento.\n"
    
    summary_text += """

INTERPRETACIÓN:
✓ Estratificación basada en características basales sin incluir tratamiento.
✓ Resultados de tratamiento son exploratorios y dependen del emparejamiento PSM.
✓ Se recomienda validación externa y confirmación con modelos causales adicionales.
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"\n✓ PDF generado: {pdf_filename}")

print("\n" + "="*60)
print("EXPORTANDO RESULTADOS")
print("="*60)

cox_df.to_csv(BASE_DIR / 'Hazard_Ratios.csv', index=False)
print("✓ Hazard_Ratios.csv")

if not subgroup_df.empty:
    subgroup_df.to_csv(BASE_DIR / 'Analisis_Subgrupos.csv', index=False)
    print("✓ Analisis_Subgrupos.csv")

beneficio_df.to_csv(BASE_DIR / 'Beneficio_QuimioRadio_TNM.csv', index=False)
print("✓ Beneficio_QuimioRadio_TNM.csv")

feature_importance.to_csv(BASE_DIR / 'Importancia_Variables.csv', index=False)
print("✓ Importancia_Variables.csv")

if not psm_balance_table.empty:
    psm_balance_table.to_csv(BASE_DIR / 'PSM_Balance_Table.csv', index=False)
    print("✓ PSM_Balance_Table.csv")

if not sweet_spots_df.empty:
    sweet_spots_df.to_csv(BASE_DIR / 'Sweet_Spots_QuimioRadio.csv', index=False)
    print("✓ Sweet_Spots_QuimioRadio.csv")
    sweet_spots_df.to_json(BASE_DIR / 'Sweet_Spots_QuimioRadio.json', orient='records', force_ascii=False)
    print("✓ Sweet_Spots_QuimioRadio.json")

model_metrics = pd.DataFrame({
    'Modelo': list(results.keys()),
    'AUC': [results[m]['auc'] for m in results.keys()]
})
model_metrics.to_csv('Metricas_Modelos.csv', index=False)
print("✓ Metricas_Modelos.csv")

# Exportar análisis de grupos de riesgo ML
if not comparison_rf_df.empty:
    comparison_rf_df.to_csv('Comparacion_RF_Grupo_Riesgo.csv', index=False)
    print("✓ Comparacion_RF_Grupo_Riesgo.csv")
if not comparison_gbm_df.empty:
    comparison_gbm_df.to_csv('Comparacion_GBM_Grupo_Riesgo.csv', index=False)
    print("✓ Comparacion_GBM_Grupo_Riesgo.csv")

if risk_psm_balance_rf:
    rf_balance_export = []
    for risk, table in risk_psm_balance_rf.items():
        temp = table.copy()
        temp.insert(0, 'Grupo_Riesgo_RF', risk)
        rf_balance_export.append(temp)
    pd.concat(rf_balance_export, ignore_index=True).to_csv('PSM_Balance_Riesgo_RF.csv', index=False)
    print("✓ PSM_Balance_Riesgo_RF.csv")

if risk_psm_balance_gbm:
    gbm_balance_export = []
    for risk, table in risk_psm_balance_gbm.items():
        temp = table.copy()
        temp.insert(0, 'Grupo_Riesgo_GBM', risk)
        gbm_balance_export.append(temp)
    pd.concat(gbm_balance_export, ignore_index=True).to_csv('PSM_Balance_Riesgo_GBM.csv', index=False)
    print("✓ PSM_Balance_Riesgo_GBM.csv")

# Exportar características de grupos de riesgo
risk_characteristics = []
for risk_group in ['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo']:
    df_risk = df_t34_ml[df_t34_ml['Risk_Group_RF'] == risk_group]
    probs = df_risk['Predicted_Mortality_RF']
    if df_risk.empty:
        risk_characteristics.append({
            'Grupo': risk_group,
            'N': 0,
            'Prob_Muerte_Min': np.nan,
            'Prob_Muerte_Max': np.nan,
            'Prob_Muerte_Media': np.nan,
            'Edad_Media': np.nan,
            'Supervivencia_Mediana': np.nan,
            'Mortalidad_Observada_%': np.nan
        })
        continue
    
    risk_characteristics.append({
        'Grupo': risk_group,
        'N': len(df_risk),
        'Prob_Muerte_Min': probs.min(),
        'Prob_Muerte_Max': probs.max(),
        'Prob_Muerte_Media': probs.mean(),
        'Edad_Media': df_risk['Age_Median'].mean(),
        'Supervivencia_Mediana': df_risk[duration_col].median(),
        'Mortalidad_Observada_%': df_risk[event_col].mean() * 100
    })

risk_char_df = pd.DataFrame(risk_characteristics)
risk_char_df.to_csv('Caracteristicas_Grupos_Riesgo.csv', index=False)
print("✓ Caracteristicas_Grupos_Riesgo.csv")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO")
print("="*60)
print(f"\nReporte PDF: {pdf_filename}")
print("\nArchivos generados:")
print("  • Reporte PDF con resúmenes, balance PSM, missingness y calibración")
print("  • Tablas CSV: hazard ratios, subgrupos TNM, comparaciones ML y métricas")
print("\nCaracterísticas del análisis:")
print("  • Modelos ML con validación cruzada y curva de calibración")
print("  • Estratificación de riesgo basada en características basales (sin tratamiento)")
print("  • Comparaciones exploratorias de tratamiento tras emparejamiento")
print("  • Cox univariante con hazard ratios e intervalos de confianza")
print("  • Curvas de Kaplan-Meier con censura administrativa a 10 años")
print("\n" + "="*60)
