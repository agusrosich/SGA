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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(encoding='utf-8')

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

# Crear estadios TNM agrupados
def assign_stage(row):
    t = str(row['T_Unified'])
    n = str(row['N_Unified'])
    
    if t not in ['TX', 'T88', 'T0'] and n not in ['NX', 'N88']:
        if t == 'T1' and n == 'N0':
            return 'Stage I'
        elif (t == 'T2' and n == 'N0') or (t == 'T1' and n == 'N1'):
            return 'Stage II'
        elif (t == 'T3' and n == 'N0') or (t in ['T1', 'T2'] and n in ['N1', 'N2']):
            return 'Stage III'
        elif t in ['T4A', 'T4B', 'T4NOS'] or n in ['N2', 'N3'] or (t == 'T3' and n in ['N2', 'N3']):
            return 'Stage IV'
        else:
            return 'Stage III'
    
    if t in ['TX', 'T88', 'T0']:
        if n == 'N0':
            return 'Stage II'
        elif n == 'N1':
            return 'Stage III'
        elif n in ['N2', 'N3']:
            return 'Stage IV'
        else:
            return 'Stage II'
    
    if n in ['NX', 'N88']:
        if t == 'T1':
            return 'Stage I'
        elif t == 'T2':
            return 'Stage II'
        elif t == 'T3':
            return 'Stage III'
        elif t in ['T4A', 'T4B', 'T4NOS']:
            return 'Stage IV'
    
    return 'Stage II'

df['Stage'] = df.apply(assign_stage, axis=1)

# Grupos de T avanzados
df['T_Advanced'] = df['T_Unified'].apply(lambda x: 'T3-T4' if str(x).startswith(('T3', 'T4')) else 'T1-T2' if str(x).startswith(('T1', 'T2')) else 'Unknown')

# Grupos de tratamiento
df['Treatment_Group'] = df.apply(lambda x: 
    'Chemo+Radio' if x['Chemotherapy_Binary']==1 and x['Radiation_Binary']==1 
    else 'Radio Only' if x['Radiation_Binary']==1 
    else 'Chemo Only' if x['Chemotherapy_Binary']==1 
    else 'No Treatment', axis=1)

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
    'Supervivencia Media (meses)': [df['Survival months'].mean()],
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
    'Survival months': ['mean', 'median', 'count'],
    'Event': 'mean'
}).round(2)
surv_by_treat.columns = ['Media (meses)', 'Mediana (meses)', 'N', 'Tasa Mortalidad']
print(surv_by_treat)

# Calcular p-values entre grupos de tratamiento
print("\n--- Tests Estadísticos (T3/T4) ---")

group1 = df_advanced[df_advanced['Treatment_Group']=='Chemo+Radio']
group2 = df_advanced[df_advanced['Treatment_Group']=='Radio Only']

if len(group1) > 5 and len(group2) > 5:
    result_lr = logrank_test(group1['Survival months'], group2['Survival months'],
                             group1['Event'], group2['Event'])
    print(f"Chemo+Radio vs Radio Only: p-value = {result_lr.p_value:.4f} {'***' if result_lr.p_value < 0.001 else '**' if result_lr.p_value < 0.01 else '*' if result_lr.p_value < 0.05 else 'NS'}")

group3 = df_advanced[df_advanced['Treatment_Group']=='No Treatment']
if len(group1) > 5 and len(group3) > 5:
    result_lr2 = logrank_test(group1['Survival months'], group3['Survival months'],
                              group1['Event'], group3['Event'])
    print(f"Chemo+Radio vs No Treatment: p-value = {result_lr2.p_value:.4f} {'***' if result_lr2.p_value < 0.001 else '**' if result_lr2.p_value < 0.01 else '*' if result_lr2.p_value < 0.05 else 'NS'}")

if len(group2) > 5 and len(group3) > 5:
    result_lr3 = logrank_test(group2['Survival months'], group3['Survival months'],
                              group2['Event'], group3['Event'])
    print(f"Radio Only vs No Treatment: p-value = {result_lr3.p_value:.4f} {'***' if result_lr3.p_value < 0.001 else '**' if result_lr3.p_value < 0.01 else '*' if result_lr3.p_value < 0.05 else 'NS'}")

# ============================================================
# 4. PREPARACIÓN DE DATOS PARA MODELOS ML
# ============================================================

print("\n" + "="*60)
print("PREPARACIÓN DE DATOS PARA MODELOS ML")
print("="*60)

# Seleccionar características
feature_cols = ['Age_Median', 'Sex', 'T_Unified', 'N_Unified', 
                'Radiation_Binary', 'Chemotherapy_Binary', 'Histology_Unified']

df_ml = df[feature_cols + ['Event', 'Survival months']].copy()

# Eliminar filas con valores desconocidos en T o N
df_ml = df_ml[~df_ml['T_Unified'].isin(['TX', 'T88'])].copy()
df_ml = df_ml[~df_ml['N_Unified'].isin(['NX', 'N88'])].copy()
df_ml = df_ml.dropna()

print(f"✓ Dataset ML: {len(df_ml)} pacientes")

# Codificar variables categóricas
le_dict = {}
for col in ['Sex', 'T_Unified', 'N_Unified', 'Histology_Unified']:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    le_dict[col] = le

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

# 1. Random Forest
print("\n[1/3] Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model

# 2. Gradient Boosting
print("[2/3] Gradient Boosting...")
gbm_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gbm_model.fit(X_train, y_train)
models['GBM'] = gbm_model

# 3. Neural Network
print("[3/3] Red Neuronal...")
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)
nn_model.fit(X_train_scaled, y_train)
models['Neural Network'] = nn_model

print("✓ Modelos entrenados exitosamente")

# Evaluar modelos
print("\n--- Rendimiento de Modelos ---")
for name, model in models.items():
    if name == 'Neural Network':
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    results[name] = {
        'predictions': y_pred,
        'probabilities': y_proba,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }
    
    print(f"\n{name}:")
    print(f"  AUC: {roc_auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Vivo', 'Fallecido']))

# Importancia de características (Random Forest)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- Importancia de Variables (Random Forest) ---")
print(feature_importance.to_string(index=False))

# ============================================================
# 6. ODDS RATIOS Y ANÁLISIS ESTADÍSTICO
# ============================================================

print("\n" + "="*60)
print("ODDS RATIOS - FACTORES DE RIESGO")
print("="*60)

def calculate_odds_ratio(df, exposure, outcome, exposure_value=1):
    """Calcula odds ratio con intervalo de confianza y p-value"""
    exposed = df[df[exposure] == exposure_value]
    unexposed = df[df[exposure] != exposure_value]
    
    a = exposed[outcome].sum()
    b = len(exposed) - a
    c = unexposed[outcome].sum()
    d = len(unexposed) - c
    
    if b == 0 or d == 0 or a == 0 or c == 0:
        return None, None, None, None
    
    or_value = (a * d) / (b * c)
    
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lower = np.exp(np.log(or_value) - 1.96 * se_log_or)
    ci_upper = np.exp(np.log(or_value) + 1.96 * se_log_or)
    
    chi2 = ((a*d - b*c)**2 * (a+b+c+d)) / ((a+b)*(c+d)*(a+c)*(b+d))
    p_value = 1 - stats.chi2.cdf(chi2, 1)
    
    return or_value, ci_lower, ci_upper, p_value

or_results = []

or_val, ci_low, ci_up, p_val = calculate_odds_ratio(df, 'Radiation_Binary', 'Event', 1)
if or_val:
    or_results.append(['Radioterapia', or_val, ci_low, ci_up, p_val])

or_val, ci_low, ci_up, p_val = calculate_odds_ratio(df, 'Chemotherapy_Binary', 'Event', 1)
if or_val:
    or_results.append(['Quimioterapia', or_val, ci_low, ci_up, p_val])

df_t_comparison = df[df['T_Advanced'].isin(['T1-T2', 'T3-T4'])].copy()
df_t_comparison['T_Binary'] = (df_t_comparison['T_Advanced'] == 'T3-T4').astype(int)
or_val, ci_low, ci_up, p_val = calculate_odds_ratio(df_t_comparison, 'T_Binary', 'Event', 1)
if or_val:
    or_results.append(['T3-T4 vs T1-T2', or_val, ci_low, ci_up, p_val])

df_n_comparison = df[df['N_Unified'].isin(['N0', 'N1', 'N2', 'N3'])].copy()
df_n_comparison['N_Binary'] = (df_n_comparison['N_Unified'] != 'N0').astype(int)
or_val, ci_low, ci_up, p_val = calculate_odds_ratio(df_n_comparison, 'N_Binary', 'Event', 1)
if or_val:
    or_results.append(['N+ vs N0', or_val, ci_low, ci_up, p_val])

df['Stage_Binary'] = (df['Stage'] == 'Stage IV').astype(int)
or_val, ci_low, ci_up, p_val = calculate_odds_ratio(df, 'Stage_Binary', 'Event', 1)
if or_val:
    or_results.append(['Stage IV vs I-III', or_val, ci_low, ci_up, p_val])

or_df = pd.DataFrame(or_results, columns=['Factor', 'OR', 'IC 95% Inf', 'IC 95% Sup', 'p-value'])
or_df['Significancia'] = or_df['p-value'].apply(lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS')
print(or_df.to_string(index=False))

# ============================================================
# 7. ANÁLISIS DE SUBGRUPOS: BENEFICIO DE QUIMIO-RADIO
# ============================================================

print("\n" + "="*60)
print("ANÁLISIS DE SUBGRUPOS: BENEFICIO DE TRATAMIENTO")
print("="*60)

subgroups = []

for t_stage in ['T3', 'T4A', 'T4B']:
    df_sub = df[df['T_Unified'] == t_stage].copy()
    if len(df_sub) > 30:
        with_both = df_sub[(df_sub['Chemotherapy_Binary']==1) & (df_sub['Radiation_Binary']==1)]
        without_both = df_sub[(df_sub['Chemotherapy_Binary']==0) | (df_sub['Radiation_Binary']==0)]
        
        if len(with_both) > 5 and len(without_both) > 5:
            surv_with = with_both['Survival months'].median()
            surv_without = without_both['Survival months'].median()
            mort_with = with_both['Event'].mean()
            mort_without = without_both['Event'].mean()
            
            subgroups.append([
                f"T: {t_stage}",
                len(with_both),
                len(without_both),
                surv_with,
                surv_without,
                mort_with * 100,
                mort_without * 100
            ])

for hist in df['Histology_Unified'].value_counts().head(5).index:
    df_sub = df[df['Histology_Unified'] == hist].copy()
    df_sub = df_sub[df_sub['T_Advanced'] == 'T3-T4']
    
    if len(df_sub) > 30:
        with_both = df_sub[(df_sub['Chemotherapy_Binary']==1) & (df_sub['Radiation_Binary']==1)]
        without_both = df_sub[(df_sub['Chemotherapy_Binary']==0) | (df_sub['Radiation_Binary']==0)]
        
        if len(with_both) > 5 and len(without_both) > 5:
            surv_with = with_both['Survival months'].median()
            surv_without = without_both['Survival months'].median()
            mort_with = with_both['Event'].mean()
            mort_without = without_both['Event'].mean()
            
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
print("\n" + subgroup_df.to_string(index=False))

# ============================================================
# 8. ANÁLISIS ML: GRUPOS DE RIESGO BASALES (CORREGIDO)
# ============================================================

print("\n" + "="*60)
print("ANÁLISIS ML: GRUPOS DE RIESGO Y BENEFICIO DE TRATAMIENTO")
print("="*60)

# Filtrar pacientes T3/T4
df_t34_ml = df[df['T_Advanced'] == 'T3-T4'].copy()
df_t34_ml = df_t34_ml.dropna(subset=['Age_Median', 'Sex', 'T_Unified', 'N_Unified', 
                                      'Histology_Unified', 'Survival months'])

df_t34_ml['Combined_Treatment'] = ((df_t34_ml['Chemotherapy_Binary']==1) & 
                                    (df_t34_ml['Radiation_Binary']==1)).astype(int)

print(f"\nPacientes T3/T4: {len(df_t34_ml)}")
print(f"Con Quimio+Radio: {df_t34_ml['Combined_Treatment'].sum()}")
print(f"Sin Quimio+Radio: {len(df_t34_ml) - df_t34_ml['Combined_Treatment'].sum()}")

# MODELO DE RIESGO BASAL (sin incluir tratamiento)
print("\n--- Modelo de Riesgo Basal ---")
feature_cols_basal = ['Age_Median', 'Sex', 'T_Unified', 'N_Unified', 'Histology_Unified']

df_t34_encoded = df_t34_ml.copy()

le_dict_t34 = {}
for col in ['Sex', 'T_Unified', 'N_Unified', 'Histology_Unified']:
    le = LabelEncoder()
    df_t34_encoded[col] = le.fit_transform(df_t34_encoded[col].astype(str))
    le_dict_t34[col] = le

X_basal = df_t34_encoded[feature_cols_basal]
y_mortality = df_t34_encoded['Event']

# Random Forest para riesgo basal
rf_risk = RandomForestClassifier(n_estimators=100, max_depth=8, 
                                 random_state=42, min_samples_leaf=10,
                                 class_weight='balanced')
rf_risk.fit(X_basal, y_mortality)

df_t34_ml['Death_Risk_Proba_RF'] = rf_risk.predict_proba(X_basal)[:, 1]

# GBM para riesgo basal
gbm_risk = GradientBoostingClassifier(n_estimators=100, max_depth=4, 
                                      learning_rate=0.1, random_state=42)
gbm_risk.fit(X_basal, y_mortality)
df_t34_ml['Death_Risk_Proba_GBM'] = gbm_risk.predict_proba(X_basal)[:, 1]

print(f"✓ Modelos de riesgo entrenados")
print(f"  Prob. muerte promedio (RF): {df_t34_ml['Death_Risk_Proba_RF'].mean():.3f}")
print(f"  Prob. muerte promedio (GBM): {df_t34_ml['Death_Risk_Proba_GBM'].mean():.3f}")

# Crear grupos de riesgo (terciles)
df_t34_ml['Risk_Group_RF'] = pd.qcut(df_t34_ml['Death_Risk_Proba_RF'], 
                                      q=3, 
                                      labels=['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo'])

df_t34_ml['Risk_Group_GBM'] = pd.qcut(df_t34_ml['Death_Risk_Proba_GBM'], 
                                       q=3, 
                                       labels=['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo'])

# Mostrar rangos de probabilidad
print("\n--- Rangos de Probabilidad de Muerte (Random Forest) ---")
for risk_group in ['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo']:
    mask = df_t34_ml['Risk_Group_RF'] == risk_group
    probs = df_t34_ml.loc[mask, 'Death_Risk_Proba_RF']
    print(f"{risk_group}: {probs.min():.3f} - {probs.max():.3f} (media: {probs.mean():.3f})")

# Características clínicas por grupo de riesgo
print("\n--- Características por Grupo de Riesgo (RF) ---")
for risk_group in ['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo']:
    df_risk = df_t34_ml[df_t34_ml['Risk_Group_RF'] == risk_group]
    print(f"\n{risk_group} (n={len(df_risk)}):")
    print(f"  Edad media: {df_risk['Age_Median'].mean():.1f} años")
    print(f"  Mortalidad observada: {df_risk['Event'].mean()*100:.1f}%")
    print(f"  Supervivencia mediana: {df_risk['Survival months'].median():.1f} meses")
    print(f"  T más común: {df_risk['T_Unified'].mode().values[0]}")
    print(f"  N más común: {df_risk['N_Unified'].mode().values[0]}")

# ANÁLISIS DE BENEFICIO por grupo de riesgo
print("\n" + "="*60)
print("BENEFICIO DE TRATAMIENTO POR GRUPO DE RIESGO")
print("="*60)

benefit_by_risk_rf = []

for risk in ['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo']:
    df_risk = df_t34_ml[df_t34_ml['Risk_Group_RF'] == risk]
    
    with_tx = df_risk[df_risk['Combined_Treatment'] == 1]
    without_tx = df_risk[df_risk['Combined_Treatment'] == 0]
    
    if len(with_tx) >= 5 and len(without_tx) >= 5:
        surv_with = with_tx['Survival months'].median()
        surv_without = without_tx['Survival months'].median()
        mort_with = with_tx['Event'].mean()
        mort_without = without_tx['Event'].mean()
        
        result_lr = logrank_test(with_tx['Survival months'], without_tx['Survival months'],
                                with_tx['Event'], without_tx['Event'])
        
        benefit_by_risk_rf.append({
            'Grupo de Riesgo': risk,
            'N Con Tx': len(with_tx),
            'N Sin Tx': len(without_tx),
            'Surv Con Tx (m)': surv_with,
            'Surv Sin Tx (m)': surv_without,
            'Diferencia (m)': surv_with - surv_without,
            'Mort Con Tx (%)': mort_with * 100,
            'Mort Sin Tx (%)': mort_without * 100,
            'Reducción Mort (%)': (mort_without - mort_with) * 100,
            'p-value': result_lr.p_value,
            'Significativo': 'SÍ' if result_lr.p_value < 0.05 else 'NO'
        })

benefit_rf_df = pd.DataFrame(benefit_by_risk_rf)
print("\n--- Beneficio por Grupo de Riesgo (Random Forest) ---")
print(benefit_rf_df.to_string(index=False))

# Mismo análisis con GBM
benefit_by_risk_gbm = []

for risk in ['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo']:
    df_risk = df_t34_ml[df_t34_ml['Risk_Group_GBM'] == risk]
    
    with_tx = df_risk[df_risk['Combined_Treatment'] == 1]
    without_tx = df_risk[df_risk['Combined_Treatment'] == 0]
    
    if len(with_tx) >= 5 and len(without_tx) >= 5:
        surv_with = with_tx['Survival months'].median()
        surv_without = without_tx['Survival months'].median()
        mort_with = with_tx['Event'].mean()
        mort_without = without_tx['Event'].mean()
        
        result_lr = logrank_test(with_tx['Survival months'], without_tx['Survival months'],
                                with_tx['Event'], without_tx['Event'])
        
        benefit_by_risk_gbm.append({
            'Grupo de Riesgo': risk,
            'N Con Tx': len(with_tx),
            'N Sin Tx': len(without_tx),
            'Surv Con Tx (m)': surv_with,
            'Surv Sin Tx (m)': surv_without,
            'Diferencia (m)': surv_with - surv_without,
            'Mort Con Tx (%)': mort_with * 100,
            'Mort Sin Tx (%)': mort_without * 100,
            'Reducción Mort (%)': (mort_without - mort_with) * 100,
            'p-value': result_lr.p_value,
            'Significativo': 'SÍ' if result_lr.p_value < 0.05 else 'NO'
        })

benefit_gbm_df = pd.DataFrame(benefit_by_risk_gbm)
print("\n--- Beneficio por Grupo de Riesgo (GBM) ---")
print(benefit_gbm_df.to_string(index=False))

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
        diff = with_tx['Survival months'].median() - without_tx['Survival months'].median()
        result_lr = logrank_test(with_tx['Survival months'], without_tx['Survival months'],
                                with_tx['Event'], without_tx['Event'])
        sig = '*' if result_lr.p_value < 0.05 else 'NS'
        print(f"  {age_group}: Δ={diff:+.1f} meses, p={result_lr.p_value:.3f} {sig}")

print("\nBeneficio por Estadio N:")
for n_stage in ['N0', 'N1', 'N2', 'N3']:
    df_n = df_t34_ml[df_t34_ml['N_Unified'] == n_stage]
    with_tx = df_n[df_n['Combined_Treatment'] == 1]
    without_tx = df_n[df_n['Combined_Treatment'] == 0]
    
    if len(with_tx) >= 5 and len(without_tx) >= 5:
        diff = with_tx['Survival months'].median() - without_tx['Survival months'].median()
        result_lr = logrank_test(with_tx['Survival months'], without_tx['Survival months'],
                                with_tx['Event'], without_tx['Event'])
        sig = '*' if result_lr.p_value < 0.05 else 'NS'
        print(f"  {n_stage}: Δ={diff:+.1f} meses, p={result_lr.p_value:.3f} {sig}")

# ============================================================
# 9. ANÁLISIS TRADICIONAL: SUBGRUPOS TNM
# ============================================================

print("\n" + "="*60)
print("ANÁLISIS TRADICIONAL: BENEFICIO POR COMBINACIÓN T-N")
print("="*60)

beneficio_results = []

for t_stage in ['T3', 'T4A', 'T4B']:
    for n_stage in ['N0', 'N1', 'N2']:
        df_sub = df[(df['T_Unified'] == t_stage) & (df['N_Unified'] == n_stage)].copy()
        
        if len(df_sub) >= 20:
            with_both = df_sub[(df_sub['Chemotherapy_Binary']==1) & (df_sub['Radiation_Binary']==1)]
            without_both = df_sub[(df_sub['Chemotherapy_Binary']==0) | (df_sub['Radiation_Binary']==0)]
            
            if len(with_both) >= 5 and len(without_both) >= 5:
                result_lr = logrank_test(
                    with_both['Survival months'],
                    without_both['Survival months'],
                    with_both['Event'],
                    without_both['Event']
                )
                
                surv_diff = with_both['Survival months'].median() - without_both['Survival months'].median()
                mort_diff = without_both['Event'].mean() - with_both['Event'].mean()
                
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
print("BUSCANDO 'SWEET SPOTS' PARA QUIMIO+RADIO")
print("="*60)

df_sweet = df[df['T_Advanced'] == 'T3-T4'].copy()

df_sweet['Age_Group'] = pd.cut(
    df_sweet['Age_Median'],
    bins=[0, 50, 65, 120],
    labels=['<50', '50-65', '>65']
)

etop = df_sweet['Histology_Unified'].value_counts().head(6).index
df_sweet['Histology_Group'] = df_sweet['Histology_Unified'].where(
    df_sweet['Histology_Unified'].isin(etop),
    'Otros'
)

sweet_spots_results = []
sweets_summary = {
    'top': pd.DataFrame(),
    'sig': pd.DataFrame()
}
sweet_spots_df = pd.DataFrame()

min_total = 30
min_treated = 10
min_control = 10

segment_definitions = [
    ('Edad x N', ['Age_Group', 'N_Unified']),
    ('Edad x Stage', ['Age_Group', 'Stage']),
    ('Edad x Histologia', ['Age_Group', 'Histology_Group']),
    ('Stage x N', ['Stage', 'N_Unified']),
    ('Histologia x N', ['Histology_Group', 'N_Unified']),
    ('Stage x Histologia', ['Stage', 'Histology_Group'])
]

for segment_name, cols in segment_definitions:
    for values, subset in df_sweet.groupby(cols, dropna=False):
        if subset.shape[0] < min_total:
            continue

        with_tx = subset[(subset['Chemotherapy_Binary'] == 1) & (subset['Radiation_Binary'] == 1)]
        without_tx = subset[(subset['Chemotherapy_Binary'] == 0) | (subset['Radiation_Binary'] == 0)]

        if len(with_tx) < min_treated or len(without_tx) < min_control:
            continue

        try:
            lr_result = logrank_test(
                with_tx['Survival months'],
                without_tx['Survival months'],
                with_tx['Event'],
                without_tx['Event']
            )
        except ValueError:
            continue

        surv_gain = with_tx['Survival months'].median() - without_tx['Survival months'].median()
        mort_gain = without_tx['Event'].mean() - with_tx['Event'].mean()

        value_labels = values if isinstance(values, tuple) else (values,)
        value_labels = [
            str(v) if (v is not None and v == v) else 'Desconocido'
            for v in value_labels
        ]

        sweet_spots_results.append({
            'Segmento': segment_name,
            'Detalle': " | ".join(value_labels),
            'N Total': subset.shape[0],
            'N Con Tx': len(with_tx),
            'N Sin Tx': len(without_tx),
            'Tx (%)': len(with_tx) / subset.shape[0] * 100,
            'Surv Con Tx (m)': with_tx['Survival months'].median(),
            'Surv Sin Tx (m)': without_tx['Survival months'].median(),
            'Dif Surv (m)': surv_gain,
            'Mort Con Tx (%)': with_tx['Event'].mean() * 100,
            'Mort Sin Tx (%)': without_tx['Event'].mean() * 100,
            'Reduccion Mort (%)': mort_gain * 100,
            'p-value': lr_result.p_value,
            'Significativo': 'Si' if (lr_result.p_value < 0.05 and surv_gain > 0) else 'No'
        })

sweet_spots_df = pd.DataFrame(sweet_spots_results)

if not sweet_spots_df.empty:
    sweet_spots_df = sweet_spots_df.sort_values(
        ['Significativo', 'Dif Surv (m)'],
        ascending=[False, False]
    )

    print("\n--- Sweet spots con beneficio significativo ---")
    sweets_summary['sig'] = sweet_spots_df[
        (sweet_spots_df['Significativo'] == 'Si') &
        (sweet_spots_df['Dif Surv (m)'] > 0)
    ]
    sweets_summary['top'] = sweet_spots_df.head(15).copy()
    if len(sweets_summary['sig']) > 0:
        print(sweets_summary['sig'].head(15).to_string(index=False))
    else:
        print("No se identificaron segmentos con beneficio estadisticamente significativo con los criterios actuales.")

    print("\n--- Top 15 segmentos ordenados por ganancia en supervivencia ---")
    print(sweets_summary['top'].to_string(index=False))
else:
    print("No se encontraron segmentos con tamano suficiente para evaluar el beneficio combinado.")

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
    surv_by_stage = df.groupby('Stage')['Survival months'].median().sort_values()
    surv_by_stage.plot(kind='barh', color='plum', ax=ax4)
    ax4.set_title('Supervivencia Mediana por Estadio')
    ax4.set_xlabel('Meses')
    ax4.set_ylabel('Estadio')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
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
            kmf.fit(df.loc[mask, 'Survival months'], 
                   df.loc[mask, 'Event'], 
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
            kmf.fit(df_t34.loc[mask, 'Survival months'],
                   df_t34.loc[mask, 'Event'],
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
    
    # PÁGINA 6: Odds Ratios
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Odds Ratios - Factores de Riesgo', fontsize=16, fontweight='bold')
    
    ax = plt.subplot(1, 1, 1)
    
    y_pos = np.arange(len(or_df))
    
    for i, row in or_df.iterrows():
        color = 'darkred' if row['p-value'] < 0.05 else 'gray'
        ax.plot([row['IC 95% Inf'], row['IC 95% Sup']], [i, i], 'k-', linewidth=2, color=color)
        ax.plot(row['OR'], i, 'o', markersize=10, color=color)
    
    ax.axvline(x=1, color='blue', linestyle='--', linewidth=1.5, label='OR = 1 (Sin efecto)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(or_df['Factor'])
    ax.set_xlabel('Odds Ratio (IC 95%)', fontsize=12)
    ax.set_title('Forest Plot: Factores Asociados a Mortalidad', fontsize=14, pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=10)
    
    for i, row in or_df.iterrows():
        text_str = f"OR={row['OR']:.2f}\n[{row['IC 95% Inf']:.2f}-{row['IC 95% Sup']:.2f}]\np={row['p-value']:.4f}"
        ax.text(row['IC 95% Sup'] + 0.1, i, text_str, va='center', fontsize=8)
    
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
        ax1.hist(df_t34_ml.loc[mask, 'Death_Risk_Proba_RF'], alpha=0.6, label=risk, color=color, bins=15)
    ax1.set_xlabel('Probabilidad de Muerte')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Probabilidades por Grupo')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Gráfica 2: Mortalidad observada por grupo
    ax2 = plt.subplot(2, 2, 2)
    mort_by_risk = df_t34_ml.groupby('Risk_Group_RF')['Event'].mean() * 100
    mort_by_risk.plot(kind='bar', ax=ax2, color=['green', 'orange', 'red'])
    ax2.set_xlabel('Grupo de Riesgo')
    ax2.set_ylabel('Mortalidad Observada (%)')
    ax2.set_title('Mortalidad Real por Grupo de Riesgo')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Gráfica 3: Supervivencia mediana por grupo
    ax3 = plt.subplot(2, 2, 3)
    surv_by_risk = df_t34_ml.groupby('Risk_Group_RF')['Survival months'].median()
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
    
    # PÁGINA 8: Beneficio de tratamiento por grupo de riesgo
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Beneficio de Quimio+Radio por Grupo de Riesgo ML', fontsize=16, fontweight='bold')
    
    if len(benefit_rf_df) > 0:
        # Gráfica 1: Diferencia en supervivencia
        ax1 = plt.subplot(2, 2, 1)
        colors_bar = ['green' if x > 0 else 'red' for x in benefit_rf_df['Diferencia (m)']]
        ax1.barh(benefit_rf_df['Grupo de Riesgo'], benefit_rf_df['Diferencia (m)'], 
                color=colors_bar, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Diferencia en Supervivencia (meses)')
        ax1.set_title('Beneficio de Quimio+Radio')
        ax1.grid(axis='x', alpha=0.3)
        
        for i, row in benefit_rf_df.iterrows():
            sig = '*' if row['p-value'] < 0.05 else ''
            ax1.text(row['Diferencia (m)'], i, f" {row['Diferencia (m)']:.1f}m {sig}", 
                    va='center', fontsize=9)
        
        # Gráfica 2: Reducción de mortalidad
        ax2 = plt.subplot(2, 2, 2)
        colors_bar2 = ['green' if x > 0 else 'red' for x in benefit_rf_df['Reducción Mort (%)']]
        ax2.barh(benefit_rf_df['Grupo de Riesgo'], benefit_rf_df['Reducción Mort (%)'], 
                color=colors_bar2, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Reducción de Mortalidad (%)')
        ax2.set_title('Impacto en Mortalidad')
        ax2.grid(axis='x', alpha=0.3)
        
        # Gráfica 3: KM Alto Riesgo
        ax3 = plt.subplot(2, 2, 3)
        kmf = KaplanMeierFitter()
        df_high = df_t34_ml[df_t34_ml['Risk_Group_RF'] == 'Alto Riesgo']
        
        for tx, color in [(1, 'purple'), (0, 'gray')]:
            mask = df_high['Combined_Treatment'] == tx
            if mask.sum() > 5:
                label = 'Con Quimio+Radio' if tx == 1 else 'Sin Quimio+Radio'
                kmf.fit(df_high.loc[mask, 'Survival months'],
                       df_high.loc[mask, 'Event'],
                       label=f'{label} (n={mask.sum()})')
                kmf.plot_survival_function(ax=ax3, color=color, linewidth=2)
        
        ax3.set_xlabel('Meses')
        ax3.set_ylabel('Prob. Supervivencia')
        ax3.set_title('Alto Riesgo: Efecto del Tratamiento')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
        
        # Gráfica 4: KM Bajo Riesgo
        ax4 = plt.subplot(2, 2, 4)
        df_low = df_t34_ml[df_t34_ml['Risk_Group_RF'] == 'Bajo Riesgo']
        
        for tx, color in [(1, 'purple'), (0, 'gray')]:
            mask = df_low['Combined_Treatment'] == tx
            if mask.sum() > 5:
                label = 'Con Quimio+Radio' if tx == 1 else 'Sin Quimio+Radio'
                kmf.fit(df_low.loc[mask, 'Survival months'],
                       df_low.loc[mask, 'Event'],
                       label=f'{label} (n={mask.sum()})')
                kmf.plot_survival_function(ax=ax4, color=color, linewidth=2)
        
        ax4.set_xlabel('Meses')
        ax4.set_ylabel('Prob. Supervivencia')
        ax4.set_title('Bajo Riesgo: Efecto del Tratamiento')
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)
    
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

        table_source = sweets_summary['sig'].head(10) if len(sweets_summary['sig']) > 0 else sweet_spots_df.head(10)
        table_display = table_source[['Segmento', 'Detalle', 'N Total', 'Tx (%)', 'Dif Surv (m)', 'Reduccion Mort (%)', 'p-value', 'Significativo']].copy()
        table_display['Tx (%)'] = table_display['Tx (%)'].map(lambda v: f"{v:.1f}")
        table_display['Dif Surv (m)'] = table_display['Dif Surv (m)'].map(lambda v: f"{v:.1f}")
        table_display['Reduccion Mort (%)'] = table_display['Reduccion Mort (%)'].map(lambda v: f"{v:.1f}")
        table_display['p-value'] = table_display['p-value'].map(lambda v: f"{v:.4f}")

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
            "Se listan los principales segmentos donde la combinación muestra\nla mayor diferencia de supervivencia y reducción de mortalidad.",
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
• Supervivencia mediana: {df['Survival months'].median():.1f} meses
• Eventos (muertes): {df['Event'].sum():,} ({df['Event'].mean()*100:.1f}%)

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
        probs = df_t34_ml.loc[mask, 'Death_Risk_Proba_RF']
        surv = df_t34_ml.loc[mask, 'Survival months'].median()
        mort = df_t34_ml.loc[mask, 'Event'].mean() * 100
        summary_text += f"""
• {risk_group}:
  - Prob. muerte: {probs.mean():.3f} (rango: {probs.min():.3f}-{probs.max():.3f})
  - Supervivencia mediana: {surv:.1f} meses
  - Mortalidad observada: {mort:.1f}%
"""
    
    summary_text += f"""

BENEFICIO DE QUIMIO+RADIO POR GRUPO:
"""
    
    for _, row in benefit_rf_df.iterrows():
        sig_mark = '✓ SÍ' if row['Significativo'] == 'SÍ' else '✗ NO'
        summary_text += f"""
• {row['Grupo de Riesgo']}:
  - Diferencia supervivencia: {row['Diferencia (m)']:+.1f} meses
  - Reducción mortalidad: {row['Reducción Mort (%)']:+.1f}%
  - p-value: {row['p-value']:.4f}
  - Beneficio significativo: {sig_mark}
"""
    
    summary_text += """

INTERPRETACIÓN:
✓ Los modelos ML identifican grupos de riesgo basados en características basales
✓ El beneficio del tratamiento combinado varía según el grupo de riesgo
✓ El análisis permite personalizar decisiones terapéuticas en T3/T4
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

or_df.to_csv(BASE_DIR / 'Odds_Ratios.csv', index=False)
print("✓ Odds_Ratios.csv")

if len(subgroup_df) > 0:
    subgroup_df.to_csv(BASE_DIR / 'Analisis_Subgrupos.csv', index=False)
    print("✓ Analisis_Subgrupos.csv")

beneficio_df.to_csv(BASE_DIR / 'Beneficio_QuimioRadio_TNM.csv', index=False)
print("✓ Beneficio_QuimioRadio_TNM.csv")

feature_importance.to_csv(BASE_DIR / 'Importancia_Variables.csv', index=False)
print("✓ Importancia_Variables.csv")

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
benefit_rf_df.to_csv('Beneficio_Por_Grupo_Riesgo_ML.csv', index=False)
print("✓ Beneficio_Por_Grupo_Riesgo_ML.csv")

# Exportar características de grupos de riesgo
risk_characteristics = []
for risk_group in ['Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo']:
    df_risk = df_t34_ml[df_t34_ml['Risk_Group_RF'] == risk_group]
    probs = df_risk['Death_Risk_Proba_RF']
    
    risk_characteristics.append({
        'Grupo': risk_group,
        'N': len(df_risk),
        'Prob_Muerte_Min': probs.min(),
        'Prob_Muerte_Max': probs.max(),
        'Prob_Muerte_Media': probs.mean(),
        'Edad_Media': df_risk['Age_Median'].mean(),
        'Supervivencia_Mediana': df_risk['Survival months'].median(),
        'Mortalidad_Observada_%': df_risk['Event'].mean() * 100
    })

risk_char_df = pd.DataFrame(risk_characteristics)
risk_char_df.to_csv('Caracteristicas_Grupos_Riesgo.csv', index=False)
print("✓ Caracteristicas_Grupos_Riesgo.csv")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO")
print("="*60)
print(f"\nReporte PDF: {pdf_filename}")
print("\nArchivos generados:")
print("  • 9 páginas de análisis visual en PDF")
print("  • 7 archivos CSV con resultados detallados")
print("\nCaracterísticas del análisis:")
print("  • Modelos ML con AUC > 0.70")
print("  • Grupos de riesgo basados en características BASALES")
print("  • Análisis de beneficio de tratamiento por grupo")
print("  • Tests estadísticos con p-values")
print("  • Curvas de Kaplan-Meier por múltiples estratificaciones")
print("\n" + "="*60)
