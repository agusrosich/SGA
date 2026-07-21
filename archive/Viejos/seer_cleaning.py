import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Carga los datos del archivo CSV"""
    print("Cargando datos...")
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def clean_column_names(df):
    """Limpia los nombres de las columnas"""
    df.columns = df.columns.str.strip()
    return df

def handle_missing_values(df):
    """Identifica y maneja valores perdidos"""
    print("\n=== ANÁLISIS DE VALORES PERDIDOS ===")
    
    # Mapeo de valores que indican "desconocido" o "no aplica"
    unknown_values = ['Unknown', 'Blank(s)', 'N/A', 'Not applicable', 
                      'Unknown/not stated', 'Not available']
    
    # Reemplazar valores desconocidos por NaN
    df = df.replace(unknown_values, np.nan)
    
    # Mostrar porcentaje de valores perdidos
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_info = pd.DataFrame({
        'Columna': missing_pct.index,
        'Porcentaje_Perdido': missing_pct.values
    })
    print(missing_info[missing_info['Porcentaje_Perdido'] > 0].to_string(index=False))
    
    return df

def transform_age(df):
    """Transforma la columna de edad a valores numéricos"""
    print("\n=== TRANSFORMANDO EDAD ===")
    
    age_col = 'Age recode with <1 year olds and 90+'
    if age_col in df.columns:
        # Extraer el valor numérico del rango de edad
        df['Age_Numeric'] = df[age_col].str.extract('(\d+)').astype(float)
        
        # Para los casos de "90+ years", asignar 90
        df.loc[df[age_col].str.contains('90\+', na=False), 'Age_Numeric'] = 90
        
        print(f"Edad transformada. Rango: {df['Age_Numeric'].min()} - {df['Age_Numeric'].max()}")
    
    return df

def transform_sex(df):
    """Codifica la variable sexo"""
    if 'Sex' in df.columns:
        df['Sex_Encoded'] = df['Sex'].map({'Male': 1, 'Female': 0})
    return df

def transform_icd_histology(df):
    """Transforma los códigos ICD-O-3 de histología"""
    print("\n=== TRANSFORMANDO CÓDIGOS ICD-O-3 ===")
    
    if 'Histologic Type ICD-O-3' in df.columns:
        # Mantener el código ICD como está (ya es numérico)
        df['ICD_Histology'] = df['Histologic Type ICD-O-3']
        
        # Crear categorías generales basadas en rangos de ICD-O-3
        # Carcinomas: 8000-8999
        # Adenomas y adenocarcinomas: 8140-8389
        # Carcinoma mucoepidermoide: 8430
        # Carcinoma adenoide quístico: 8200
        
        def categorize_histology(code):
            if pd.isna(code):
                return 'Unknown'
            code = int(code)
            if code == 8200:
                return 'Adenoid_Cystic'
            elif code == 8430:
                return 'Mucoepidermoid'
            elif 8140 <= code <= 8389:
                return 'Adenocarcinoma'
            elif 8500 <= code <= 8580:
                return 'Ductal_Lobular'
            elif 8000 <= code <= 8005:
                return 'Neoplasm_Malignant'
            elif 8010 <= code <= 8084:
                return 'Epithelial_Squamous'
            else:
                return 'Other_Carcinoma'
        
        df['Histology_Category'] = df['ICD_Histology'].apply(categorize_histology)
        
        print("Distribución de categorías histológicas:")
        print(df['Histology_Category'].value_counts())
    
    return df

def transform_primary_site(df):
    """Transforma los códigos de sitio primario"""
    if 'Primary Site' in df.columns:
        df['Primary_Site_Code'] = df['Primary Site']
        
        # Crear categorías de sitio
        def categorize_site(code):
            if pd.isna(code):
                return 'Unknown'
            code = int(code)
            # Códigos C07-C08 son glándulas salivales
            if code == 70:
                return 'Parotid'
            elif code == 80:
                return 'Submandibular'
            elif code == 81:
                return 'Sublingual'
            elif code == 89:
                return 'Other_Salivary'
            else:
                return 'Other'
        
        df['Site_Category'] = df['Primary_Site_Code'].apply(categorize_site)
    
    return df

def transform_staging(df):
    """Transforma las variables de estadificación TNM"""
    print("\n=== TRANSFORMANDO ESTADIFICACIÓN TNM ===")
    
    # Columnas de T stage
    t_columns = [
        'Derived EOD 2018 T Recode (2018+)',
        'Derived AJCC T, 7th ed (2010-2015)',
        'Derived SEER Combined T (2016-2017)'
    ]
    
    # Columnas de N stage
    n_columns = [
        'Derived EOD 2018 N Recode (2018+)',
        'Derived AJCC N, 7th ed (2010-2015)',
        'Derived SEER Combined N Src (2016-2017)'
    ]
    
    # Unificar T stage
    for col in t_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    df['T_Stage_Unified'] = df[t_columns].fillna('').agg(''.join, axis=1)
    df['T_Stage_Unified'] = df['T_Stage_Unified'].replace('', np.nan)
    
    # Extraer valor numérico de T (T1=1, T2=2, etc.)
    df['T_Numeric'] = df['T_Stage_Unified'].str.extract('T(\d)').astype(float)
    
    # Unificar N stage
    for col in n_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    df['N_Stage_Unified'] = df[n_columns].fillna('').agg(''.join, axis=1)
    df['N_Stage_Unified'] = df['N_Stage_Unified'].replace('', np.nan)
    
    # Extraer valor numérico de N
    df['N_Numeric'] = df['N_Stage_Unified'].str.extract('N(\d)').astype(float)
    
    return df

def transform_treatment(df):
    """Transforma las variables de tratamiento"""
    print("\n=== TRANSFORMANDO TRATAMIENTO ===")
    
    # Radioterapia
    if 'Radiation recode' in df.columns:
        df['Radiation_Binary'] = df['Radiation recode'].apply(
            lambda x: 1 if pd.notna(x) and 'Yes' in str(x) else 0
        )
    
    # Quimioterapia
    if 'Chemotherapy recode (yes, no/unk)' in df.columns:
        df['Chemotherapy_Binary'] = df['Chemotherapy recode (yes, no/unk)'].apply(
            lambda x: 1 if pd.notna(x) and 'Yes' in str(x) else 0
        )
    
    # Tiempo desde diagnóstico a tratamiento
    if 'Time from diagnosis to treatment in days recode' in df.columns:
        df['Treatment_Days'] = pd.to_numeric(
            df['Time from diagnosis to treatment in days recode'], 
            errors='coerce'
        )
    
    return df

def transform_survival(df):
    """Transforma la variable de supervivencia (objetivo)"""
    print("\n=== TRANSFORMANDO SUPERVIVENCIA ===")
    
    if 'Survival months' in df.columns:
        df['Survival_Months'] = pd.to_numeric(
            df['Survival months'], 
            errors='coerce'
        )
        
        # Crear categorías de supervivencia
        df['Survival_Category'] = pd.cut(
            df['Survival_Months'],
            bins=[0, 12, 36, 60, float('inf')],
            labels=['<1_year', '1-3_years', '3-5_years', '>5_years']
        )
        
        print(f"Supervivencia - Media: {df['Survival_Months'].mean():.2f} meses")
        print(f"Supervivencia - Mediana: {df['Survival_Months'].median():.2f} meses")
        print("\nDistribución por categoría:")
        print(df['Survival_Category'].value_counts())
    
    return df

def encode_categorical_variables(df):
    """Codifica variables categóricas restantes"""
    print("\n=== CODIFICANDO VARIABLES CATEGÓRICAS ===")
    
    categorical_cols = [
        'Histology recode - broad groupings',
        'Perineural Invasion Recode (2010+)',
        'COD to site recode',
        'Residual Tumor Volume Post Cytoreduction Recode (2010+)',
        'Site recode ICD-O-3/WHO 2008 (for SIRs)',
        'Diagnostic Confirmation'
    ]
    
    le = LabelEncoder()
    
    for col in categorical_cols:
        if col in df.columns:
            # Llenar NaN con categoría 'Unknown'
            df[col] = df[col].fillna('Unknown')
            
            # Codificar
            col_encoded = col.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('+', '')
            df[f'{col_encoded}_Encoded'] = le.fit_transform(df[col].astype(str))
    
    return df

def select_ml_features(df):
    """Selecciona las características finales para machine learning"""
    print("\n=== SELECCIONANDO CARACTERÍSTICAS PARA ML ===")
    
    # Características numéricas
    numeric_features = [
        'Age_Numeric',
        'Sex_Encoded',
        'ICD_Histology',
        'Primary_Site_Code',
        'T_Numeric',
        'N_Numeric',
        'Radiation_Binary',
        'Chemotherapy_Binary',
        'Treatment_Days'
    ]
    
    # Características categóricas codificadas
    categorical_encoded = [col for col in df.columns if col.endswith('_Encoded')]
    
    # Variable objetivo
    target = ['Survival_Months', 'Survival_Category']
    
    # Combinar todas las características
    all_features = numeric_features + categorical_encoded + target
    
    # Filtrar solo las columnas que existen
    available_features = [col for col in all_features if col in df.columns]
    
    df_ml = df[available_features].copy()
    
    print(f"Total de características seleccionadas: {len(available_features) - 2}")
    print(f"Variables objetivo: {target}")
    
    return df_ml

def clean_final_dataset(df_ml):
    """Limpieza final del dataset"""
    print("\n=== LIMPIEZA FINAL ===")
    
    # Eliminar filas donde la variable objetivo es nula
    initial_rows = len(df_ml)
    df_ml = df_ml.dropna(subset=['Survival_Months'])
    rows_removed = initial_rows - len(df_ml)
    
    print(f"Filas eliminadas por falta de supervivencia: {rows_removed}")
    print(f"Filas finales: {len(df_ml)}")
    
    # Reporte de valores perdidos en características
    missing_summary = df_ml.isnull().sum()
    if missing_summary.sum() > 0:
        print("\nValores perdidos restantes por columna:")
        print(missing_summary[missing_summary > 0])
    
    return df_ml

def save_cleaned_data(df_ml, output_path='SEER_cleaned_ml_ready.csv'):
    """Guarda el dataset limpio"""
    df_ml.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ Dataset limpio guardado en: {output_path}")
    
    # Guardar también información sobre las características
    feature_info = pd.DataFrame({
        'Feature': df_ml.columns,
        'Type': df_ml.dtypes,
        'Non_Null_Count': df_ml.count(),
        'Null_Count': df_ml.isnull().sum()
    })
    
    info_path = output_path.replace('.csv', '_feature_info.csv')
    feature_info.to_csv(info_path, index=False)
    print(f"✓ Información de características guardada en: {info_path}")

def main():
    """Función principal que ejecuta todo el pipeline de limpieza"""
    print("="*60)
    print("LIMPIEZA DE DATOS SEER - CÁNCER DE GLÁNDULA SALIVAL")
    print("="*60)
    
    # 1. Cargar datos
    input_file = 'ExportadaSEERPurificada.csv'
    df = load_data(input_file)
    
    # 2. Limpiar nombres de columnas
    df = clean_column_names(df)
    
    # 3. Manejar valores perdidos
    df = handle_missing_values(df)
    
    # 4. Transformaciones específicas
    df = transform_age(df)
    df = transform_sex(df)
    df = transform_icd_histology(df)
    df = transform_primary_site(df)
    df = transform_staging(df)
    df = transform_treatment(df)
    df = transform_survival(df)
    
    # 5. Codificar variables categóricas
    df = encode_categorical_variables(df)
    
    # 6. Seleccionar características para ML
    df_ml = select_ml_features(df)
    
    # 7. Limpieza final
    df_ml = clean_final_dataset(df_ml)
    
    # 8. Guardar dataset limpio
    save_cleaned_data(df_ml)
    
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"Dimensiones finales: {df_ml.shape}")
    print(f"\nPrimeras filas del dataset limpio:")
    print(df_ml.head())
    print("\nEstadísticas descriptivas:")
    print(df_ml.describe())
    
    return df_ml

if __name__ == "__main__":
    df_cleaned = main()