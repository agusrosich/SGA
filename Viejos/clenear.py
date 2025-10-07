import pandas as pd
import numpy as np
import re

# Cargar el archivo
df = pd.read_csv('ExportadaSEERPurificada.csv')

print(f"Filas originales: {len(df)}")

# 1. CONVERTIR RADIOTERAPIA Y QUIMIOTERAPIA A BINARIO (1/0) - MEJORADO
def convertir_radioterapia_binario(valor):
    """Convierte cualquier tipo de radioterapia a 1, solo No explícito a 0"""
    if pd.isna(valor):
        return 0
    
    valor_str = str(valor).strip().lower()
    
    # Cualquier mención de radiación = 1
    if any(keyword in valor_str for keyword in ['beam', 'radiation', 'radioactive', 'radioisotopes', 
                                                  'implants', 'radiotherapy', 'rt', 'yes']):
        return 1
    
    # Solo si dice explícitamente "no" o "none" = 0
    if valor_str in ['no', 'none', 'no radiation', 'refused']:
        return 0
    
    # Unknown o valores ambiguos = 0
    return 0

def convertir_quimioterapia_binario(valor):
    """Convierte quimioterapia a binario"""
    if pd.isna(valor):
        return 0
    
    valor_str = str(valor).strip().lower()
    
    # Si dice yes o chemotherapy = 1
    if 'yes' in valor_str or 'chemo' in valor_str:
        return 1
    
    # No explícito = 0
    return 0

df['Radiation_Binary'] = df['Radiation recode'].apply(convertir_radioterapia_binario)
df['Chemotherapy_Binary'] = df['Chemotherapy recode (yes, no/unk)'].apply(convertir_quimioterapia_binario)

# 2. EXTRAER EDAD MEDIA DE LOS RANGOS
def extraer_edad_media(rango_edad):
    """Extrae la edad media de rangos como '45-49 years' o '90+ years'"""
    if pd.isna(rango_edad):
        return np.nan
    
    rango = str(rango_edad).strip()
    
    if '90+' in rango or '85+' in rango:
        return 90
    elif '<1' in rango or 'less than 1' in rango.lower():
        return 0.5
    
    patron = r'(\d+)-(\d+)'
    match = re.search(patron, rango)
    
    if match:
        inicio = int(match.group(1))
        fin = int(match.group(2))
        return (inicio + fin) / 2
    
    numeros = re.findall(r'\d+', rango)
    if numeros:
        return float(numeros[0])
    
    return np.nan

df['Age_Median'] = df['Age recode with <1 year olds and 90+'].apply(extraer_edad_media)

# 3. UNIFICAR COLUMNAS DE T (TNM) - MEJORADO
def unificar_tnm(row, columnas_tnm):
    """Unifica múltiples columnas TNM, siendo menos restrictivo con los valores válidos"""
    for col in columnas_tnm:
        if col not in row.index:
            continue
            
        valor = row[col]
        
        # Solo rechazar valores explícitamente inválidos
        if pd.isna(valor):
            continue
            
        valor_str = str(valor).strip()
        
        # Lista más específica de valores inválidos
        valores_invalidos = ['Blank(s)', 'NA', 'N/A', '']
        
        # Si el valor no está vacío y no es explícitamente inválido
        if valor_str and valor_str not in valores_invalidos:
            # Incluso si dice "Unknown" o "Blank" pero sin paréntesis, puede ser válido
            # Solo rechazar los casos muy específicos
            if valor_str.lower() in ['blank', 'unknown', 'not applicable']:
                continue
            return valor_str
    
    return np.nan

columnas_t = [
    'Derived EOD 2018 T Recode (2018+)',
    'Derived AJCC T, 7th ed (2010-2015)',
    'Derived SEER Combined T (2016-2017)'
]

df['T_Unified'] = df.apply(lambda row: unificar_tnm(row, columnas_t), axis=1)

# 4. UNIFICAR COLUMNAS DE N (TNM)
columnas_n = [
    'Derived AJCC N, 7th ed (2010-2015)',
    'Derived EOD 2018 N Recode (2018+)',
    'Derived SEER Combined N Src (2016-2017)'
]

df['N_Unified'] = df.apply(lambda row: unificar_tnm(row, columnas_n), axis=1)

# 5. ESTANDARIZAR TNM - Limpiar formato
def estandarizar_tnm(valor_tnm):
    """Limpia y estandariza valores TNM sin ser demasiado restrictivo"""
    if pd.isna(valor_tnm):
        return np.nan
    
    valor = str(valor_tnm).strip()
    
    # Solo limpiar espacios múltiples
    valor = re.sub(r'\s+', ' ', valor)
    
    return valor

df['T_Unified'] = df['T_Unified'].apply(estandarizar_tnm)
df['N_Unified'] = df['N_Unified'].apply(estandarizar_tnm)

# 6. FILTRAR PACIENTES - MÁS PERMISIVO
# Solo eliminar si AMBOS T y N son nulos (no si solo uno es nulo)
print(f"\nAntes de filtrar TNM: {len(df)}")
print(f"Pacientes con T válido: {df['T_Unified'].notna().sum()}")
print(f"Pacientes con N válido: {df['N_Unified'].notna().sum()}")
print(f"Pacientes con al menos T o N: {(df['T_Unified'].notna() | df['N_Unified'].notna()).sum()}")

# Crear copia con filtro más permisivo
df_clean = df[(df['T_Unified'].notna()) | (df['N_Unified'].notna())].copy()

print(f"Después de filtrar TNM: {len(df_clean)}")
print(f"Pacientes eliminados: {len(df) - len(df_clean)}")

# 7. MAPEO Y UNIFICACIÓN DE HISTOLOGÍA (mismo que antes)
def unificar_histologia(codigo_icd, nombre_broad):
    """
    Unifica y simplifica nombres de histología basado en código ICD-O-3
    """
    
    histology_unified = {
        # CARCINOMAS DE CÉLULAS PEQUEÑAS
        8041: 'Small cell carcinoma',
        8042: 'Small cell carcinoma',
        8043: 'Small cell carcinoma',
        8044: 'Small cell carcinoma',
        8045: 'Small cell carcinoma',
        
        # ADENOCARCINOMAS
        8140: 'Adenocarcinoma',
        8141: 'Adenocarcinoma',
        8143: 'Adenocarcinoma',
        8144: 'Adenocarcinoma',
        8145: 'Adenocarcinoma',
        8147: 'Adenocarcinoma',
        8200: 'Adenocarcinoma',
        8201: 'Adenocarcinoma',
        8210: 'Adenocarcinoma',
        8211: 'Adenocarcinoma',
        8230: 'Adenocarcinoma',
        8260: 'Adenocarcinoma',
        8261: 'Adenocarcinoma',
        8262: 'Adenocarcinoma',
        8263: 'Adenocarcinoma',
        8310: 'Adenocarcinoma',
        8323: 'Adenocarcinoma',
        8380: 'Adenocarcinoma',
        8390: 'Adenocarcinoma',
        8400: 'Adenocarcinoma',
        8401: 'Adenocarcinoma',
        8410: 'Adenocarcinoma',
        8420: 'Adenocarcinoma',
        8440: 'Adenocarcinoma',
        8480: 'Mucinous adenocarcinoma',
        8481: 'Mucinous adenocarcinoma',
        8490: 'Signet ring cell carcinoma',
        8500: 'Adenocarcinoma',
        8502: 'Adenocarcinoma',
        8503: 'Adenocarcinoma',
        8504: 'Adenocarcinoma',
        8507: 'Adenocarcinoma',
        8510: 'Adenocarcinoma',
        8570: 'Adenocarcinoma',
        8571: 'Adenocarcinoma',
        8572: 'Adenocarcinoma',
        8573: 'Adenocarcinoma',
        8574: 'Adenocarcinoma with neuroendocrine differentiation',
        8575: 'Adenocarcinoma',
        
        # CARCINOMA MUCOEPIDERMOIDE
        8430: 'Mucoepidermoid carcinoma',
        
        # ACINAR CELL CARCINOMA
        8550: 'Acinar cell carcinoma',
        8551: 'Acinar cell carcinoma',
        8552: 'Acinar cell carcinoma',
        
        # ADENOSQUAMOUS Y EPITELIAL-MIOEPITELIAL
        8560: 'Adenosquamous carcinoma',
        8561: 'Adenosquamous carcinoma',
        8562: 'Epithelial-myoepithelial carcinoma',
        
        # CARCINOMAS ESCAMOSOS
        8070: 'Squamous cell carcinoma',
        8071: 'Squamous cell carcinoma',
        8072: 'Squamous cell carcinoma',
        8073: 'Squamous cell carcinoma',
        8074: 'Squamous cell carcinoma',
        8075: 'Squamous cell carcinoma',
        8076: 'Squamous cell carcinoma',
        8077: 'Squamous cell carcinoma',
        8078: 'Squamous cell carcinoma',
        8083: 'Squamous cell carcinoma',
        8084: 'Squamous cell carcinoma',
        
        # CARCINOMA ADENOIDE QUÍSTICO
        8200: 'Adenoid cystic carcinoma',
        
        # TUMOR MIXTO / PLEOMÓRFICO
        8940: 'Mixed tumor',
        8941: 'Carcinoma ex pleomorphic adenoma',
        
        # CARCINOMAS INDIFERENCIADOS
        8010: 'Carcinoma NOS',
        8020: 'Undifferentiated carcinoma',
        8021: 'Undifferentiated carcinoma',
        8022: 'Undifferentiated carcinoma',
        8030: 'Undifferentiated carcinoma',
        8031: 'Undifferentiated carcinoma',
        8032: 'Undifferentiated carcinoma',
        8033: 'Undifferentiated carcinoma',
        
        # CARCINOMA MIOEPITELIAL
        8982: 'Myoepithelial carcinoma',
        
        # TUMORES NEUROENDOCRINOS
        8013: 'Neuroendocrine carcinoma',
        8240: 'Neuroendocrine tumor',
        8241: 'Neuroendocrine tumor',
        8242: 'Neuroendocrine tumor',
        8243: 'Neuroendocrine tumor',
        8244: 'Neuroendocrine tumor',
        8245: 'Neuroendocrine tumor',
        8246: 'Neuroendocrine carcinoma',
        8247: 'Neuroendocrine carcinoma',
        8249: 'Neuroendocrine tumor',
        
        # CARCINOMA BASOCELULAR
        8090: 'Basal cell carcinoma',
        8091: 'Basal cell carcinoma',
        8092: 'Basal cell carcinoma',
        8093: 'Basal cell carcinoma',
        8094: 'Basal cell carcinoma',
        8095: 'Basal cell carcinoma',
        8097: 'Basal cell carcinoma',
        8098: 'Basal cell carcinoma',
        
        # CARCINOMA DE CÉLULAS TRANSICIONALES
        8120: 'Transitional cell carcinoma',
        8121: 'Transitional cell carcinoma',
        8122: 'Transitional cell carcinoma',
        8123: 'Transitional cell carcinoma',
        8124: 'Transitional cell carcinoma',
        8130: 'Transitional cell carcinoma',
        8131: 'Transitional cell carcinoma',
        
        # ONCOCITOMA
        8290: 'Oncocytic carcinoma',
        8293: 'Oncocytic carcinoma',
        
        # MELANOMAS
        8720: 'Melanoma',
        8721: 'Melanoma',
        8722: 'Melanoma',
        8723: 'Melanoma',
        8730: 'Melanoma',
        8740: 'Melanoma',
        8741: 'Melanoma',
        8742: 'Melanoma',
        8743: 'Melanoma',
        8744: 'Melanoma',
        8745: 'Melanoma',
        8746: 'Melanoma',
        
        # SARCOMAS
        8800: 'Sarcoma',
        8801: 'Sarcoma',
        8802: 'Sarcoma',
        8803: 'Sarcoma',
        8804: 'Sarcoma',
        8805: 'Sarcoma',
        8806: 'Sarcoma',
        8810: 'Fibrosarcoma',
        8815: 'Fibrosarcoma',
        8890: 'Leiomyosarcoma',
        8891: 'Leiomyosarcoma',
        8910: 'Rhabdomyosarcoma',
        8920: 'Rhabdomyosarcoma',
        
        # LINFOMAS
        9590: 'Lymphoma',
        9591: 'Lymphoma',
        9650: 'Hodgkin lymphoma',
        9680: 'Lymphoma',
        9699: 'Lymphoma',
        
        # LEUCEMIAS
        9733: 'Plasmacytoma',
        9823: 'Leukemia',
        9861: 'Leukemia',
        9863: 'Leukemia',
        9866: 'Leukemia',
        9867: 'Leukemia',
    }
    
    if pd.notna(codigo_icd):
        codigo_int = int(codigo_icd)
        if codigo_int in histology_unified:
            return histology_unified[codigo_int]
    
    if pd.notna(nombre_broad):
        nombre = str(nombre_broad)
        nombre = re.sub(r'\d{4}-\d{4}:\s*', '', nombre)
        nombre = re.sub(r'^\d{4}\s+', '', nombre)
        return nombre.strip()
    
    return 'Unknown'

df_clean['Histology_Unified'] = df_clean.apply(
    lambda row: unificar_histologia(row['Histologic Type ICD-O-3'], 
                                     row['Histology recode - broad groupings']), 
    axis=1
)

# 8. SELECCIONAR COLUMNAS FINALES
columnas_finales = [
    'Age_Median',
    'Sex',
    'Histologic Type ICD-O-3',
    'Histology_Unified',
    'T_Unified',
    'N_Unified',
    'Radiation_Binary',
    'Chemotherapy_Binary',
    'Time from diagnosis to treatment in days recode',
    'Perineural Invasion Recode (2010+)',
    'COD to site recode',
    'Survival months',
    'Residual Tumor Volume Post Cytoreduction Recode (2010+)',
    'Primary Site',
    'Site recode ICD-O-3/WHO 2008 (for SIRs)',
    'Diagnostic Confirmation'
]

df_final = df_clean[columnas_finales].copy()

# 9. GUARDAR ARCHIVO
df_final.to_csv('ExportadaSEER_Limpia.csv', index=False)

print("\n=== RESUMEN DE LIMPIEZA ===")
print(f"Filas finales: {len(df_final)}")
print(f"Columnas finales: {len(df_final.columns)}")
print(f"\nValores únicos de T: {df_final['T_Unified'].nunique()}")
print(f"Valores únicos de N: {df_final['N_Unified'].nunique()}")

print(f"\n=== DISTRIBUCIÓN RADIOTERAPIA ===")
print(f"Total con RT (1): {df_final['Radiation_Binary'].sum()}")
print(f"Total sin RT (0): {(df_final['Radiation_Binary'] == 0).sum()}")
print(f"Porcentaje con RT: {(df_final['Radiation_Binary'].sum() / len(df_final) * 100):.1f}%")

print(f"\n=== DISTRIBUCIÓN QUIMIOTERAPIA ===")
print(f"Total con QT (1): {df_final['Chemotherapy_Binary'].sum()}")
print(f"Total sin QT (0): {(df_final['Chemotherapy_Binary'] == 0).sum()}")
print(f"Porcentaje con QT: {(df_final['Chemotherapy_Binary'].sum() / len(df_final) * 100):.1f}%")

print(f"\nEdad media (estadísticas):")
print(df_final['Age_Median'].describe())

print(f"\n=== HISTOLOGÍAS UNIFICADAS ===")
print(f"Tipos histológicos únicos: {df_final['Histology_Unified'].nunique()}")
print(f"\nDistribución de histologías (top 15):")
print(df_final['Histology_Unified'].value_counts().head(15))

print("\n✓ Archivo guardado como 'ExportadaSEER_Limpia.csv'")