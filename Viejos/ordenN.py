import pandas as pd
import re

def standardize_t_column(t_value):
    """
    Estandariza los valores de la columna T_Unified.
    
    Mapea valores clínicos (c) y patológicos (p) al formato estándar T.
    Ejemplos: c1 -> T1, p2 -> T2, c4A -> T4A
    """
    if pd.isna(t_value):
        return 'TX'
    
    t_value = str(t_value).strip()
    
    # Manejo de valores especiales
    if t_value == '88':
        return 'T88'
    
    # Si ya está en formato estándar T (T0, T1, T2, etc.)
    if re.match(r'^T', t_value, re.IGNORECASE):
        return t_value.upper()
    
    # Convertir formato clínico (c) a T: c1 -> T1, c4A -> T4A
    if re.match(r'^c', t_value, re.IGNORECASE):
        return 'T' + t_value[1:].upper()
    
    # Convertir formato patológico (p) a T: p1 -> T1, p4B -> T4B
    if re.match(r'^p', t_value, re.IGNORECASE):
        return 'T' + t_value[1:].upper()
    
    # Valor por defecto
    return 'TX'


def standardize_n_column(n_value):
    """
    Estandariza los valores de la columna N_Unified.
    
    Maneja valores especiales y unifica el formato a N0, N1, N2, N3, NX.
    """
    if pd.isna(n_value):
        return 'NX'
    
    n_value = str(n_value).strip()
    
    # Manejo de valores especiales
    if n_value == '88':
        return 'N88'
    
    # Valores que indican información clínica o patológica sin especificar N
    if n_value in ['Clinical', 'Pathologic']:
        return 'NX'
    
    # Si ya está en formato estándar N (N0, N1, N2, etc.)
    if re.match(r'^N[0-3X]', n_value, re.IGNORECASE):
        # Extraer solo el número base (N0, N1, N2, N3) sin subcategorías
        base_match = re.match(r'^(N[0-3X])', n_value, re.IGNORECASE)
        if base_match:
            return base_match.group(1).upper()
    
    # Valor por defecto
    return 'NX'


def process_csv(input_file, output_file):
    """
    Procesa el archivo CSV, estandariza T_Unified y N_Unified (reemplazándolas),
    y elimina filas donde ambas sean TX y NX.
    
    Args:
        input_file (str): Ruta al archivo CSV de entrada
        output_file (str): Ruta al archivo CSV de salida
    """
    # Leer el CSV
    print(f"Leyendo archivo: {input_file}")
    df = pd.read_csv(input_file)
    filas_iniciales = len(df)
    
    # Mostrar valores únicos antes de la estandarización
    print("\n--- Valores únicos ANTES de estandarizar ---")
    print(f"T_Unified: {sorted(df['T_Unified'].unique())}")
    print(f"N_Unified: {sorted(df['N_Unified'].unique())}")
    
    # Aplicar estandarización REEMPLAZANDO las columnas originales
    print("\nAplicando estandarización...")
    df['T_Unified'] = df['T_Unified'].apply(standardize_t_column)
    df['N_Unified'] = df['N_Unified'].apply(standardize_n_column)
    
    # Mostrar valores únicos después de la estandarización
    print("\n--- Valores únicos DESPUÉS de estandarizar ---")
    print(f"T_Unified: {sorted(df['T_Unified'].unique())}")
    print(f"N_Unified: {sorted(df['N_Unified'].unique())}")
    
    # Eliminar filas donde T_Unified = 'TX' Y N_Unified = 'NX'
    print("\nEliminando filas donde T_Unified='TX' y N_Unified='NX'...")
    filas_antes_eliminar = len(df)
    df = df[~((df['T_Unified'] == 'TX') & (df['N_Unified'] == 'NX'))]
    filas_eliminadas = filas_antes_eliminar - len(df)
    
    # Mostrar estadísticas
    print("\n--- Estadísticas finales ---")
    print(f"Total de filas iniciales: {filas_iniciales}")
    print(f"Filas eliminadas (TX y NX): {filas_eliminadas}")
    print(f"Total de filas finales: {len(df)}")
    print(f"\nDistribución T_Unified:")
    print(df['T_Unified'].value_counts().sort_index())
    print(f"\nDistribución N_Unified:")
    print(df['N_Unified'].value_counts().sort_index())
    
    # Guardar el resultado
    df.to_csv(output_file, index=False)
    print(f"\nArchivo guardado: {output_file}")
    
    return df


def main():
    """
    Función principal para ejecutar el script.
    """
    input_file = 'ExportadaSEER_Limpia.csv'
    output_file = 'ExportadaSEER_Estandarizada.csv'
    
    try:
        df = process_csv(input_file, output_file)
        print("\n✓ Estandarización completada exitosamente")
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{input_file}'")
    except Exception as e:
        print(f"Error durante el procesamiento: {str(e)}")


if __name__ == "__main__":
    main()