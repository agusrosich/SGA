#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Limpieza y preparación de SEER (glándulas salivales) para ML y supervivencia.

Uso:
  python prep_salivary_ml.py --in ExportadaSEERPurificada.csv --outdir ./salivary_ml_out

Requiere: pandas, numpy, scikit-learn, pyarrow (para parquet), joblib
"""

import argparse
import json
import os
import re
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------- utilidades ---------------------------------- #

DATE_LIKE_PAT = re.compile(r"(date|fecha|dx|diag|death|defunc|fallec|last|contact|seguim|follow)", re.I)
SITE_LIKE_PAT = re.compile(r"(site|topography|topografia|primary|primario|c[0-9]{2})", re.I)
HISTO_LIKE_PAT = re.compile(r"(histo|morpho|morfolog|icd[-_ ]?o)", re.I)
STATUS_LIKE_PAT = re.compile(r"(vital|status|estado)", re.I)
ALIVE_PAT = re.compile(r"(alive|vivo|viva|living|\b0\b)", re.I)
DEAD_PAT = re.compile(r"(dead|deceased|fallec|muert|\b1\b)", re.I)

ID_LIKE_PAT = re.compile(r"(id|mrn|cedula|document|ssn|registro|patient|paciente)", re.I)
LEAKAGE_PAT = re.compile(r"(survival|sobrevida|os_months|os_days|event|muert|fallec)", re.I)

ICDO_SITE_COL_CANDIDATES = [
    "primary_site", "site_recode", "icd_o_3_site", "site", "topography", "topografia"
]

ICDO_HISTO_COL_CANDIDATES = [
    "histology", "histologic_type_icd_o_3", "icd_o_3_histology", "morphology", "morfologia"
]

DATE_COL_ALIASES = {
    "dx": ["date_of_diagnosis", "diagnosis_date", "fecha_diagnostico", "dx_date", "fechadx", "diag_date"],
    "death": ["date_of_death", "death_date", "fecha_fallecimiento", "defuncion", "defunc_date"],
    "last": ["date_of_last_contact", "last_contact", "fecha_ultimo_contacto", "last_followup", "followup_date"]
}

STATUS_COL_CANDIDATES = [
    "vital_status", "vitalstatus", "estado_vital", "status", "vivo_muerto"
]

PRECOMPUTED_OS_CANDIDATES = [
    "overall_survival_months", "os_months", "meses_sobrevida", "sobrevida_meses",
    "overall_survival_days", "os_days", "dias_sobrevida", "sobrevida_dias"
]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9a-zA-Z_]+", "", regex=True)
        .str.lower()
    )
    return df

def parse_date_series(s: pd.Series) -> pd.Series:
    # Intenta múltiples formatos comunes; retorna NaT si falla
    s = s.astype(str).replace({"nan": np.nan, "NaT": np.nan})
    # Remueve subcadenas horarias si existen
    s = s.str.replace(r"T.*$", "", regex=True).str.strip()
    # Corrige variantes dd/mm/aaaa y mm/dd/aaaa según heurística (preferimos día/mes si ambiguo)
    # Usamos infer_datetime_format y dayfirst=True como default razonable para datos LATAM/UE
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)

def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def guess_date_cols(df):
    found = {k: None for k in ("dx", "death", "last")}
    # Probar alias conocidos primero
    for key, aliases in DATE_COL_ALIASES.items():
        col = first_existing(df, aliases)
        if col:
            found[key] = col
    # Si falta alguno, heurística por regex
    for key in found:
        if found[key] is None:
            for col in df.columns:
                if DATE_LIKE_PAT.search(col):
                    # asignar por palabras clave
                    if key == "dx" and re.search(r"(dx|diag|diagnos)", col, re.I):
                        found[key] = col; break
                    if key == "death" and re.search(r"(death|defunc|fallec)", col, re.I):
                        found[key] = col; break
                    if key == "last" and re.search(r"(last|follow|seguim|contact)", col, re.I):
                        found[key] = col; break
    return found

def guess_status_col(df):
    col = first_existing(df, STATUS_COL_CANDIDATES)
    if col: return col
    for c in df.columns:
        if STATUS_LIKE_PAT.search(c): 
            return c
    return None

def is_salivary_row(row, site_col, text_cols):
    # C07–C08 (parótida, glándulas salivales mayores)
    for col in [site_col] + text_cols:
        if col and col in row and pd.notna(row[col]):
            v = str(row[col]).upper()
            if "C07" in v or "C08" in v:
                return True
            if "SALIV" in v:  # "salivary", "salivar", etc.
                return True
            if "PAROT" in v:  # parótida
                return True
            if "SUBMAND" in v or "SUBMAX" in v or "SUBLING" in v:
                return True
    return False

def vital_to_event(val):
    if pd.isna(val): 
        return np.nan
    s = str(val).strip()
    if DEAD_PAT.search(s): 
        return 1
    if ALIVE_PAT.search(s): 
        return 0
    # Intenta normalizaciones típicas
    if s.lower() in {"dead", "deceased", "muerto", "fallecido", "1", "yes", "si"}:
        return 1
    if s.lower() in {"alive", "vivo", "viva", "0", "no"}:
        return 0
    return np.nan

# ----------------------------- pipeline principal -------------------------- #

def main(inp, outdir, random_state=42, test_size=0.2, min_survival_days=1):
    os.makedirs(outdir, exist_ok=True)

    # Lectura robusta a separadores comunes
    try:
        df = pd.read_csv(inp, low_memory=False)
    except Exception:
        df = pd.read_csv(inp, sep=";", low_memory=False)

    df = normalize_columns(df)
    original_cols = df.columns.tolist()

    # Columnas de sitio/histología y texto auxiliar
    site_col = first_existing(df, ICDO_SITE_COL_CANDIDATES)
    histo_col = first_existing(df, ICDO_HISTO_COL_CANDIDATES)
    text_cols = []
    for c in df.columns:
        if SITE_LIKE_PAT.search(c) or HISTO_LIKE_PAT.search(c):
            if c not in (site_col, histo_col):
                text_cols.append(c)

    # Fecha de dx, muerte, último contacto
    date_cols = guess_date_cols(df)
    status_col = guess_status_col(df)

    # Precalculo de OS si existe
    os_pre = first_existing(df, PRECOMPUTED_OS_CANDIDATES)

    # Parseo de fechas
    for key, col in date_cols.items():
        if col and col in df.columns:
            df[col + "_parsed"] = parse_date_series(df[col])
    # Status a evento
    if status_col:
        df["event_from_status"] = df[status_col].apply(vital_to_event)

    # Cálculo de sobrevida
    os_days, os_months, event = None, None, None
    if os_pre:
        # Si la columna está en días, convierto a meses
        if "day" in os_pre or "dia" in os_pre:
            df["overall_survival_days"] = pd.to_numeric(df[os_pre], errors="coerce")
            df["overall_survival_months"] = df["overall_survival_days"] / 30.4375
        else:
            df["overall_survival_months"] = pd.to_numeric(df[os_pre], errors="coerce")
            df["overall_survival_days"] = df["overall_survival_months"] * 30.4375
        os_days = "overall_survival_days"
        os_months = "overall_survival_months"
        # evento: si hay status lo usamos, sino inferimos evento si OS se acompaña de death_date
        if status_col:
            df["event"] = df["event_from_status"]
        elif date_cols["death"] and date_cols["death"] + "_parsed" in df:
            df["event"] = np.where(df[date_cols["death"] + "_parsed"].notna(), 1, 0)
        else:
            df["event"] = np.nan
    else:
        # Calcular desde fechas
        dxp = date_cols["dx"] + "_parsed" if date_cols["dx"] else None
        deathp = date_cols["death"] + "_parsed" if date_cols["death"] else None
        lastp = date_cols["last"] + "_parsed" if date_cols["last"] else None

        if dxp and (deathp or lastp):
            dx = df[dxp]
            death = df[deathp] if deathp in df else pd.Series(pd.NaT, index=df.index)
            last = df[lastp] if lastp in df else pd.Series(pd.NaT, index=df.index)

            # Elegimos punto final: si hay muerte, a muerte; si no, a último contacto
            # evento desde status si existe; de lo contrario, presencia de Fecha de muerte
            event = df["event_from_status"] if "event_from_status" in df else np.where(death.notna(), 1, 0)
            # tiempo (días)
            end = death.where(death.notna(), last)
            os_days_series = (end - dx).dt.days
            df["overall_survival_days"] = os_days_series
            df["overall_survival_months"] = os_days_series / 30.4375
            df["event"] = pd.to_numeric(event, errors="coerce")
            os_days, os_months = "overall_survival_days", "overall_survival_months"
        else:
            raise RuntimeError(
                "No se encontraron columnas suficientes para calcular sobrevida. "
                "Proporcione columnas de fechas (diagnóstico y muerte/último contacto) o una columna de OS."
            )

    # Filtrado por glándulas salivales
    if site_col is None and not text_cols:
        print("AVISO: no se encontró columna clara de sitio; se continuará sin filtro de sitio.")
        df_saliv = df.copy()
    else:
        df_saliv = df[df.apply(lambda r: is_salivary_row(r, site_col, text_cols), axis=1)].copy()
        if df_saliv.empty:
            print("AVISO: el filtro por glándulas salivales no encontró filas; se usará el dataset completo.")
            df_saliv = df.copy()

    # Limpiezas básicas
    # quitar tiempos negativos/0, eventos fuera de {0,1}
    df_saliv = df_saliv[
        (pd.to_numeric(df_saliv[os_days], errors="coerce") >= min_survival_days) &
        (df_saliv["event"].isin([0, 1]))
    ].copy()

    # columnas numéricas/categóricas
    # descartamos columnas con >60% de faltantes y columnas con un solo valor
    miss_frac = df_saliv.isna().mean()
    nunique = df_saliv.nunique(dropna=True)
    drop_cols = set(miss_frac[miss_frac > 0.6].index.tolist()) | set(nunique[nunique <= 1].index.tolist())

    # eliminar identificadores explícitos y potencial leakage de supervivencia
    for c in df_saliv.columns:
        if ID_LIKE_PAT.search(c):
            drop_cols.add(c)
        if LEAKAGE_PAT.search(c) and c not in (os_days, os_months, "event"):
            drop_cols.add(c)
        # evitar usar directamente fechas crudas en ML clásico (podrían fugar info)
        if DATE_LIKE_PAT.search(c) and not c.endswith("_parsed") and c not in (os_days, os_months):
            drop_cols.add(c)

    # conservamos variables fecha parseada para derivar edad al dx si hay nacimiento; si no, las quitamos
    # derivar edad si es posible
    birth_candidates = [c for c in df_saliv.columns if re.search(r"(birth|nacim|dob|fecha_nacimiento)", c, re.I)]
    age_col = first_existing(df_saliv, ["age", "edad", "age_at_diagnosis", "edad_dx"])
    dxp_col = date_cols["dx"] + "_parsed" if date_cols["dx"] and date_cols["dx"] + "_parsed" in df_saliv.columns else None

    if age_col is None and birth_candidates and dxp_col:
        bcol = birth_candidates[0]
        df_saliv[bcol + "_parsed"] = parse_date_series(df_saliv[bcol])
        df_saliv["age"] = ((df_saliv[dxp_col] - df_saliv[bcol + "_parsed"]).dt.days / 365.25).astype(float)
        age_col = "age"
    elif age_col:
        # normalizar a float
        df_saliv[age_col] = pd.to_numeric(df_saliv[age_col], errors="coerce")

    # si dejamos _parsed, que no entren directo al modelo
    for c in df_saliv.columns:
        if c.endswith("_parsed"):
            drop_cols.add(c)

    # columnas finales candidatas a features
    target_cols = [os_months, "event"]
    feature_cols = [c for c in df_saliv.columns if c not in set(target_cols) | drop_cols]

    # separar X, y
    X = df_saliv[feature_cols].copy()
    y_surv = df_saliv[["overall_survival_months", "event"]].copy()

    # etiqueta de clasificación 5 años: y_5y = 1 si murió antes o igual a 60 meses, 0 si censurado o sobrevive >60
    # En clínica muchas veces interesa "mortalidad a 5 años"
    y_5y = ((df_saliv["event"] == 1) & (df_saliv["overall_survival_months"] <= 60)).astype(int)

    # tipado: numéricas vs categóricas
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Preprocesamiento: imputación + escalado para num, imputación + one-hot para cat
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))  # sparse-friendly
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    # Split para ML clásico (clasificación 5y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_5y, test_size=test_size, random_state=random_state, stratify=y_5y if y_5y.nunique() == 2 else None
    )

    # Ajustar solo el preprocesador y guardar objetos listos para usar en modelos
    preprocessor.fit(X_train)

    # Guardados
    # Datos limpios completos
    df_saliv.to_parquet(os.path.join(outdir, "clean_full.parquet"), index=False)
    df_saliv.to_csv(os.path.join(outdir, "clean_full.csv"), index=False)

    # Conjuntos de ML
    dump(preprocessor, os.path.join(outdir, "preprocessor.joblib"))
    X_train.to_parquet(os.path.join(outdir, "X_train_raw.parquet"))
    X_test.to_parquet(os.path.join(outdir, "X_test_raw.parquet"))
    pd.Series(y_train, name="y_5y").to_frame().to_parquet(os.path.join(outdir, "y_train_5y.parquet"))
    pd.Series(y_test, name="y_5y").to_frame().to_parquet(os.path.join(outdir, "y_test_5y.parquet"))

    # Dataset de supervivencia (lifelines / scikit-survival)
    y_surv.to_parquet(os.path.join(outdir, "survival_dataset.parquet"), index=False)

    # Metadatos y reporte breve
    report = {
        "input_file": os.path.abspath(inp),
        "n_rows_input": int(len(df)),
        "n_rows_final": int(len(df_saliv)),
        "columns_original": original_cols,
        "columns_final_features": feature_cols,
        "dropped_columns": sorted(list(drop_cols)),
        "target_columns": target_cols,
        "detected_columns": {
            "site_col": site_col,
            "histo_col": histo_col,
            "status_col": status_col,
            "dates": date_cols,
            "precomputed_os_col": os_pre,
            "age_col_used": age_col
        },
        "class_5y_distribution": {
            "train": {str(k): int((y_train == k).sum()) for k in np.unique(y_train)},
            "test": {str(k): int((y_test == k).sum()) for k in np.unique(y_test)}
        },
        "numeric_feature_count": len(num_cols),
        "categorical_feature_count": len(cat_cols),
        "notes": [
            "event=1: fallecido; event=0: censurado",
            "overall_survival_months usa 30.4375 días/mes",
            f"Se eliminaron filas con OS < {min_survival_days} días o evento no binario."
        ]
    }
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Listo.")
    print(f"Filas de entrada: {len(df)} | filas finales: {len(df_saliv)}")
    print(f"Features finales: {len(feature_cols)} (num: {len(num_cols)}, cat: {len(cat_cols)})")
    print(f"Salidas en: {os.path.abspath(outdir)}")
    if site_col:
        print(f"Filtro de sitio aplicado usando columna: {site_col} (y auxiliares: {', '.join(text_cols[:3])}...)")
    else:
        print("No se aplicó filtro por sitio (no hallado).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Ruta al CSV de entrada (ExportadaSEERPurificada.csv)")
    ap.add_argument("--outdir", default="./salivary_ml_out", help="Directorio de salida")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()
    main(args.inp, args.outdir, random_state=args.random_state, test_size=args.test_size)
