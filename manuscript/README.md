# Documentación de trabajo

Esta carpeta contiene la documentación de respaldo del análisis, no el
manuscrito final (ver `manuscrito_final/`).

## Archivos

- `puntos_a_cumplir.md`: checklist de la revisión metodológica.
- `data_availability_and_emulation_limits.md`: mapeo auditable de variables
  disponibles y no disponibles para la emulación del RTOG 1008.
- `deep-research-report (1).md`: notas de investigación de respaldo.

## Estructura del proyecto

- `01_entrenamiento_seer/`: parte 1, entrenamiento/estimación causal sobre la
  cohorte SEER amplia (T3/T4, N conocido, RT registrada).
  Script: `train_seer_model.py`. Outputs en `01_entrenamiento_seer/outputs/`.
- `02_simulacion_rtog1008/`: parte 2, cohorte de alto riesgo tipo RTOG 1008
  (T3-4 OR N1-3, histología comparable) y simulación repetida del ensayo de
  252 pacientes como predicción pre-resultados. Script: `simulate_rtog1008.py`.
  Outputs en `02_simulacion_rtog1008/outputs/`.
- `common/causal_core.py`: motor estadístico compartido por ambas partes
  (propensity score, IPTW, Cox ponderado, RMST, bootstrap, validación, E-value).
- `manuscrito_final/`: manuscrito final (`manuscript.md` + `assets/`).
- `archive/`: pipelines y resultados superados (pipeline con gradient
  boosting, tablas/figuras del diseño anterior, build previo del manuscrito,
  app web vieja).

## Correr el análisis

Desde la raíz del repositorio:

```bat
scripts\run_part1_seer_training.bat 1000
scripts\run_part2_rtog1008_simulation.bat 1000 1000
```

Parte 1 recibe el número de iteraciones bootstrap. Parte 2 recibe bootstrap y
número de ensayos simulados de 252 pacientes. Los outputs primarios quedan en
`01_entrenamiento_seer/outputs/` y `02_simulacion_rtog1008/outputs/`
respectivamente.

La parte 1 usa la cohorte elegible completa (T3/T4), censura por sobrevida
global, IPTW estabilizado, un modelo de Cox ponderado y ajustado, y RMST a 10
años como ATE. La parte 2 aplica el criterio T3-4 OR N1-3 del RTOG 1008 real
(la rama de margen T1-2N0 no es verificable en SEER) y una restricción de
histología comparable, y construye el ensayo simulado de N=252 como
predicción falsificable frente al resultado real del RTOG 1008. El campo de
tratamiento no establece quimiorradioterapia concurrente en ninguna de las
dos partes.

## Armar el DOCX del manuscrito final

Desde la raíz del repositorio:

```bat
scripts\build_manuscript_docx.bat
```

El archivo generado queda en:

```text
manuscrito_final\output\salivary_gland_causal_survival_manuscript.docx
```
