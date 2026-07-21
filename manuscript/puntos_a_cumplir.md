# Puntos a cumplir

## 1. Endpoint y análisis de supervivencia

- [ ] Reemplazar la regresión convencional de meses de supervivencia por un modelo de supervivencia que maneje censura.
- [ ] Eliminar las curvas de Kaplan–Meier construidas sobre tiempos predichos puntuales.
- [ ] Eliminar el log-rank aplicado a valores de supervivencia predichos.
- [ ] Usar como endpoint principal la supervivencia a tiempo fijo o la RMST a 5–10 años.
- [ ] Incorporar la censura en todos los modelos de resultado.
- [ ] Eliminar el supuesto de que todos los pacientes presentan el evento.
- [ ] Tratar a los pacientes vivos en el último seguimiento como observaciones censuradas.
- [ ] Evitar interpretar el seguimiento observado de un paciente vivo como el tiempo real hasta la muerte.
- [ ] Informar la mediana de seguimiento mediante reverse Kaplan–Meier.

## 2. Estimando causal y control de confusión

- [ ] Definir explícitamente el estimando causal: ATE, ATT, diferencia de RMST o diferencia absoluta de supervivencia.
- [ ] Modelar la probabilidad de recibir quimioterapia mediante propensity score.
- [ ] Evaluar la positividad y el solapamiento entre los grupos de tratamiento.
- [ ] Aplicar IPTW, overlap weighting o un estimador doubly robust.
- [ ] Explicar con mayor énfasis el posible *confounding by indication*.
- [ ] Reconocer el impacto de covariables no disponibles, como márgenes, ENE, PNI y *performance status*.
- [ ] Añadir, si es posible, un análisis de sensibilidad frente a confusión no medida.

## 3. Bootstrap e incertidumbre

- [ ] Repetir todo el proceso de modelado dentro de cada iteración del bootstrap.
- [ ] Explicar exactamente qué se remuestrea y qué se recalcula en el bootstrap.
- [ ] No interpretar la proporción de bootstraps positivos como una probabilidad clínica de beneficio.
- [ ] Reportar intervalos de confianza del efecto y no solo el valor p.

## 4. Definición de la cohorte y tiempo cero

- [ ] Confirmar que todos los pacientes hayan sido sometidos a cirugía.
- [ ] Restringir la cohorte a radioterapia postoperatoria y no simplemente a cualquier radioterapia.
- [ ] Excluir a los pacientes con enfermedad metastásica al diagnóstico.
- [ ] Definir un tiempo cero clínicamente coherente, idealmente la cirugía o el inicio de la radioterapia adyuvante.
- [ ] Verificar la secuencia temporal entre cirugía, radioterapia y quimioterapia.
- [ ] Describir las fechas de diagnóstico y el período de inclusión de la cohorte.
- [ ] Aclarar el manejo de datos faltantes para cada variable.

## 5. Relación con RTOG 1008 y definición del tratamiento

- [ ] Aproximar con mayor precisión los criterios de elegibilidad de RTOG 1008.
- [ ] Diferenciar claramente una cohorte “RTOG 1008-like” de una emulación estricta del ensayo.
- [ ] Evitar denominar el análisis “randomized trial” si el entrenamiento proviene de tratamiento observacional.
- [ ] Sustituir “concurrent chemoradiotherapy” por una expresión más prudente si SEER no confirma la concurrencia.
- [ ] Aclarar que SEER no permite identificar el uso de cisplatino semanal ni el agente utilizado.
- [ ] Mantener como fortaleza central la predicción pre-resultados falsificable frente a RTOG 1008.

## 6. Histologías y análisis de sensibilidad

- [ ] Realizar un análisis que excluya el carcinoma escamoso.
- [ ] Justificar la inclusión del carcinoma escamoso primario de glándula salival.
- [ ] Realizar un análisis limitado a histologías comparables con las incluidas en RTOG 1008.
- [ ] Considerar un análisis restringido a tumores de glándulas salivales mayores.
- [ ] Presentar análisis de sensibilidad por los principales grupos histológicos.

## 7. Modelos predictivos y validación

- [ ] Separar claramente el modelo de mortalidad cáncer-específica del modelo del endpoint primario.
- [ ] Explicar cuál es la función del *gradient boosting classifier* dentro del análisis principal.
- [ ] No presentar el AUC del clasificador secundario como validación del modelo de supervivencia.
- [ ] Evaluar el modelo de supervivencia con C-index, Brier score y calibración temporal.
- [ ] Reportar el rendimiento mediante validación cruzada o validación interna correctamente anidada.
- [ ] Investigar el sobreajuste sugerido por la diferencia entre el AUC aparente y el AUC con validación cruzada.

## 8. Ensayo simulado y balance entre grupos

- [ ] No usar una única muestra aleatoria de 252 pacientes como resultado principal.
- [ ] Estimar ambos desenlaces potenciales en la misma población para evitar variabilidad de Monte Carlo innecesaria.
- [ ] Mantener la simulación de 252 pacientes solo como análisis secundario ilustrativo.
- [ ] Repetir múltiples simulaciones de 252 pacientes si se conserva el formato de ensayo simulado.
- [ ] Reportar diferencias estandarizadas para evaluar el balance entre los brazos.
- [ ] Revisar el desequilibrio en afectación ganglionar y estadio entre los brazos simulados.
- [ ] Reportar la distribución real de tratamientos en los 954 pacientes elegibles.
- [ ] Comparar las características basales de los grupos observacionales originales.

## 9. Beneficio terapéutico y análisis de subgrupos

- [ ] Retirar “predicted chemotherapy benefit group” de la tabla basal.
- [ ] Definir explícitamente los puntos de corte de beneficio pequeño, moderado y grande.
- [ ] Evitar categorías de beneficio *post hoc* sin validación.
- [ ] Preespecificar cualquier análisis de heterogeneidad del efecto terapéutico.
- [ ] Limitar los análisis de subgrupos para reducir hallazgos espurios.
- [ ] Evitar concluir ausencia de beneficio únicamente por un valor p no significativo.
- [ ] Definir previamente qué magnitud se considera clínicamente relevante.
- [ ] Reformular las conclusiones como predicciones del modelo y no como evidencia clínica definitiva.

## 10. Transparencia de selección y reporte

- [ ] Incorporar un diagrama de flujo detallado de la selección de pacientes.
- [ ] Presentar el número de pacientes excluidos por cada criterio.
- [ ] Simplificar las tablas exportadas y eliminar encabezados técnicos fragmentados.
- [ ] Sustituir las referencias a archivos CSV por tablas completas y legibles dentro del manuscrito o del suplemento.
- [ ] Mejorar la calidad editorial de las figuras y unificar tipografía, resolución y nomenclatura.
- [ ] Revisar la consistencia numérica entre el texto, las tablas y las figuras.
