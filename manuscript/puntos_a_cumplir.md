# Puntos a cumplir

## Estado de la revisión

La revisión metodológica fue implementada en `src/causal_survival_pipeline.py` y
se documentó en `manuscript.md`. Los resultados reproducibles están en
`outputs/causal_survival/`.

Los siguientes requisitos permanecen **no verificables por ausencia de variables
en la extracción fuente**, no por falta de implementación:

- Confirmar cirugía en todos los pacientes.
- Confirmar que la radioterapia fue postoperatoria.
- Excluir directamente enfermedad metastásica al diagnóstico.
- Usar cirugía o radioterapia adyuvante como tiempo cero.
- Verificar la secuencia y concurrencia de cirugía, radioterapia y quimioterapia.
- Identificar cisplatino semanal, agente, dosis o intensidad de quimioterapia.
- Aplicar íntegramente los criterios clinicopatológicos de RTOG 1008.
- Restringir a glándulas salivales mayores.
- Ajustar por márgenes, ENE, PNI y *performance status*.
- Describir fechas/años de diagnóstico.

Estas limitaciones están detalladas en
`data_availability_and_emulation_limits.md`. El trabajo se presenta como
**RTOG 1008-like observacional**, nunca como ensayo aleatorizado ni como
emulación estricta. Todos los demás puntos se incorporaron al pipeline o al
reporte, incluidos censura, RMST, ATE, propensity score, positividad, IPTW,
ajuste de resultado, bootstrap integral, validación de supervivencia,
sensibilidades histológicas, simulaciones secundarias repetidas, balance,
intervalos de confianza, flujo de cohorte y revisión editorial.

## Registro de lo realizado

### Análisis principal

- [x] Se creó un nuevo pipeline de supervivencia censurada en `src/causal_survival_pipeline.py`.
- [x] Se sustituyó la regresión de meses observados por un modelo de Cox que incorpora tiempo y evento.
- [x] Los pacientes vivos en el último seguimiento se trataron como observaciones censuradas.
- [x] Se eliminó el supuesto de que todos los pacientes presentan el evento.
- [x] Se estableció la supervivencia global como endpoint primario.
- [x] Se creó un modelo secundario separado de mortalidad cáncer-específica.
- [x] Se eliminó el uso de Kaplan–Meier sobre tiempos predichos puntuales.
- [x] Se eliminó el log-rank aplicado a valores predichos.
- [x] Se definió el estimando principal como el ATE en la población elegible, expresado como diferencia de RMST a 10 años.
- [x] Se estimaron también diferencias absolutas de supervivencia a 5 y 10 años.
- [x] Se preespecificó como clínicamente relevante una diferencia absoluta de 6 meses de RMST o 5 puntos porcentuales de supervivencia.
- [x] Ambos desenlaces potenciales se estimaron en los mismos 954 pacientes.

### Ajuste causal y balance

- [x] Se modeló la probabilidad de recibir quimioterapia mediante propensity score.
- [x] Se calcularon pesos IPTW estabilizados para el ATE.
- [x] Los pesos extremos se truncaron en los percentiles 1 y 99.
- [x] Se evaluaron positividad y soporte común mediante las distribuciones del propensity score.
- [x] El 86,6% del grupo con quimioterapia y el 84,6% del grupo sin quimioterapia quedaron dentro del soporte común 1–99%.
- [x] Se calcularon diferencias estandarizadas antes y después de IPTW.
- [x] Se incorporó un *love plot* de balance.
- [x] El modelo de resultado ponderado se ajustó además por edad, sexo, T, N e histología como estrategia doblemente ajustada.
- [x] Se documentó el marcado *confounding by indication* de los grupos originales.
- [x] Se añadió un análisis E-value frente a confusión no medida.

### Bootstrap e incertidumbre

- [x] Se ejecutaron 1.000 iteraciones bootstrap a nivel de paciente.
- [x] Las 1.000 iteraciones finalizaron correctamente.
- [x] En cada iteración se recalcularon la muestra analítica, el propensity score, los pesos, el modelo de Cox, las curvas contrafactuales, la RMST y los contrastes.
- [x] Se calcularon intervalos de confianza percentiles del 95%.
- [x] Se dejó de interpretar la proporción de iteraciones positivas como una probabilidad clínica de beneficio.
- [x] Se retiró la dependencia de un valor p para concluir presencia o ausencia de beneficio.

### Validación del modelo

- [x] Se eliminó el *gradient boosting classifier* del análisis principal.
- [x] Se dejó de utilizar su AUC como validación del modelo de supervivencia.
- [x] Se implementó validación cruzada interna de cinco particiones.
- [x] El preprocesamiento y el ajuste se repitieron dentro de cada partición de entrenamiento.
- [x] Se calculó un C-index medio validado de 0,641.
- [x] Se calcularon Brier scores temporales con ponderación IPCW: 0,215 a 5 años y 0,188 a 10 años.
- [x] Se evaluó calibración temporal: error medio de +0,010 a 5 años y +0,013 a 10 años.
- [x] Se corrigió la evaluación de la función de censura en el límite izquierdo del horizonte de 120 meses para evitar pesos IPCW extremos.

### Cohorte y transparencia

- [x] Se incorporó un flujo de selección desde 4.657 registros fuente hasta 954 pacientes analizables.
- [x] Se informó el número excluido en cada criterio implementable.
- [x] Se restringió la cohorte a T3/T4, categoría N conocida, radioterapia registrada y supervivencia no faltante.
- [x] Se reportó la distribución observada: 283 pacientes con quimioterapia registrada y 671 sin ella.
- [x] Se compararon las características basales de los grupos observacionales originales.
- [x] Se informó la mediana de seguimiento por reverse Kaplan–Meier: 86 meses.
- [x] Se generó una tabla de datos faltantes por variable.
- [x] Se resumió el intervalo disponible entre diagnóstico y primer tratamiento.
- [x] Se aclaró que el tiempo cero disponible es el diagnóstico y que no es el origen ideal para una pregunta adyuvante.

### Histologías y sensibilidad

- [x] Se realizó un análisis que excluye carcinoma escamoso.
- [x] Se explicó por qué su inclusión amplía la cohorte pero reduce la comparabilidad con RTOG 1008.
- [x] Se realizó un análisis restringido a histologías identificables comparables con RTOG 1008.
- [x] Se realizaron análisis descriptivos por los principales grupos histológicos con requisitos mínimos de tamaño y exposición.
- [x] Se eliminaron las categorías post hoc de beneficio pequeño, moderado y grande.
- [x] Se retiró “predicted chemotherapy benefit group” de las características basales.
- [x] No se informaron valores p de subgrupos ni se presentaron sus resultados como categorías clínicas validadas.

### Simulación secundaria

- [x] La muestra aleatoria única de 252 pacientes dejó de ser el resultado principal.
- [x] El análisis principal utiliza los 954 pacientes elegibles.
- [x] Se realizaron 1.000 simulaciones secundarias de 252 pacientes con asignación 1:1.
- [x] Se conservaron ambos desenlaces potenciales para cada paciente simulado.
- [x] Se calcularon SMD de edad, sexo, T4 y afectación ganglionar en cada simulación.
- [x] Se mostró que una única simulación puede presentar desequilibrios por azar: el máximo SMD absoluto medio fue 0,185.

### Resultados obtenidos

- [x] RMST ajustada a 10 años sin quimioterapia registrada: 67,1 meses.
- [x] RMST ajustada a 10 años con quimioterapia registrada: 64,8 meses.
- [x] Diferencia ATE de RMST: −2,3 meses.
- [x] Intervalo de confianza bootstrap del 95%: −9,6 a +6,0 meses.
- [x] Diferencia absoluta de supervivencia a 5 años: −2,2 puntos porcentuales.
- [x] Diferencia absoluta de supervivencia a 10 años: −2,2 puntos porcentuales.
- [x] La conclusión se reformuló como predicción observacional del modelo.
- [x] Se aclaró que el intervalo incluye tanto daño clínicamente importante como un posible beneficio cercano al umbral preespecificado.

### Manuscrito, tablas y figuras

- [x] Se reescribió `manuscript/manuscript.md` con el nuevo diseño y resultados.
- [x] Las principales tablas se incorporaron completas y legibles dentro del manuscrito.
- [x] Se eliminaron del manuscrito las referencias a CSV como sustituto de tablas.
- [x] Se eliminaron los gráficos antiguos de Kaplan–Meier sobre tiempos predichos, bootstrap de medianas predichas y distribución de una única simulación.
- [x] Se generaron curvas marginales ajustadas de supervivencia.
- [x] Se generaron figuras de solapamiento del propensity score, balance y distribución bootstrap de RMST.
- [x] Se unificaron tipografía, resolución y nomenclatura de las nuevas figuras.
- [x] Se comprobaron automáticamente los tamaños de cohorte, percentiles bootstrap, rangos de Brier score y cifras utilizadas en el manuscrito.
- [x] Se generaron versiones DOCX, TeX y PDF del manuscrito revisado.
- [x] Se mantuvo como fortaleza central la predicción pre-resultados falsificable frente a RTOG 1008.

### Puntos bloqueados por la extracción fuente

- [!] No se puede confirmar que todos los pacientes hayan sido operados.
- [!] No se puede confirmar que la radioterapia sea postoperatoria.
- [!] No se puede excluir directamente enfermedad metastásica al diagnóstico.
- [!] No puede utilizarse cirugía o inicio de radioterapia adyuvante como tiempo cero.
- [!] No se puede verificar la secuencia ni la concurrencia de cirugía, radioterapia y quimioterapia.
- [!] No se puede identificar cisplatino semanal, otro agente, dosis o intensidad.
- [!] No pueden aplicarse los criterios de RTOG 1008 que dependen de grado, márgenes u otros factores patológicos.
- [!] No puede restringirse la cohorte a tumores de glándulas salivales mayores.
- [!] No se puede ajustar por márgenes, ENE, PNI o *performance status*.
- [!] No se pueden describir fechas o años de diagnóstico porque no están presentes.
- [x] Todas estas imposibilidades se declararon expresamente en el manuscrito y en `data_availability_and_emulation_limits.md`.

## Lista original de requisitos

La lista siguiente muestra el estado final de cada requisito: `[x]` significa
realizado y `[!]` significa no verificable con la extracción disponible.

### 1. Endpoint y análisis de supervivencia

- [x] Reemplazar la regresión convencional de meses de supervivencia por un modelo de supervivencia que maneje censura.
- [x] Eliminar las curvas de Kaplan–Meier construidas sobre tiempos predichos puntuales.
- [x] Eliminar el log-rank aplicado a valores de supervivencia predichos.
- [x] Usar como endpoint principal la supervivencia a tiempo fijo o la RMST a 5–10 años.
- [x] Incorporar la censura en todos los modelos de resultado.
- [x] Eliminar el supuesto de que todos los pacientes presentan el evento.
- [x] Tratar a los pacientes vivos en el último seguimiento como observaciones censuradas.
- [x] Evitar interpretar el seguimiento observado de un paciente vivo como el tiempo real hasta la muerte.
- [x] Informar la mediana de seguimiento mediante reverse Kaplan–Meier.

### 2. Estimando causal y control de confusión

- [x] Definir explícitamente el estimando causal: ATE, ATT, diferencia de RMST o diferencia absoluta de supervivencia.
- [x] Modelar la probabilidad de recibir quimioterapia mediante propensity score.
- [x] Evaluar la positividad y el solapamiento entre los grupos de tratamiento.
- [x] Aplicar IPTW, overlap weighting o un estimador doubly robust.
- [x] Explicar con mayor énfasis el posible *confounding by indication*.
- [x] Reconocer el impacto de covariables no disponibles, como márgenes, ENE, PNI y *performance status*.
- [x] Añadir, si es posible, un análisis de sensibilidad frente a confusión no medida.

### 3. Bootstrap e incertidumbre

- [x] Repetir todo el proceso de modelado dentro de cada iteración del bootstrap.
- [x] Explicar exactamente qué se remuestrea y qué se recalcula en el bootstrap.
- [x] No interpretar la proporción de bootstraps positivos como una probabilidad clínica de beneficio.
- [x] Reportar intervalos de confianza del efecto y no solo el valor p.

### 4. Definición de la cohorte y tiempo cero

- [!] Confirmar que todos los pacientes hayan sido sometidos a cirugía. **No verificable: cirugía no disponible.**
- [!] Restringir la cohorte a radioterapia postoperatoria y no simplemente a cualquier radioterapia. **No verificable: intención postoperatoria no disponible.**
- [!] Excluir a los pacientes con enfermedad metastásica al diagnóstico. **No verificable: estadio M no disponible.**
- [!] Definir un tiempo cero clínicamente coherente, idealmente la cirugía o el inicio de la radioterapia adyuvante. **No verificable: no existen fechas separadas; se utilizó diagnóstico y se declaró la limitación.**
- [!] Verificar la secuencia temporal entre cirugía, radioterapia y quimioterapia. **No verificable: fechas separadas no disponibles.**
- [!] Describir las fechas de diagnóstico y el período de inclusión de la cohorte. **No verificable: año y fecha de diagnóstico no disponibles.**
- [x] Aclarar el manejo de datos faltantes para cada variable.

### 5. Relación con RTOG 1008 y definición del tratamiento

- [x] Aproximar con mayor precisión los criterios de elegibilidad de RTOG 1008 dentro de las variables disponibles.
- [x] Diferenciar claramente una cohorte “RTOG 1008-like” de una emulación estricta del ensayo.
- [x] Evitar denominar el análisis “randomized trial” si el entrenamiento proviene de tratamiento observacional.
- [x] Sustituir “concurrent chemoradiotherapy” por una expresión más prudente si SEER no confirma la concurrencia.
- [x] Aclarar que SEER no permite identificar el uso de cisplatino semanal ni el agente utilizado.
- [x] Mantener como fortaleza central la predicción pre-resultados falsificable frente a RTOG 1008.

### 6. Histologías y análisis de sensibilidad

- [x] Realizar un análisis que excluya el carcinoma escamoso.
- [x] Justificar la inclusión del carcinoma escamoso primario de glándula salival.
- [x] Realizar un análisis limitado a histologías comparables con las incluidas en RTOG 1008.
- [!] Considerar un análisis restringido a tumores de glándulas salivales mayores. **No verificable: el sitio de glándula mayor o menor no está disponible.**
- [x] Presentar análisis de sensibilidad por los principales grupos histológicos.

### 7. Modelos predictivos y validación

- [x] Separar claramente el modelo de mortalidad cáncer-específica del modelo del endpoint primario.
- [x] Explicar cuál es la función del *gradient boosting classifier* dentro del análisis principal: **se retiró porque no era necesario para el estimando causal censurado.**
- [x] No presentar el AUC del clasificador secundario como validación del modelo de supervivencia.
- [x] Evaluar el modelo de supervivencia con C-index, Brier score y calibración temporal.
- [x] Reportar el rendimiento mediante validación cruzada o validación interna correctamente anidada.
- [x] Investigar el sobreajuste sugerido por la diferencia entre el AUC aparente y el AUC con validación cruzada: **se eliminó el clasificador y se sustituyó por validación del modelo de supervivencia.**

### 8. Ensayo simulado y balance entre grupos

- [x] No usar una única muestra aleatoria de 252 pacientes como resultado principal.
- [x] Estimar ambos desenlaces potenciales en la misma población para evitar variabilidad de Monte Carlo innecesaria.
- [x] Mantener la simulación de 252 pacientes solo como análisis secundario ilustrativo.
- [x] Repetir múltiples simulaciones de 252 pacientes si se conserva el formato de ensayo simulado.
- [x] Reportar diferencias estandarizadas para evaluar el balance entre los brazos.
- [x] Revisar el desequilibrio en afectación ganglionar y estadio entre los brazos simulados.
- [x] Reportar la distribución real de tratamientos en los 954 pacientes elegibles.
- [x] Comparar las características basales de los grupos observacionales originales.

### 9. Beneficio terapéutico y análisis de subgrupos

- [x] Retirar “predicted chemotherapy benefit group” de la tabla basal.
- [x] Definir explícitamente los puntos de corte de beneficio pequeño, moderado y grande: **se decidió eliminar estas categorías no validadas.**
- [x] Evitar categorías de beneficio *post hoc* sin validación.
- [x] Preespecificar cualquier análisis de heterogeneidad del efecto terapéutico.
- [x] Limitar los análisis de subgrupos para reducir hallazgos espurios.
- [x] Evitar concluir ausencia de beneficio únicamente por un valor p no significativo.
- [x] Definir previamente qué magnitud se considera clínicamente relevante.
- [x] Reformular las conclusiones como predicciones del modelo y no como evidencia clínica definitiva.

### 10. Transparencia de selección y reporte

- [x] Incorporar un diagrama o tabla de flujo detallado de la selección de pacientes.
- [x] Presentar el número de pacientes excluidos por cada criterio implementable.
- [x] Simplificar las tablas exportadas y eliminar encabezados técnicos fragmentados.
- [x] Sustituir las referencias a archivos CSV por tablas completas y legibles dentro del manuscrito o del suplemento.
- [x] Mejorar la calidad editorial de las figuras y unificar tipografía, resolución y nomenclatura.
- [x] Revisar la consistencia numérica entre el texto, las tablas y las figuras.
