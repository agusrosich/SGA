# Modelización predictiva de resultados en cánceres de glándulas salivales mediante aprendizaje automático: validación prospectiva simulada con el ensayo RTOG 1008

Autores: Federico Lorenzo; Agustin Rosich; Jesica Lell; Sergio Aguiar; Valentina Ferreira; Karina Ochandorena; Eduardo Larrinaga; Natalia Gadea; Nicolas Larragueta; Aldo Quarneti

## Resumen

### Objetivo

No se ha establecido si la adición de quimioterapia a la radioterapia mejora la supervivencia en el cáncer de glándulas salivales de alto riesgo. Desarrollamos un modelo falsable, formulado antes de conocer los resultados, que predice el efecto promedio de la quimioterapia sobre la supervivencia en la enfermedad T3/T4 avanzada, para su comparación futura con RTOG 1008.

### Métodos y materiales

Analizamos una cohorte estandarizada derivada del programa Surveillance, Epidemiology, and End Results (SEER) de tumores T3/T4 con categoría N conocida y radioterapia (RT) registrada. La supervivencia global se modelizó teniendo en cuenta la censura. El estimando preespecificado fue el efecto promedio del tratamiento (ATE), expresado como la diferencia en el tiempo medio de supervivencia restringido (RMST) hasta 120 meses entre RT + quimioterapia y solo RT. Se utilizaron la ponderación por puntuación de propensión y un modelo de Cox ajustado por covariables para abordar las diferencias medidas en la selección del tratamiento. Ambos resultados potenciales se estandarizaron a la misma población, y el análisis completo se repitió en 1.000 muestras bootstrap a nivel de paciente. El rendimiento interno se evaluó mediante validación cruzada con cinco particiones.

### Resultados

De 4.657 registros de origen, 954 cumplieron los criterios de elegibilidad; 283 recibieron RT + quimioterapia y 671, solo RT. El RMST ajustado a 10 años fue de 64,8 meses con RT + quimioterapia y de 67,1 meses con solo RT. El ATE fue de −2,3 meses (IC del 95% por bootstrap, −9,6 a +6,0; P=0,580). Las diferencias absolutas ajustadas de supervivencia fueron de −2,2 puntos porcentuales tanto a 5 como a 10 años. La mediana de seguimiento, estimada por el método de Kaplan–Meier inverso, fue de 86 meses. El índice C medio con validación cruzada fue de 0,641; las puntuaciones de Brier ponderadas por la probabilidad inversa de censura fueron de 0,215 a 5 años y de 0,188 a 10 años.

### Conclusiones

El modelo no predijo un beneficio promedio en la supervivencia global al añadir quimioterapia a la radioterapia en la enfermedad T3/T4 avanzada. La estimación concuerda con la evidencia retrospectiva contraria a la intensificación rutinaria con quimioterapia y aporta una predicción previa a los resultados para su contrastación prospectiva frente a RTOG 1008.

## Introducción

La cirugía seguida de radioterapia posoperatoria adaptada al riesgo es la estrategia locorregional establecida para las neoplasias malignas de glándulas salivales de alto riesgo [1–9], mientras que el valor incremental de la quimioterapia sigue sin resolverse [1,2,10]. Los primeros informes institucionales sugirieron una posible ventaja con la intensificación del tratamiento [11,12], pero comparaciones retrospectivas más amplias no han demostrado un beneficio de supervivencia consistente [13–21]. RTOG 1008 se diseñó para resolver esta cuestión comparando la radioterapia posoperatoria sola con la radioterapia más cisplatino semanal en la enfermedad de alto riesgo resecada [22–24].

El objetivo de este estudio fue predecir si la quimioterapia produce una ganancia promedio de supervivencia clínicamente relevante cuando se añade a la radioterapia en el cáncer de glándulas salivales T3/T4 avanzado. Utilizamos métodos de supervivencia causal que tienen en cuenta la censura y conservamos el tamaño muestral de RTOG 1008 únicamente como referencia para una simulación secundaria.

## Métodos y materiales

### Fuente de datos y criterios de elegibilidad aplicables

La fuente fue un extracto estandarizado del programa SEER del National Cancer Institute que contenía 4.657 registros y 11 variables: edad agrupada, sexo, histología CIE-O-3, histología armonizada, categorías T y N armonizadas, indicadores binarios de radioterapia y quimioterapia, demora desde el diagnóstico hasta el primer tratamiento, causa de muerte recodificada y meses de supervivencia. Se exigieron de forma secuencial enfermedad T3/T4, categoría N conocida, radioterapia registrada y tiempo de supervivencia disponible.

Este fue un análisis observacional similar a RTOG 1008. El diagnóstico fue el origen temporal disponible, y el tratamiento se clasificó a partir de los indicadores de radioterapia y quimioterapia del registro. Dado que los datos de SEER están desidentificados y son de acceso público, este análisis estuvo exento de revisión por un comité de ética institucional.

### Tratamiento, desenlace y estimando

Los grupos de tratamiento se denominaron solo RT y RT + quimioterapia. El registro no especificaba el agente sistémico ni confirmaba la concurrencia.

El desenlace primario fue la supervivencia global. Toda muerte registrada se contabilizó como evento; los pacientes codificados como vivos se censuraron en su último seguimiento observado. El seguimiento se truncó administrativamente a los 120 meses. Un modelo secundario independiente de riesgos específicos por causa consideró como eventos las muertes por cáncer de glándulas salivales y censuró las demás causas.

La supervivencia global fue el desenlace primario. El estimando preespecificado del efecto del tratamiento fue el ATE, expresado como el RMST medio con RT + quimioterapia menos el RMST medio con solo RT hasta los 120 meses. El RMST no sustituye a la supervivencia global: resume el área bajo la curva de supervivencia global como tiempo medio de supervivencia dentro del horizonte de 10 años y expresa el contraste entre tratamientos directamente en meses, sin requerir riesgos proporcionales. Los estimandos secundarios fueron las diferencias absolutas ajustadas de supervivencia a 60 y 120 meses. Antes de examinar los resultados, se definió como clínicamente relevante una diferencia absoluta de RMST de 6 meses o una diferencia absoluta de supervivencia de 5 puntos porcentuales.

### Ajuste por confusión, positividad y balance

Las puntuaciones de propensión se estimaron mediante regresión logística a partir de la edad, el sexo, la categoría T, la categoría N y la histología. Los pesos ATE estabilizados se truncaron en sus percentiles 1 y 99. La positividad se evaluó a partir de las distribuciones de propensión específicas de cada tratamiento y de su soporte común entre los percentiles 1 y 99. El balance se evaluó con diferencias de medias estandarizadas (SMD), con |SMD| <0,10 como objetivo diagnóstico.

A continuación, ajustamos un modelo de resultado de riesgos proporcionales de Cox ponderado que también incluía todas las covariables del modelo de propensión. Esta combinación de ponderación y ajuste del modelo de resultado se utilizó como estrategia de doble ajuste frente a la confusión medida. Las curvas de supervivencia contrafactuales marginales se generaron asignando cada nivel de tratamiento a todos los pacientes y promediando las predicciones sobre la misma población de 954 personas. El RMST se obtuvo integrando esas curvas marginales.

### Bootstrap y validación interna

El bootstrap no paramétrico remuestreó pacientes con reemplazo. En cada una de las 1.000 iteraciones se recalcularon la muestra analítica, el modelo de propensión, los pesos estabilizados y truncados, el modelo de Cox ponderado y ajustado, las curvas de supervivencia contrafactuales, los RMST y los contrastes entre tratamientos. Los intervalos de confianza del 95% por percentiles resumen la incertidumbre muestral. Los valores de P bilaterales para el RMST y para las diferencias de supervivencia a tiempo fijo se calcularon mediante una aproximación normal con el error estándar bootstrap. La fracción de estimaciones bootstrap superiores a cero no se interpretó como una probabilidad clínica de beneficio.

La validación cruzada interna con cinco particiones reajustó el preprocesamiento, los pesos de propensión y el modelo de resultado dentro de cada partición de entrenamiento. La evaluación en los pacientes reservados utilizó el índice C de Harrell, las puntuaciones de Brier dependientes del tiempo con ponderación por la probabilidad inversa de censura (IPCW) y el error de calibración (riesgo medio predicho menos riesgo observado por Kaplan–Meier) a los 60 y 120 meses. Se eliminaron un clasificador de mortalidad basado en gradient boosting y su AUC, porque no validan un modelo de supervivencia con censura. Los análisis se realizaron en Python 3.14 con lifelines 0.30, scikit-learn 1.9, pandas 2.3, NumPy 2.5 y SciPy 1.18, con una semilla aleatoria fija.

### Análisis de sensibilidad y secundarios

Los análisis de sensibilidad preespecificados excluyeron el carcinoma escamoso y restringieron la cohorte a las histologías más comparables con RTOG 1008 que podían identificarse en el extracto: carcinoma mucoepidermoide, adenocarcinoma, carcinoma de células acinares, carcinoma adenoide quístico, carcinoma SAI (sin otra indicación) y carcinoma ex adenoma pleomorfo. Las estimaciones descriptivas por histología se limitaron a grupos con al menos 80 pacientes y al menos 15 pacientes por exposición; no se utilizaron valores de P por subgrupo ni categorías de beneficio definidas a posteriori. Un valor E aproximado evaluó la sensibilidad del hazard ratio del tratamiento a la confusión no medida.

La cohorte completa siguió siendo el análisis primario. Solo como análisis secundario ilustrativo, se extrajeron 1.000 muestras de 252 pacientes asignados 1:1. Para cada paciente muestreado se conservó el RMST potencial con ambos tratamientos, y en cada simulación se registró el balance estandarizado.

## Resultados

### Selección de la cohorte y grupos de tratamiento observados

El resumen de la selección de la cohorte (tabla suplementaria S1) muestra cómo los 4.657 registros de origen se redujeron a los 954 pacientes incluidos en el análisis primario. La mayoría de las exclusiones se debieron a enfermedad distinta de T3/T4 (n=2.990); los pasos siguientes excluyeron a 276 pacientes con categoría N desconocida y a 437 sin radioterapia registrada, y ningún paciente carecía de tiempo de supervivencia.

El grupo de RT + quimioterapia era más joven, tenía mayor proporción de varones y más enfermedad T4 y con ganglios positivos, lo que demuestra diferencias sustanciales en la selección del tratamiento antes del ajuste (tabla 1). La mediana de la demora desde el diagnóstico hasta el primer tratamiento fue de 23 días (rango intercuartílico [RIC], 0–47) con solo RT y de 26 días (RIC, 0–41) con RT + quimioterapia. La mediana de seguimiento por Kaplan–Meier inverso fue de 86 meses.

Tabla 1. Características demográficas y clinicopatológicas basales de la población elegible, en conjunto y por grupo de tratamiento observado.

| Característica | Total (N=954) | Solo RT (N=671) | RT + quimioterapia (N=283) | Valor de P |
|---|---:|---:|---:|---:|
| Edad, mediana (RIC), años | 67 (57–77) | 72 (57–82) | 62 (57–72) | <0,001 |
| Sexo: masculino | 626 (65,6%) | 408 (60,8%) | 218 (77,0%) | <0,001 |
| Sexo: femenino | 328 (34,4%) | 263 (39,2%) | 65 (23,0%) |  |
| Categoría T: T3 | 522 (54,7%) | 385 (57,4%) | 137 (48,4%) | 0,014 |
| Categoría T: T4 | 432 (45,3%) | 286 (42,6%) | 146 (51,6%) |  |
| Categoría N: N0 | 462 (48,4%) | 383 (57,1%) | 79 (27,9%) | <0,001 |
| Categoría N: N1 | 125 (13,1%) | 95 (14,2%) | 30 (10,6%) |  |
| Categoría N: N2 | 268 (28,1%) | 148 (22,1%) | 120 (42,4%) |  |
| Categoría N: N3 | 99 (10,4%) | 45 (6,7%) | 54 (19,1%) |  |
| Histología: carcinoma de células escamosas | 219 (23,0%) | 138 (20,6%) | 81 (28,6%) | <0,001 |
| Histología: adenocarcinoma | 193 (20,2%) | 125 (18,6%) | 68 (24,0%) |  |
| Histología: carcinoma adenoide quístico | 135 (14,2%) | 108 (16,1%) | 27 (9,5%) |  |
| Histología: carcinoma mucoepidermoide | 110 (11,5%) | 93 (13,9%) | 17 (6,0%) |  |
| Histología: carcinoma de células acinares | 69 (7,2%) | 58 (8,6%) | 11 (3,9%) |  |
| Histología: carcinoma SAI | 71 (7,4%) | 40 (6,0%) | 31 (11,0%) |  |
| Histología: otras | 157 (16,5%) | 109 (16,2%) | 48 (17,0%) |  |

Los valores son n (%), salvo que se indique lo contrario. Los porcentajes son porcentajes por columna y pueden no sumar 100% por el redondeo. Los valores de P comparan solo RT con RT + quimioterapia y se calcularon con la prueba U de Mann–Whitney para la edad y la prueba de chi cuadrado de Pearson para las variables categóricas; describen los grupos no ajustados y no son pruebas del efecto del tratamiento. RIC, rango intercuartílico; RT, radioterapia; SAI, sin otra indicación.

### Positividad y balance de covariables

Las puntuaciones de propensión oscilaron entre 0,054 y 0,853 con RT + quimioterapia y entre 0,022 y 0,819 con solo RT. Dentro del intervalo común definido por los percentiles 1 a 99 de ambos grupos se encontraba el 86,6% y el 84,6% de los pacientes, respectivamente. La amplia superposición respaldó la comparación ponderada, mientras que las colas más escasas específicas de cada grupo motivaron el truncamiento de los pesos extremos (figura 1). Entre los desequilibrios iniciales importantes figuraban N2 (SMD 0,446), N3 (0,376), el sexo (0,356) y la edad (−0,348). Tras la ponderación por la probabilidad inversa del tratamiento (IPTW), casi todas las covariables se desplazaron hacia cero y todas las categorías codificadas salvo una tuvieron |SMD| inferior a 0,10; el mayor desequilibrio residual fue de 0,108 en la categoría poco frecuente de carcinoma adenoescamoso (figura 2).

![Figura 1. Distribución de las puntuaciones de propensión estimadas por grupo de tratamiento observado. Los histogramas normalizados por densidad comparan a los pacientes que recibieron solo RT (gris) con los que recibieron RT + quimioterapia (azul). La región central superpuesta indica que las comparaciones ponderadas están respaldadas para muchos pacientes, mientras que las colas relativamente escasas específicas de cada grupo señalan una positividad limitada y motivan el truncamiento de los pesos ATE estabilizados en los percentiles 1 y 99. ATE, efecto promedio del tratamiento; RT, radioterapia.](assets/propensity_overlap.png)

![Figura 2. Balance de covariables antes y después de la ponderación por la probabilidad inversa del tratamiento. Los puntos son diferencias de medias estandarizadas que comparan RT + quimioterapia con solo RT antes de la ponderación (rojo) y después de la ponderación ATE estabilizada y truncada (azul); se muestran las 20 categorías de covariables codificadas con mayor desequilibrio absoluto antes de la ponderación. Las líneas verticales discontinuas marcan el umbral de balance preespecificado de |SMD| = 0,10; los valores más cercanos a cero indican mejor balance. La ponderación llevó casi todas las covariables medidas al rango objetivo, con un |SMD| residual máximo de 0,108 en una categoría histológica poco frecuente. ATE, efecto promedio del tratamiento; IPTW, ponderación por la probabilidad inversa del tratamiento; RT, radioterapia; SAI, sin otra indicación; SMD, diferencia de medias estandarizada.](assets/covariate_balance_love_plot.png)

### Estimación del efecto primario

Tabla 2. Estimaciones estandarizadas de supervivencia global y contrastes entre tratamientos en la población elegible.

| Estimando | Solo RT | RT + quimioterapia | Diferencia (RT + quimioterapia − solo RT) | Valor de P |
|---|---:|---:|---:|---:|
| RMST hasta 120 meses | 67,1 meses | 64,8 meses | −2,3 meses (IC del 95%: −9,6 a +6,0) | 0,580 |
| Supervivencia a 60 meses | — | — | −2,2 puntos porcentuales | 0,581 |
| Supervivencia a 120 meses | — | — | −2,2 puntos porcentuales | 0,578 |

Los valores de RMST son las áreas bajo las curvas marginales ajustadas de supervivencia global hasta 120 meses. Las diferencias se definen como RT + quimioterapia menos solo RT, de modo que los valores negativos favorecen a solo RT. El intervalo de confianza del RMST es el intervalo de percentiles de 1.000 muestras bootstrap a nivel de paciente; los valores de P bilaterales utilizan el error estándar bootstrap. Los guiones indican que las estimaciones de supervivencia a tiempo fijo por grupo no se muestran en esta tabla resumen. IC, intervalo de confianza; RMST, tiempo medio de supervivencia restringido; RT, radioterapia.

La estimación puntual estandarizada del RMST y ambos contrastes a tiempo fijo favorecieron a solo RT, aunque ninguno alcanzó significación estadística. Por tanto, los resultados no predicen un beneficio promedio de supervivencia con la quimioterapia, mientras que el intervalo de confianza cuantifica la incertidumbre restante (tabla 2). Las curvas marginales ajustadas de supervivencia se mantuvieron próximas durante todo el seguimiento, sin una separación sostenida a favor de la quimioterapia (figura 3). En consonancia con esta incertidumbre, la distribución bootstrap a nivel de paciente del contraste de RMST se centró por debajo de cero, pero abarcó valores favorables a cualquiera de las dos estrategias (figura 4).

![Figura 3. Curvas marginales ajustadas de supervivencia global con solo RT y con RT + quimioterapia. Para cada estrategia de tratamiento, el modelo de Cox ponderado predijo la curva de supervivencia contrafactual de cada paciente, y las predicciones se promediaron sobre la misma población elegible de 954 personas. La supervivencia se truncó administrativamente a los 120 meses. Las curvas se mantienen próximas durante todo el seguimiento y no muestran una separación sostenida a favor de la quimioterapia; sus áreas integradas arrojan valores de RMST a 10 años de 67,1 y 64,8 meses, respectivamente. RMST, tiempo medio de supervivencia restringido; RT, radioterapia.](assets/adjusted_marginal_survival.png)

![Figura 4. Distribución bootstrap a nivel de paciente del contraste de tratamiento ajustado en el RMST a 10 años. En cada una de las 1.000 iteraciones se remuestrearon pacientes y se repitieron la construcción de la cohorte, la estimación de la puntuación de propensión, el cálculo y truncamiento de los pesos, el ajuste del modelo de resultado, la estandarización y la integración del RMST. El eje horizontal representa RT + quimioterapia menos solo RT, en meses; los valores inferiores a cero favorecen a solo RT y los superiores a cero, a RT + quimioterapia. La línea vertical discontinua marca el valor nulo de cero. La distribución se centra en −2,3 meses y cruza el cero (IC del 95% por percentiles, −9,6 a +6,0), lo que indica una incertidumbre sustancial en torno a la estimación puntual. ATE, efecto promedio del tratamiento; IC, intervalo de confianza; RMST, tiempo medio de supervivencia restringido; RT, radioterapia.](assets/bootstrap_rmst_difference.png)

### Rendimiento del modelo

Tabla 3. Discriminación, error de predicción y calibración del modelo de supervivencia con validación cruzada de cinco particiones.

| Métrica | 5 años | 10 años |
|---|---:|---:|
| Puntuación de Brier con IPCW, media con validación cruzada | 0,215 | 0,188 |
| Error de calibración, media con validación cruzada | +0,010 | +0,013 |

Los valores son medias de las cinco particiones reservadas. Las puntuaciones de Brier con IPCW más bajas indican una mayor exactitud global de la predicción. El error de calibración es el riesgo predicho medio menos el riesgo observado por Kaplan–Meier, de modo que los valores positivos indican una ligera sobrepredicción promedio de la mortalidad. El índice C de Harrell con validación cruzada, que resume la discriminación a lo largo del seguimiento y no en un único horizonte, fue de 0,641. IPCW, ponderación por la probabilidad inversa de censura.

El modelo de supervivencia mostró una discriminación moderada y un error de predicción aceptable en ambos horizontes clínicos. El error de calibración fue cercano a cero a los 5 y 10 años, lo que indica escasa sobrepredicción o infrapredicción promedio (tabla 3). El rendimiento se informa para el propio modelo de supervivencia; no se utiliza el AUC de ningún clasificador como validación del modelo de supervivencia.

### Análisis de sensibilidad

Tabla 4. Análisis de sensibilidad preespecificados de la estimación ajustada del efecto del tratamiento con definiciones alternativas de elegibilidad histológica.

| Análisis | N | Diferencia de RMST | Diferencia de supervivencia a 5 años | Diferencia de supervivencia a 10 años |
|---|---:|---:|---:|---:|
| Primario | 954 | −2,3 meses | −2,2 pp | −2,2 pp |
| Excluido el carcinoma escamoso | 727 | −4,1 meses | −3,9 pp | −4,0 pp |
| Histologías comparables con RTOG 1008 | 622 | −5,4 meses | −5,2 pp | −5,4 pp |

Las diferencias se definen como RT + quimioterapia menos solo RT; los valores negativos favorecen a solo RT. Cada análisis de sensibilidad repitió la ponderación por puntuación de propensión, el modelo de resultado ponderado y la estandarización dentro de la cohorte especificada. El subconjunto comparable con RTOG 1008 incluyó carcinoma mucoepidermoide, adenocarcinoma, carcinoma de células acinares, carcinoma adenoide quístico, carcinoma SAI y carcinoma ex adenoma pleomorfo. Estos análisis evalúan la robustez frente a la elegibilidad histológica y no son comparaciones aleatorizadas de subgrupos. pp, puntos porcentuales; RMST, tiempo medio de supervivencia restringido; RT, radioterapia; SAI, sin otra indicación.

La dirección de la estimación primaria se mantuvo tras excluir el carcinoma escamoso y tras restringir la cohorte a las histologías comparables con RTOG 1008 (tabla 4). Las estimaciones descriptivas por histología fueron inestables, incluida una estimación negativa de gran magnitud para el carcinoma adenoide quístico basada en solo 27 pacientes expuestos, y no se interpretaron como categorías validadas de beneficio terapéutico. El hazard ratio ajustado del tratamiento fue de 1,08 (IC del 95%: 0,85–1,37; P de Wald robusta=0,548); su valor E para la estimación puntual fue de 1,36, mientras que el valor E para el intervalo de confianza fue de 1,00, porque el intervalo incluía el valor nulo.

El modelo secundario independiente de riesgos específicos por causa para la mortalidad por cáncer estimó una diferencia de RMST a 10 años de −3,7 meses. Este análisis es secundario y no sustituye al desenlace de supervivencia global.

### Simulaciones ilustrativas con N=252

En las 1.000 simulaciones secundarias, la diferencia media de RMST fue de −2,4 meses, con estimaciones simuladas individuales entre −10,6 y +7,9 meses. La media del |SMD| máximo para edad, sexo, T4 y estado ganglionar fue de 0,185, lo que muestra por qué una única muestra aleatoria puede parecer desequilibrada. Estas simulaciones son ilustrativas y no constituyen el resultado primario.

## Discusión

En este análisis similar a RTOG 1008 del cáncer de glándulas salivales T3/T4 avanzado, el modelo no predijo un beneficio promedio en la supervivencia global al añadir quimioterapia a la radioterapia. Tras ajustar por la censura y por las diferencias medidas en la selección del tratamiento, la diferencia de RMST a 10 años fue de −2,3 meses (IC del 95% por bootstrap, −9,6 a +6,0; P=0,580), y las diferencias absolutas de supervivencia a 5 y 10 años fueron de −2,2 puntos porcentuales en ambos casos. La dirección y la magnitud de estas estimaciones no respaldan un beneficio sustancial de la intensificación con quimioterapia en el conjunto de la población. Esta predicción coincide con la posición actual de las guías: la radioterapia posoperatoria está establecida ante características adversas, mientras que la quimioterapia concurrente rutinaria no está respaldada fuera de un ensayo clínico [1,2]. RTOG 1008 existe precisamente por la falta de evidencia prospectiva de eficacia [22–24].

El primer punto de referencia para la interpretación es la literatura sobre radioterapia posoperatoria. Terhaard et al. identificaron la enfermedad T3/T4 y la resección incompleta como factores adversos para la recurrencia, y mostraron que los estadios T y N, el alto grado y la invasión perineural se asociaban con las metástasis a distancia y la supervivencia [4]. Mahmood et al. asociaron la radioterapia adyuvante con una mejor supervivencia en tumores de glándulas salivales mayores de alto grado o localmente avanzados [6]. Schoenfeld et al. informaron que la radioterapia posoperatoria de intensidad modulada fue bien tolerada y logró un elevado control local [7], mientras que Hosni et al. destacaron que las metástasis a distancia seguían siendo un patrón dominante de fracaso en pacientes de alto riesgo pese a la radioterapia posoperatoria [8]. La revisión sistemática contemporánea de Wang et al. también respaldó el papel locorregional de la radioterapia posoperatoria y subrayó la ausencia de evidencia global sólida a favor de añadir quimioterapia concurrente [9]. Las series específicas de parótida y de histología refuerzan además el uso de radioterapia posoperatoria ante características adversas como márgenes positivos, alto grado y enfermedad T3/T4 [25–27]. En conjunto, estos estudios establecen la radioterapia como base locorregional del tratamiento, pero no demuestran que la quimioterapia añada un beneficio de supervivencia.

El segundo punto de referencia, más directo, es la literatura comparativa sobre quimiorradioterapia. Las primeras series institucionales sugirieron que la quimioterapia concurrente podría mejorar los resultados en pacientes seleccionados de alto riesgo [11,12]. Estos informes aportaron una justificación biológica y clínica importante para intensificar el tratamiento, pero el reducido tamaño muestral, los diseños no aleatorizados y la susceptibilidad al sesgo de selección del tratamiento limitaron la interpretación causal. Comparaciones institucionales posteriores no demostraron una ventaja clara de la quimiorradioterapia en la supervivencia global [13,16]. En pacientes de mayor edad, el análisis SEER-Medicare de Tanvetyanon et al. comunicó resultados que no favorecieron la intensificación con quimioterapia [14], y el amplio análisis de la NCDB de Amini et al. no encontró una ventaja de supervivencia global de la quimiorradioterapia adyuvante frente a la radioterapia sola [15]. Estos conjuntos de datos más grandes desplazaron el balance de la evidencia en contra del uso rutinario de quimioterapia, aunque la confusión residual siguió siendo inevitable.

Los estudios más recientes han seguido en gran medida la misma dirección, aunque sugieren que cualquier beneficio podría concentrarse en subgrupos biológicos o clinicopatológicos seleccionados. Kang et al. no encontraron beneficio en la supervivencia global ni en la específica de la enfermedad al añadir quimioterapia en el cáncer avanzado de glándulas salivales mayores [19]. Sin embargo, Hsieh et al. y Shen et al. describieron posibles señales en pacientes con enfermedad ganglionar, resección R2, carcinoma adenoide quístico o combinaciones de tumores T3/T4 de alto grado y gran carga ganglionar [20,21]. De forma similar, el estudio emparejado por puntuación de propensión de Hsieh et al. sobre carcinoma adenoide quístico sugirió una mejora del control locorregional sin una mejora correspondiente de la supervivencia global [17]. Esta distinción es clínicamente importante: un efecto radiosensibilizador puede mejorar el control local sin superar el riesgo de metástasis a distancia ni traducirse en una supervivencia más prolongada. Por tanto, la evidencia disponible no respalda la adición rutinaria de quimioterapia en una población de alto riesgo no seleccionada, pero tampoco excluye un beneficio en un subgrupo biológicamente enriquecido.

Nuestros hallazgos siguen la dirección predominante de la evidencia comparativa. La estimación primaria no favoreció a la quimioterapia, y ni la exclusión del carcinoma escamoso ni la restricción a las histologías comparables con RTOG 1008 revelaron una ventaja global de supervivencia. Por tanto, la predicción central es que la quimioterapia no mejorará la supervivencia global promedio en una población amplia con enfermedad T3/T4 avanzada. Como en la literatura retrospectiva reciente, un beneficio limitado a un subgrupo biológico estrechamente seleccionado sigue siendo posible, pero no está demostrado.

El contexto prospectivo más importante es RTOG 1008. Ese ensayo se diseñó porque la evidencia retrospectiva era insuficiente y contradictoria: los primeros estudios pequeños sugirieron un posible beneficio [11,12], mientras que las comparaciones institucionales y de registros más amplias no confirmaron una ventaja de supervivencia consistente [13–16]. RTOG 1008 compara directamente la radioterapia posoperatoria sola con la radioterapia más cisplatino semanal en tumores malignos de glándulas salivales de alto riesgo resecados [22–24]. El estudio GORTEC-REFCOR SANTAL evalúa de forma similar la radioterapia con o sin cisplatino en tumores de glándulas salivales y nasosinusales [28]. En conjunto, estos ensayos muestran que la radiosensibilización con platino sigue siendo una estrategia de investigación plausible, mientras que las guías de ASCO y ESMO-EURACAN no respaldan su uso rutinario fuera de ensayos clínicos [1,2]. Por consiguiente, este análisis no debe plantearse como un sustituto de RTOG 1008. Su principal valor es la falsabilidad prospectiva: aporta una predicción basada en un modelo, formulada antes de conocer los resultados, que podrá compararse posteriormente con la evidencia aleatorizada.

Si RTOG 1008 demuestra un beneficio de supervivencia clínicamente relevante, refutará la predicción actual e identificará dimensiones del efecto del tratamiento que el modelo de registro no captó. Si no demuestra beneficio, el resultado respaldará la predicción de que la quimioterapia no debe añadirse de forma rutinaria a la radioterapia posoperatoria fuera de subgrupos seleccionados o de ensayos clínicos.

Este trabajo también pone de relieve la diferencia entre la modelización pronóstica y la predictiva. Los modelos existentes estiman la recurrencia, la supervivencia posoperatoria, las metástasis a distancia o el pronóstico basal [29–35]. Sin embargo, un riesgo alto no implica automáticamente un beneficio de la quimioterapia. Al estimar la supervivencia con ambas estrategias de tratamiento en la misma población, este análisis se dirige directamente al efecto incremental de la quimioterapia y no solo al pronóstico. Su predicción clínicamente contrastable es sencilla: añadir quimioterapia a la radioterapia no mejorará la supervivencia global promedio en el cáncer de glándulas salivales T3/T4 avanzado.

## Limitaciones

Este análisis de registro sigue siendo susceptible a confusión residual, en particular por factores de riesgo patológicos, estado funcional, momento del tratamiento y detalles terapéuticos no incluidos en el extracto. El diagnóstico fue el origen temporal disponible, no pudo confirmarse la concurrencia de los tratamientos y las estimaciones por histología se basaron en pocos pacientes. El modelo de Cox también impone riesgos proporcionales, y la validación fue interna. Estas limitaciones afectan a la certeza causal y a la resolución por subgrupos, pero no modifican la predicción poblacional preespecificada del estudio.

## Conclusión

El modelo predice que añadir quimioterapia a la radioterapia no mejora la supervivencia global promedio en el cáncer de glándulas salivales T3/T4 avanzado. La diferencia ajustada de RMST a 10 años fue de −2,3 meses, sin un efecto favorable a los 5 ni a los 10 años. Esta predicción falsable, formulada antes de conocer los resultados, queda a la espera de su comparación con RTOG 1008.

## Referencias

1. Geiger JL, Ismaila N, Beadle B, Caudell JJ, Chau N, Deschler D, et al. Management of Salivary Gland Malignancy: ASCO Guideline. J Clin Oncol. 2021;39:1909-1941. doi:10.1200/JCO.21.00449.
2. van Herpen C, Locati LD, But-Hadzic J, Bossi P, Cavalieri S, Licitra L, et al. Salivary gland cancer: ESMO-EURACAN Clinical Practice Guideline for diagnosis, treatment and follow-up. ESMO Open. 2022. PMID:36567082.
3. PDQ Adult Treatment Editorial Board. Salivary Gland Cancer Treatment (PDQ). National Cancer Institute; updated 2025.
4. Terhaard CHJ, Lubsen H, van der Tweel I, Hilgers FJM, Eijkenboom WMH, Marres HAM, et al. Salivary gland carcinoma: independent prognostic factors for locoregional control, distant metastases, and overall survival: results of the Dutch head and neck oncology cooperative group. Head Neck. 2004;26:681-693. PMID:15287035.
5. Mendenhall WM, Morris CG, Amdur RJ, Werning JW, Hinerman RW, Villaret DB. Radiotherapy alone or combined with surgery for salivary gland carcinoma. Cancer. 2005;103:2544-2550. PMID:15880750.
6. Mahmood U, Koshy M, Goloubeva O, Suntharalingam M. Adjuvant radiation therapy for high-grade and/or locally advanced major salivary gland tumors. Arch Otolaryngol Head Neck Surg. 2011. PMID:22006781.
7. Schoenfeld JD, Sher DJ, Norris CM Jr, Haddad RI, Posner MR, Balboni TA, et al. Salivary gland tumors treated with adjuvant intensity-modulated radiotherapy with or without concurrent chemotherapy. Int J Radiat Oncol Biol Phys. 2012;82:308-314. PMID:21075557.
8. Hosni A, Huang SH, Goldstein D, Xu W, Chan B, Hansen A, et al. Outcomes and prognostic factors for major salivary gland carcinoma following postoperative radiotherapy. Oral Oncol. 2016;54:75-80. PMID:26723908.
9. Wang J, et al. The Current Position of Postoperative Radiotherapy for Salivary Gland Cancer: A Systematic Review and Meta-Analysis. Cancers. 2024;16:2375.
10. Cerda T, Sun XS, Vignot S, et al. A rationale for chemoradiation versus radiotherapy in salivary gland cancers? Crit Rev Oncol Hematol. 2014;91:142-158. PMID:24636481.
11. Pederson AW, Salama JK, Haraf DJ, Witt ME, Stenson KM, Portugal L, et al. Adjuvant chemoradiotherapy for locoregionally advanced and high-risk salivary gland malignancies. Head Neck Oncol. 2011;3:31. PMID:21791072.
12. Tanvetyanon T, Qin D, Padhya T, McCaffrey J, Zhu W, Boulware D, et al. Outcomes of postoperative concurrent chemoradiotherapy for locally advanced major salivary gland carcinoma. Arch Otolaryngol Head Neck Surg. 2009;135:687-692. doi:10.1001/archoto.2009.70.
13. Mifsud MJ, Tanvetyanon T, McCaffrey JC, Otto KJ, Padhya TA, Kish J, et al. Adjuvant radiotherapy versus concurrent chemoradiotherapy for the management of high-risk salivary gland carcinomas. Head Neck. 2016;38:1628-1633. doi:10.1002/hed.24484.
14. Tanvetyanon T, Fisher K, Caudell J, Otto K, Padhya T, Trotti A. Adjuvant chemoradiotherapy versus radiotherapy alone for locally advanced salivary gland carcinoma among older patients. Head Neck. 2016;38:863-870. PMID:26340707.
15. Amini A, Waxweiler TV, Brower JV, Jones BL, McDermott JD, Raben D, et al. Association of adjuvant chemoradiotherapy vs radiotherapy alone with survival in patients with resected major salivary gland carcinoma: data from the National Cancer Data Base. JAMA Otolaryngol Head Neck Surg. 2016;142:1100-1110. doi:10.1001/jamaoto.2016.2168.
16. Gebhardt BJ, Ohr JP, Ferris RL, Duvvuri U, Johnson JT, Kim S, et al. Concurrent chemoradiotherapy in the adjuvant treatment of salivary gland malignancies. Am J Clin Oncol. 2018;41:888-893. PMCID:PMC6587550.
17. Hsieh CE, Lin CY, Lee LY, Yang LY, Wang CC, Wang HM, et al. Adding concurrent chemotherapy to postoperative radiotherapy improves locoregional control but not overall survival in patients with salivary gland adenoid cystic carcinoma: a propensity score matched study. Radiat Oncol. 2016;11:47. doi:10.1186/s13014-016-0617-7.
18. Yan W, Huang S, et al. Postoperative chemoradiotherapy versus radiotherapy alone for major salivary gland malignancies: a stratified study based on the external validation of the distant metastasis risk score model. Cancers. 2022;14:5583.
19. Kang NW, Kuo YH, Wu HC, Ho CH, Chen YC, Yang CC. No survival benefit from adding chemotherapy to adjuvant radiation in advanced major salivary gland cancer. Sci Rep. 2022;12:20862. doi:10.1038/s41598-022-25468-9.
20. Hsieh RCE, et al. A multicenter retrospective analysis of patients with salivary gland carcinoma receiving postoperative chemoradiotherapy versus postoperative radiotherapy. Radiother Oncol. 2023. PMID:37659659.
21. Shen Y, Shan J. Chemoradiotherapy versus radiotherapy in high risk salivary gland cancer. World J Surg Oncol. 2024;22:181. doi:10.1186/s12957-024-03456-9.
22. NRG Oncology. RTOG-1008: A randomized phase II/phase III study of adjuvant concurrent radiation and chemotherapy versus radiation alone in resected high-risk malignant salivary gland tumors. NRG Oncology protocol page.
23. NRG Oncology. RTOG 1008 protocol, version 11/5/2015. Required sample size: Phase II 120; Phase III 252.
24. ClinicalTrials.gov. Radiation Therapy With or Without Chemotherapy in Treating Patients With High-Risk Malignant Salivary Gland Tumors. Identifier: NCT01220583.
25. Kim YH, et al. Evaluation of prognostic factors for the parotid cancer treated with surgery and postoperative radiotherapy. Cancer Res Treat. 2020. PMID:31480828.
26. Park G, et al. Postoperative radiotherapy for mucoepidermoid carcinoma of major salivary glands: long-term results of a single-institution experience. Radiat Oncol J. 2018. PMID:30630270.
27. Katano A, et al. Postoperative radiotherapy for malignant major salivary gland tumors. 2023.
28. GORTEC/REFCOR. Treatment of Salivary Glands and Nasal Tumors (SANTAL). ClinicalTrials.gov Identifier: NCT02998385.
29. Lukovic J, Sultana R, et al. Development and validation of a clinical prediction-score model for distant metastases in salivary gland cancer. Oral Oncol. 2020. PMID:31959347.
30. Ali S, Palmer FL, Yu C, DiLorenzo M, Shah JP, Kattan MW, et al. A predictive nomogram for recurrence of carcinoma of the major salivary glands. JAMA Otolaryngol Head Neck Surg. 2013;139:698-705. doi:10.1001/jamaoto.2013.3347.
31. Ali S, Palmer FL, Yu C, DiLorenzo M, Shah JP, Kattan MW, Patel SG, Ganly I. Postoperative nomograms predictive of survival after surgical management of malignant tumors of the major salivary glands. Ann Surg Oncol. 2014;21:637-642. PMID:24132626.
32. Hay A, Migliacci J, Karassawa Zanoni D, et al. Validation of nomograms for overall survival, cancer-specific survival and recurrence in major salivary gland cancer. Head Neck. 2018. PMID:29389040.
33. Chen Y, Li Y, et al. Prognostic risk factor of major salivary gland carcinomas and survival prediction model based on random survival forests. Cancer Med. 2023. PMID:36934429.
34. Du W, et al. Prognostic prediction model for salivary gland carcinoma based on machine learning. 2024. PMID:38981745.
35. Jacobs CD, et al. Prediction model to estimate overall survival benefit of postoperative radiation therapy for resected major salivary gland cancers. Oral Oncol. 2022. doi:10.1016/j.oraloncology.2022.105902.
