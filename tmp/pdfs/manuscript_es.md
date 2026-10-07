# Modelización predictiva de resultados en cánceres de glándulas salivales mediante aprendizaje automático: validación prospectiva simulada con el ensayo RTOG 1008

Autores: Federico Lorenzo; Agustin Rosich; Jesica Lell; Sergio Aguiar; Valentina Ferreira; Karina Ochandorena; Eduardo Larrinaga; Natalia Gadea; Nicolas Larragueta; Aldo Quarneti

## Resumen

### Antecedentes

No se ha establecido si la adición de quimioterapia a la radioterapia mejora la supervivencia en el cáncer de glándulas salivales de alto riesgo. Desarrollamos un modelo falsable, previo a la publicación de resultados, que predice el efecto promedio de la quimioterapia sobre la supervivencia en enfermedad T3/T4 avanzada, para compararlo en el futuro con RTOG 1008.

### Métodos

Analizamos una cohorte estandarizada derivada de SEER de tumores T3/T4, con categoría N conocida y radioterapia registrada. La supervivencia global se modelizó considerando la censura. El estimando preespecificado fue el efecto promedio del tratamiento (ATE), expresado como la diferencia en el tiempo medio de supervivencia restringida (RMST) hasta 120 meses entre RT más quimioterapia y solo RT. Se utilizaron ponderación por puntuación de propensión y un modelo de Cox ajustado por covariables para abordar las diferencias medidas en la selección del tratamiento. Ambos resultados potenciales se estandarizaron a la misma población y el análisis completo se repitió en 1.000 muestras bootstrap a nivel de paciente. El rendimiento interno se evaluó mediante validación cruzada de cinco pliegues.

### Resultados

De 4.657 registros de origen, 954 cumplieron los criterios de elegibilidad; 283 recibieron RT más quimioterapia y 671 recibieron solo RT. El RMST ajustado a 10 años fue de 64,8 meses con RT más quimioterapia y de 67,1 meses con solo RT. El ATE fue de -2,3 meses (IC del 95% por bootstrap, -9,6 a +6,0; P=0,580). Las diferencias absolutas ajustadas de supervivencia fueron de -2,2 puntos porcentuales tanto a 5 como a 10 años. La mediana de seguimiento mediante Kaplan-Meier inversa fue de 86 meses. El índice C medio con validación cruzada fue de 0,641; las puntuaciones de Brier con IPCW fueron de 0,215 a 5 años y de 0,188 a 10 años.

### Conclusiones

El modelo predijo que la adición de quimioterapia a la radioterapia no aporta un beneficio promedio en la supervivencia global de pacientes con enfermedad T3/T4 avanzada. La estimación concuerda con la evidencia retrospectiva contraria a la intensificación rutinaria con quimioterapia y ofrece una predicción previa a los resultados para su evaluación prospectiva frente a RTOG 1008.

## Introducción

La cirugía seguida por la radioterapia postoperatoria adaptada al riesgo es la estrategia loregional establecida para malignidades de glándulas salivales de alto riesgo, mientras que el valor incremental de la quimioterapia sigue sin resolverse. Los primeros informes institucionales sugirieron una posible ventaja de la intensificación del tratamiento, pero las comparaciones retrospectivas más grandes no han demostrado un beneficio de supervivencia constante. RTOG 1008 fue diseñado para resolver esta pregunta comparando la radioterapia postoperatoria sola con la radioterapia más cisplatino semanal en enfermedad de alto riesgo resecada.

El objetivo de este estudio fue predecir si la quimioterapia produce una ganancia de supervivencia promedio clínicamente relevante cuando se añade a la radioterapia en el cáncer avanzado de glándulas salivales T3/T4. Usamos métodos de supervivencia causal de censura y conservamos el tamaño de la muestra RTOG 1008 sólo como un punto de referencia de simulación secundario.

## Métodos

### Fuente de datos y criterios de elegibilidad aplicables

La fuente contenía 4.657 registros estandarizados y 11 variables: edad agrupada, sexo, histología ICD-O-3, histología armonizada, categorías T y N armonizadas, indicadores de radioterapia binaria y quimioterapia, retraso de diagnóstico a primer tratamiento, causa recaída de muerte y meses de supervivencia. Requerimos secuencialmente la enfermedad T3/T4, conocida categoría N, radioterapia registrada y tiempo de supervivencia no faltante.

Este fue un análisis observacional de tipo RTOG 1008. El diagnóstico fue el origen del tiempo disponible, y el tratamiento se clasificó de los indicadores del registro de radioterapia y quimioterapia.

### Tratamiento, desenlace y estimando

Los grupos de tratamiento fueron etiquetados únicamente RT y quimioterapia RT +. El registro no especificó el agente sistémico ni confirmó la concurrencia.

El desenlace primario era la supervivencia global. Cualquier muerte registrada contada como evento; los pacientes codificados vivos fueron censurados en su último seguimiento observado. El seguimiento fue truncado administrativamente a 120 meses. Un modelo secundario específico de causa contó muertes de cáncer salivar-gland como eventos y censuraron otras causas.

La supervivencia global fue el desenlace primario. La estimación predeterminada de efectos de tratamiento fue el ATE expresado como RMST medio con RT + quimioterapia menos significa RMST con RT sólo a 120 meses. RMST no sustituye la supervivencia global; resume la zona bajo la curva de supervivencia global como tiempo promedio de supervivencia dentro del horizonte de 10 años y expresa el contraste de tratamiento directamente en meses sin requerir riesgos proporcionales. Las estimaciones secundarias eran diferencias absolutas de supervivencia ajustadas a 60 y 120 meses. Antes de examinar los resultados, se designó clínicamente relevante una diferencia RMST absoluta de 6 meses o una diferencia de supervivencia absoluta de 5 puntos porcentuales.

### Ajuste por confusión, positividad y balance

Las puntuaciones de propensidad fueron estimadas por regresión logística de edad, sexo, categoría T, categoría N y histología. Los pesos ATE estabilizados fueron truncados en sus primeros y 99 percentiles. Se evaluó la Positividad de las distribuciones de propensión específicas para el tratamiento y su apoyo común percentil 1 al 99. El equilibrio se evaluó con diferencias medias estandarizadas (SMDs), utilizando el objetivo de diagnóstico.

A continuación, hemos equipado un modelo de resultado Cox proporcional-hazards ponderado que también se ajusta para todas las covariables de propensidad. Este ajuste combinado de ponderación y resultados se utilizó como estrategia doblemente ajustada contra la confusión medida. Las curvas de supervivencia contrafactual marginal fueron generadas estableciendo tratamiento a cada nivel para cada paciente y promediando predicciones sobre la misma población de 954 personas. RMST se obtuvo mediante la integración de esas curvas marginales.

### Bootstrap y validación interna

Los pacientes de bootstrap no paramétricos con reposición. Dentro de cada una de 1.000 iteraciones, se recalculó la muestra analítica, el modelo de propensión, los pesos estabilizados y truncados, el modelo Cox ajustado, las curvas de supervivencia contrafactual, los RMST y los contrastes de tratamiento. Los intervalos de confianza del 95% resumen la incertidumbre de muestreo. Los valores de P de dos caras para RMST y las diferencias de supervivencia de tiempo fijo se calcularon con una aproximación normal utilizando el error estándar bootstrap. La fracción de las estimaciones bootstrap por encima de cero no se interpretó como una probabilidad clínica de beneficio.

Cinco veces la validación interna refinada preprocesamiento, pesos de propensión y el modelo de resultado dentro de cada plegado de entrenamiento. La evaluación en pacientes detenidos utilizó el índice C de Harrell, las puntuaciones Brier dependientes del tiempo IPCW y el error de calibración (medio predicho menos riesgo observado Kaplan-Meier) en 60 y 120 meses. Un gradiente-boosting mortality classifier y su AUC fueron eliminados porque no validan un modelo de supervivencia censurado.

### Análisis de sensibilidad y secundarios

Los análisis de sensibilidad predeterminados excluyen el carcinoma escamoso y limitan el cohorte a las histologías más comparables con RTOG 1008 que fueron identificables en el extracto: mucoepidermoide, adenocarcinoma, célula acinar, quístico adenoide, carcinoma NOS y carcinoma ex pleomorfo adenoma. Las estimaciones específicas de histología descriptivas se limitaron a grupos con al menos 80 pacientes y al menos 15 pacientes por exposición; no se utilizaron valores de p subgrupos ni categorías de beneficios post hoc. Una sensibilidad evaluada aproximada del valor E de la relación de riesgo de tratamiento a la confusión inmedida.

La cohorte total seguía siendo primaria. Como análisis secundario ilustrativo, se dibujaron 1.000 muestras de 252 pacientes y se asignaron 1:1. Se retuvo RMST potencial bajo ambos tratamientos para cada paciente muestreado, y se registró un equilibrio estandarizado en cada simulación.

## Resultados

### Selección de la cohorte y grupos de tratamiento observados

El resumen de la selección de cohortes muestra cómo se redujeron los registros de 4.657 fuentes a los 954 pacientes incluidos en el análisis primario. La mayoría de las exclusiones se derivaron de la enfermedad fuera de T3/T4; los pasos subsiguientes requerían estado nodal conocido, radioterapia y un tiempo de supervivencia disponible.

| Paso de selección | Permaneciendo | Excluido a paso |
|---|---:|---:|
| Registros normalizados | 4,657 | 0 |
| Enfermedad T3/T4 | 1,667 | 2,990 |
| Conocida categoría N | 1,391 | 276 |
| Radioterapia grabada | 954 | 437 |
| Tiempo de supervivencia sin pérdidas | 954 | 0 |

Tabla 1. Características demográficas y clínicas basales de la población elegible, en general y por grupo de tratamiento observado.

| Características | (N=954) | solo RT (N=671) | RT + quimioterapia (N=283) | Valor de P |
|---|---:|---:|---:|---:|
| Edad, mediana (IQR), años | 67 (57–77) | 72 (57–82) | 62 (57–72) | <0.001 |
| Sexo: Hombre | 626 (65.6%) | 408 (60.8%) | 218 (77.0%) | <0.001 |
| Sexo: Mujer | 328 (34.4%) | 263 (39.2%) | 65 (23.0%) |  |
| Categoría T: T3 | 522 (54.7%) | 385 (57.4%) | 137 (48.4%) | 0.014 |
| Categoría T: T4 | 432 (45.3%) | 286 (42.6%) | 146 (51.6%) |  |
| Categoría N: N0 | 462 (48.4%) | 383 (57.1%) | 79 (27.9%) | <0.001 |
| Categoría N: N1 | 125 (13.1%) | 95 (14.2%) | 30 (10.6%) |  |
| Categoría N: N2 | 268 (28.1%) | 148 (22.1%) | 120 (42.4%) |  |
| Categoría N: N3 | 99 (10.4%) | 45 (6.7%) | 54 (19.1%) |  |
| Histología: carcinoma de células escamosas | 219 (23.0%) | 138 (20.6%) | 81 (28.6%) | <0.001 |
| Histología: Adenocarcinoma | 193 (20.2%) | 125 (18.6%) | 68 (24.0%) |  |
| Histología: Carcinoma cístico Adenoide | 135 (14.2%) | 108 (16.1%) | 27 (9.5%) |  |
| Histología: Carcinoma mucoepidermoide | 110 (11.5%) | 93 (13.9%) | 17 (6.0%) |  |
| Histología: carcinoma de células acinares | 69 (7.2%) | 58 (8.6%) | 11 (3.9%) |  |
| Histología: Carcinoma NOS | 71 (7.4%) | 40 (6.0%) | 31 (11.0%) |  |
| Histología: Otros | 157 (16.5%) | 109 (16.2%) | 48 (17.0%) |  |

Los valores son n (%) a menos que se indique lo contrario. Los porcentajes son porcentajes de columna y pueden no totalizar el 100% debido al redondeo. Los valores de P comparan RT sólo con la quimioterapia RT + y se calcularon usando el test Mann-Whitney U para la edad y la prueba de chi-cuadrado de Pearson para variables categóricas; describen los grupos no ajustados y no son pruebas de efecto de tratamiento. El grupo de quimioterapia RT + era más joven, más frecuentemente masculino, y tenía más T4 y enfermedad de ganglios positivos, demostrando diferencias sustanciales de selección de tratamientos antes del ajuste (tabla 1). La demora mediana de diagnóstico a primer tratamiento fue de 23 días (IQR 0–47) con RT sólo y 26 días (IQR 0–41) con quimioterapia RT +. El seguimiento inverso de Kaplan-Meier fue de 86 meses. IQR, rango intercuartil; RT, radioterapia.

### Positividad y balance de covariables

Las puntuaciones de propensidad oscilaron entre 0.054 y 0.853 con RT + quimioterapia y entre 0.022 y 0.819 con solo RT. Dentro del intervalo común definido por los percentiles 1 al 99 de ambos grupos fueron 86,6% y 84,6%, respectivamente. La amplia superposición soportaba la comparación ponderada, mientras que las colas más delgadas específicas de grupo motivaban la truncación de pesos extremos (figura 1). Entre los desequilibrios iniciales importantes figuraban N2 (SMD 0.446), N3 (0.376), sexo (0.356) y edad (−0.348). Después de IPTW, casi todos los covariados se movieron hacia cero y todos menos uno covariado codificado tenían |SMD| debajo de 0.10; el mayor desequilibrio residual fue de 0.108 para la escasa categoría adenosquamous (Figura 2).

![Figura 1. Distribución de puntuaciones de propensión estimados por grupo de tratamiento observado. Los histogramas normalizados por la densidad comparan a los pacientes que reciben solo RT (gris) con los que reciben RT + quimioterapia (azul). La región central superpuesta indica que las comparaciones ponderadas son soportadas para muchos pacientes, mientras que las colas relativamente escasas específicas de grupo identifican positividad limitada y motivan la truncación de pesos ATE estabilizados en los percentiles 1 y 99. ATE, efecto de tratamiento promedio; RT, radioterapia.](assets/propensity_overlap.png)

![Figura 2. Equilibrio covariable antes y después de la ponderación inversa-probabilidad del tratamiento. Los puntos son diferencias medias estandarizadas comparando RT + quimioterapia con RT sólo antes de ponderar (rojo) y después de estabilizar, truncado ATE ponderación (azul). Las líneas verticales desgarradas marcan el umbral de equilibrio preespejado TENSMDANTE = 0.10; los valores más cercanos a cero indican un mejor equilibrio. Weighting movió casi todos los covariados medidos dentro del rango de destino, con un máximo residual tenciónSMD duración de 0.108 en una categoría de histología escasa. ATE, efecto promedio del tratamiento; IPTW, ponderación de la probabilidad inversa del tratamiento; RT, radioterapia; SMD, diferencia media estandarizada.](assets/covariate_balance_love_plot.png)

### Estimación del efecto primario

Tabla 2. Estimaciones de supervivencia global normalizadas y contrastes de tratamiento en la población elegible.

| Estimando | RT sólo | RT + quimioterapia | Diferencia (RT + quimioterapia - solo RT) | Valor de P |
|---|---:|---:|---:|---:|
| RMST a 120 meses | 67,1 meses | 64,8 meses | −2,3 meses (95% CI – 9,6 a +6,0) | 0.580 |
| Supervivencia a los 60 meses | — | — | −2.2 puntos porcentuales | 0.581 |
| Supervivencia a 120 meses | — | — | −2.2 puntos porcentuales | 0.578 |

Los valores RMST son áreas bajo las curvas marginales de supervivencia global ajustadas a través de 120 meses. Las diferencias se definen como RT + quimioterapia menos solo RT, por lo que los valores negativos favorecen solo RT. El intervalo de confianza RMST es el intervalo de percentil de 1.000 muestras bootstrap de nivel paciente; los valores P de dos caras utilizan el error estándar bootstrap. Las guiones indican que las estimaciones de supervivencia de tiempo fijo específicas del brazo no se muestran en esta tabla sumaria. CI, intervalo de confianza; RMST, tiempo de supervivencia medio restringido; RT, radioterapia.

La estimación estándar de puntos RMST y ambos contrastes de tiempo fijo favorecieron sólo RT, aunque ninguno alcanzó significado estadístico. Por lo tanto, los resultados no predicen ningún beneficio promedio de supervivencia de la quimioterapia, mientras que el intervalo de confianza cuantifica la incertidumbre restante (tabla 2). Las curvas de supervivencia marginal ajustadas permanecieron cerca durante todo el seguimiento, sin separación sostenida que favorezca la quimioterapia (Figura 3). Consecuente con esta incertidumbre, la distribución a nivel de paciente del contraste RMST se centró por debajo de cero pero los valores abarcados favoreciendo la estrategia de tratamiento (figura 4).

![Figura 3. Curvas marginales de supervivencia global ajustadas bajo solo RT y quimioterapia RT +. Para cada estrategia de tratamiento, el modelo ponderado de Cox predijo la curva de supervivencia contrafactual de cada paciente y las predicciones se promediaron sobre la misma población elegible de 954 personas. La supervivencia fue truncada administrativamente a 120 meses. Las curvas permanecen cerca a lo largo del seguimiento y no muestran separación sostenida que favorezca la quimioterapia; sus áreas integradas producen valores de RMST de 10 años de 67,1 y 64,8 meses, respectivamente. RMST, tiempo de supervivencia medio restringido; RT, radioterapia.](assets/adjusted_marginal_survival.png)

![Figura 4. Distribución a nivel de pacientes del contraste de tratamiento RMST de 10 años ajustado. Cada una de 1.000 iteraciones reaparecen pacientes y repetidas construcciones de cohortes, estimación de propensión-score, cálculo de peso y truncación, ajuste de modelo de resultados, estandarización e integración RMST. El eje horizontal es RT + quimioterapia menos RT sólo en meses; valores por debajo de cero favor RT sólo y valores por encima de cero favor RT + quimioterapia. La línea vertical discontinua marca el valor nulo de cero. La distribución se centra en −2.3 meses y cruza cero (95% percentil CI, −9.6 a +6.0), indicando incertidumbre sustancial alrededor de la estimación de puntos. ATE, efecto de tratamiento promedio; CI, intervalo de confianza; RMST, tiempo de supervivencia medio restringido; RT, radioterapia.](assets/bootstrap_rmst_difference.png)

### Rendimiento del modelo

Tabla 3. Discriminación multivalidada, error de predicción y calibración del modelo de supervivencia.

| Métrica | 5 años | 10 años |
|---|---:|---:|
| Puntuación de Brier con IPCW, media con validación cruzada | 0.215 | 0.188 |
| Error de calibración, media con validación cruzada | +0.010 | +0.013 |

Los valores son medios a través de los cinco pliegues retenidos. Menor IPCW Las puntuaciones de Brier indican una mejor precisión de predicción general. El error de calibración es un riesgo predicho menos Kaplan-Meier observado riesgo, por lo que los valores positivos indican una ligera sobrepredicción promedio de la mortalidad. El índice Harrell C-index cruzado, que resume la discriminación en el seguimiento en lugar de en un horizonte único, fue de 0.641. AUC, área bajo la curva; IPCW, probabilidad inversa de censura de ponderación.

El modelo de supervivencia mostró una discriminación moderada y un error de predicción aceptable en ambos horizontes clínicos. El error de calibración estuvo cerca de cero a 5 y 10 años, indicando poco sobre- o subpredicción promedio (tabla 3). El rendimiento se reporta para el modelo de supervivencia en sí; ningún clasificador AUC se utiliza como validación de modelo de supervivencia.

### Análisis de sensibilidad

Tabla 4. Análisis de sensibilidad predeterminado de la estimación ajustada de efectos de tratamiento en las definiciones de elegibilidad histológica alternativas.

| Análisis | N | Diferencia de RMST | Diferencia de supervivencia a 5 años | Diferencia de supervivencia a 10 años |
|---|---:|---:|---:|---:|
| Primaria | 954 | -2,3 meses | −2.2 pp | −2.2 pp |
| Carcinoma escamoso | 727 | -4,1 meses | −3.9 pp | −4−0 pp |
| RTOG 1008-comparable histologies | 622 | -5,4 meses | −5.2 pp | −5.4 pp |

Las diferencias se definen como RT + quimioterapia menos solo RT; los valores negativos favorecen solo RT. Cada análisis de sensibilidad repetida propensidad-score ponderación, modelado de resultados ponderado y estandarización dentro de la cohorte especificada. El subconjunto compatible con RTOG 1008 incluía mucoepidermoide, adenocarcinoma, célula acinar, quístico adenoide, carcinoma NOS y carcinoma ex pleomorfo adenoma. Estos análisis evalúan la robustez a la elegibilidad histológica y no son comparaciones de subgrupos aleatorizadas. NOS, no especificado de otra manera; pp, puntos porcentuales; RMST, tiempo de supervivencia medio restringido; RT, radioterapia.

La dirección de la estimación primaria persistió después de excluir el carcinoma escamoso y después de restringir el cohorte a las histologías comparables con RTOG 1008 (tabla 4). Las estimaciones específicas de histología descriptiva fueron inestables, incluyendo una gran estimación negativa para el carcinoma cístico adenoide basado en sólo 27 pacientes expuestos, y no se interpretaron como categorías de beneficios de tratamiento validados. El coeficiente de riesgo de tratamiento ajustado fue de 1,08 (95% CI 0,85–1,37; Wald P=0,548 robusto); su valor de E-estimado de punto fue de 1,36, mientras que el valor de E-intervalo de confianza fue de 1,00 porque el intervalo incluyó el nulo.

El modelo secundario de riesgos específicos por causa para la mortalidad por cáncer estimó una diferencia de RMST a 10 años de -3,7 meses. Este análisis es secundario y no sustituye el desenlace de supervivencia global.

### Simulaciones ilustrativas con N=252

En 1.000 simulaciones secundarias, la diferencia media del RMST fue de -2,4 meses, con estimaciones individuales entre -10,6 y +7,9 meses. La media del SMD absoluto máximo para edad, sexo, T4 y estado ganglionar fue de 0,185, lo que demuestra por qué una única muestra aleatoria puede parecer desequilibrada. Estas simulaciones son ilustrativas y no constituyen el resultado primario.

## Discusión

En este análisis similar a RTOG 1008 de cáncer de glándulas salivales T3/T4 avanzado, el modelo predijo que añadir quimioterapia a la radioterapia no produce un beneficio promedio en la supervivencia global. Tras ajustar por censura y por las diferencias medidas en la selección del tratamiento, la diferencia de RMST a 10 años fue de -2,3 meses (IC del 95% por bootstrap, -9,6 a +6,0; P=0,580), y las diferencias absolutas de supervivencia a 5 y 10 años fueron de -2,2 puntos porcentuales en ambos casos. La dirección y la magnitud de estas estimaciones no respaldan un beneficio sustancial de la intensificación con quimioterapia en el conjunto de la población. Esta predicción coincide con las guías actuales: la radioterapia posoperatoria está establecida ante características adversas, mientras que la quimioterapia concurrente rutinaria no está respaldada fuera de un ensayo clínico [1,2]. RTOG 1008 existe precisamente por la falta de evidencia prospectiva de eficacia [22-24].

El primer punto de referencia para la interpretación es la literatura sobre radioterapia posoperatoria. Terhaard et al. identificaron la enfermedad T3/T4 y la resección incompleta como factores adversos para la recurrencia, y mostraron que los estadios T y N, el alto grado y la invasión perineural se asociaban con metástasis a distancia y supervivencia [5]. Mahmood et al. asociaron la radioterapia adyuvante con una mejor supervivencia en tumores de glándulas salivales mayores de alto grado o localmente avanzados [7]. Schoenfeld et al. informaron que la radioterapia posoperatoria de intensidad modulada fue bien tolerada y logró un elevado control local [8], mientras que Hosni et al. destacaron que la metástasis a distancia seguía siendo un patrón dominante de fracaso en pacientes de alto riesgo pese a la radioterapia posoperatoria [9]. La revisión sistemática contemporánea de Wang et al. también respaldó la función locorregional de la radioterapia posoperatoria y subrayó la ausencia de evidencia global sólida a favor de añadir quimioterapia concurrente [10]. Las series específicas de parótida e histología refuerzan además el uso de radioterapia posoperatoria ante características adversas como márgenes positivos, alto grado y enfermedad T3/T4 [33-35]. En conjunto, estos estudios establecen la radioterapia como base locorregional del tratamiento, pero no demuestran que la quimioterapia añada un beneficio de supervivencia.

El segundo punto de referencia, más directo, es la literatura comparativa sobre quimiorradioterapia. Las primeras series institucionales sugirieron que la quimioterapia concurrente podría mejorar los resultados en pacientes seleccionados de alto riesgo [11,12]. Estos informes aportaron una justificación biológica y clínica importante para intensificar el tratamiento, pero el reducido tamaño muestral, los diseños no aleatorizados y la susceptibilidad al sesgo de selección limitaron la interpretación causal. Comparaciones institucionales posteriores no demostraron una ventaja clara de la quimiorradioterapia en la supervivencia global [13,16]. En pacientes de mayor edad, el análisis SEER-Medicare de Tanvetyanon et al. comunicó resultados que no favorecieron la intensificación con quimioterapia [14], y el amplio análisis de la NCDB de Amini et al. no encontró una ventaja de supervivencia global de la quimiorradioterapia adyuvante frente a la radioterapia sola [15]. Estos conjuntos de datos más grandes desplazaron el balance de la evidencia en contra del uso rutinario de quimioterapia, aunque la confusión residual siguió siendo inevitable.

Los estudios más recientes han seguido en gran medida la misma dirección, aunque sugieren que cualquier beneficio podría concentrarse en subgrupos biológicos o clinicopatológicos seleccionados. Kang et al. no encontraron beneficio en la supervivencia global ni específica de la enfermedad al añadir quimioterapia en el cáncer avanzado de glándulas salivales mayores [19]. Sin embargo, Hsieh et al. y Shen et al. describieron posibles señales en pacientes con enfermedad ganglionar, resección R2, carcinoma adenoide quístico o combinaciones de tumores T3/T4 de alto grado y gran carga ganglionar [20,21]. De forma similar, el estudio emparejado por puntuación de propensión de Hsieh et al. sobre carcinoma adenoide quístico sugirió una mejora del control locorregional sin una mejora correspondiente de la supervivencia global [17]. Esta distinción es clínicamente importante: un efecto radiosensibilizador puede mejorar el control local sin superar el riesgo de metástasis a distancia ni traducirse en una supervivencia más prolongada. Por tanto, la evidencia disponible no respalda la adición rutinaria de quimioterapia en una población de alto riesgo no seleccionada, pero tampoco excluye un beneficio en un subgrupo biológicamente enriquecido.

Nuestros hallazgos siguen la dirección dominante de la evidencia comparativa. La estimación primaria no favoreció quimioterapia, y tampoco exclusión de squamous carcinoma ni restricción a RTOG 1008-las histologías comparables revelaron una ventaja de supervivencia global. La predicción central es por tanto que la quimioterapia no mejorará supervivencia global mediana a través de una población ancha con adelantado T3/T4 enfermedad. Como en la literatura retrospectiva reciente, un beneficio limitó a un subgrupo biológico por poco seleccionado queda posible pero unproven.

La mayoría de contexto probable importante es RTOG 1008. Aquella prueba estuvo diseñada porque la evidencia retrospectiva era insuficiente y chocando: los estudios pequeños tempranos sugirieron beneficio posible [11,12], mientras que el registro más grande y las comparaciones institucionales fallaron para confirmar una ventaja de supervivencia compatible [13–16]. RTOG 1008 directamente compara postoperative radioterapia sólo con radioterapia más semanal cisplatin en resected alto-arriesgar tumores de glándula salivales malignos [22–24]. El GORTEC-REFCOR SANTAL el estudio de modo parecido evalúa radioterapia con o sin cisplatin en glándula salival y sinonasal tumores [32]. Junto, estas pruebas asoman que platino radiosensitization queda una estrategia de búsqueda verosímil, mientras ASCO y ESMO-EURACAN las directrices no endosan su uso rutinario fuera de pruebas clínicas [1,2]. Este análisis tiene que por tanto no ser enmarcado como sustituto para RTOG 1008. Su valor principal es falsabilidad probable:  proporciona un pre-modelo de resultados-predicción basada que más tarde puede ser comparado con randomized evidencia.

Si RTOG 1008 demuestra un beneficio de supervivencia clínicamente relevante, refutará la predicción actual e identificará dimensiones del efecto del tratamiento que el modelo de registro no captó. Si no demuestra beneficio, el resultado respaldará la predicción de que la quimioterapia no debe añadirse de forma rutinaria a la radioterapia posoperatoria fuera de subgrupos seleccionados o ensayos clínicos.

Este trabajo también destaca la diferencia entre la modelización pronóstica y la predictiva. Los modelos existentes estiman la recurrencia, la supervivencia posoperatoria, las metástasis a distancia o el pronóstico basal [25-31]. Sin embargo, un riesgo alto no implica automáticamente un beneficio de la quimioterapia. Al estimar la supervivencia con ambas estrategias de tratamiento en la misma población, este análisis se dirige directamente al efecto incremental de la quimioterapia y no solo al pronóstico. Su predicción clínicamente comprobable es sencilla: añadir quimioterapia a la radioterapia no mejorará la supervivencia global promedio en el cáncer de glándulas salivales T3/T4 avanzado.

## Limitaciones

Este análisis de registro sigue siendo susceptible a confusión residual, en particular por factores de riesgo patológicos, estado funcional, momento del tratamiento y detalles terapéuticos no incluidos en el extracto. El diagnóstico fue el origen temporal disponible, no pudo confirmarse la concurrencia de los tratamientos y las estimaciones específicas por histología fueron imprecisas. El modelo de Cox también impone riesgos proporcionales y la validación fue interna. Estas limitaciones afectan la certeza causal y la resolución de los subgrupos, pero no modifican la predicción poblacional preespecificada del estudio.

## Conclusión

El modelo predice que añadir quimioterapia a la radioterapia no mejora la supervivencia global promedio en el cáncer de glándulas salivales T3/T4 avanzado. La diferencia ajustada del RMST a 10 años fue de -2,3 meses, sin un efecto favorable a 5 ni a 10 años. Esta predicción falsable, formulada antes de conocer los resultados, queda a la espera de su comparación con RTOG 1008.

## Referencias

1. Geiger JL, Ismaila N, Beadle B, Caudell JJ, Chau N, Deschler D, et al. Management of Salivary Gland Malignancy: ASCO Guideline. J Clin Oncol. 2021;39:1909-1941. doi:10.1200/JCO.21.00449.
2. van Herpen C, Locati LD, But-Hadzic J, Bossi P, Cavalieri S, Licitra L, et al. Salivary gland cancer: ESMO-EURACAN Clinical Practice Guideline for diagnosis, treatment and follow-up. ESMO Open. 2022. PMID:36567082.
3. PDQ Adult Treatment Editorial Board. Salivary Gland Cancer Treatment (PDQ). National Cancer Institute; updated 2025.
4. Cerda T, Sun XS, Vignot S, et al. A rationale for chemoradiation versus radiotherapy in salivary gland cancers? Crit Rev Oncol Hematol. 2014;91:142-158. PMID:24636481.
5. Terhaard CHJ, Lubsen H, van der Tweel I, Hilgers FJM, Eijkenboom WMH, Marres HAM, et al. Salivary gland carcinoma: independent prognostic factors for locoregional control, distant metastases, and overall survival: results of the Dutch head and neck oncology cooperative group. Head Neck. 2004;26:681-693. PMID:15287035.
6. Mendenhall WM, Morris CG, Amdur RJ, Werning JW, Hinerman RW, Villaret DB. Radiotherapy alone or combined with surgery for salivary gland carcinoma. Cancer. 2005;103:2544-2550. PMID:15880750.
7. Mahmood U, Koshy M, Goloubeva O, Suntharalingam M. Adjuvant radiation therapy for high-grade and/or locally advanced major salivary gland tumors. Arch Otolaryngol Head Neck Surg. 2011. PMID:22006781.
8. Schoenfeld JD, Sher DJ, Norris CM Jr, Haddad RI, Posner MR, Balboni TA, et al. Salivary gland tumors treated with adjuvant intensity-modulated radiotherapy with or without concurrent chemotherapy. Int J Radiat Oncol Biol Phys. 2012;82:308-314. PMID:21075557.
9. Hosni A, Huang SH, Goldstein D, Xu W, Chan B, Hansen A, et al. Outcomes and prognostic factors for major salivary gland carcinoma following postoperative radiotherapy. Oral Oncol. 2016;54:75-80. PMID:26723908.
10. Wang J, et al. The Current Position of Postoperative Radiotherapy for Salivary Gland Cancer: A Systematic Review and Meta-Analysis. Cancers. 2024;16:2375.
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
25. Lukovic J, Sultana R, et al. Development and validation of a clinical prediction-score model for distant metastases in salivary gland cancer. Oral Oncol. 2020. PMID:31959347.
26. Ali S, Palmer FL, Yu C, DiLorenzo M, Shah JP, Kattan MW, et al. A predictive nomogram for recurrence of carcinoma of the major salivary glands. JAMA Otolaryngol Head Neck Surg. 2013;139:698-705. doi:10.1001/jamaoto.2013.3347.
27. Ali S, Palmer FL, Yu C, DiLorenzo M, Shah JP, Kattan MW, Patel SG, Ganly I. Postoperative nomograms predictive of survival after surgical management of malignant tumors of the major salivary glands. Ann Surg Oncol. 2014;21:637-642. PMID:24132626.
28. Hay A, Migliacci J, Karassawa Zanoni D, et al. Validation of nomograms for overall survival, cancer-specific survival and recurrence in major salivary gland cancer. Head Neck. 2018. PMID:29389040.
29. Chen Y, Li Y, et al. Prognostic risk factor of major salivary gland carcinomas and survival prediction model based on random survival forests. Cancer Med. 2023. PMID:36934429.
30. Du W, et al. Prognostic prediction model for salivary gland carcinoma based on machine learning. 2024. PMID:38981745.
31. Jacobs CD, et al. Prediction model to estimate overall survival benefit of postoperative radiation therapy for resected major salivary gland cancers. Oral Oncol. 2022. doi:10.1016/j.oraloncology.2022.105902.
32. GORTEC/REFCOR. Treatment of Salivary Glands and Nasal Tumors (SANTAL). ClinicalTrials.gov Identifier: NCT02998385.
33. Kim YH, et al. Evaluation of prognostic factors for the parotid cancer treated with surgery and postoperative radiotherapy. Cancer Res Treat. 2020. PMID:31480828.
34. Park G, et al. Postoperative radiotherapy for mucoepidermoid carcinoma of major salivary glands: long-term results of a single-institution experience. Radiat Oncol J. 2018. PMID:30630270.
35. Katano A, et al. Postoperative radiotherapy for malignant major salivary gland tumors. 2023.
