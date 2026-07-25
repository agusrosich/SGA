# Disponibilidad de variables y límites de la emulación

## Variables disponibles en la extracción analítica

La extracción contiene edad agrupada, sexo, código histológico ICD-O-3, grupo histológico armonizado, categorías T y N armonizadas, indicadores binarios de radioterapia y quimioterapia, tiempo desde el diagnóstico hasta el primer tratamiento, causa de muerte recodificada y meses de supervivencia.

## Criterios que pueden implementarse

- Restricción a T3/T4, categoría N conocida y radioterapia registrada.
- Comparación entre quimioterapia registrada y ausencia de quimioterapia registrada.
- Supervivencia global con censura de pacientes vivos en el último seguimiento.
- Modelo secundario separado de mortalidad cáncer-específica.
- Propensity score, evaluación de positividad, IPTW, balance y modelo de resultado ajustado.
- ATE expresado como diferencia de RMST a 10 años y diferencia absoluta de supervivencia a 5 y 10 años.
- Análisis sin carcinoma escamoso y restringido a histologías comparables con RTOG 1008.
- Validación interna, bootstrap integral y simulaciones secundarias repetidas de 252 pacientes.

## Criterios no verificables con esta extracción

No están disponibles variables de cirugía, intención postoperatoria de la radioterapia, enfermedad metastásica, año o fecha de diagnóstico, fechas separadas de cirugía/radioterapia/quimioterapia, concurrencia terapéutica, agente o dosis de quimioterapia, dosis de radioterapia, sitio de glándula mayor frente a menor, grado, márgenes, extensión extranodal (ENE), invasión perineural (PNI) ni estado funcional.

En consecuencia:

- No puede confirmarse que todos los pacientes hayan sido operados.
- No puede confirmarse que la radioterapia sea postoperatoria.
- No puede excluirse directamente enfermedad metastásica al diagnóstico.
- No puede establecerse un tiempo cero en cirugía o inicio de radioterapia; el origen disponible es el diagnóstico.
- No puede verificarse la secuencia ni la concurrencia de los tratamientos.
- No puede identificarse cisplatino semanal ni ningún agente concreto.
- No puede restringirse el análisis a tumores de glándulas salivales mayores.
- No puede reproducirse estrictamente la elegibilidad de RTOG 1008.

Por estos motivos, el trabajo debe describirse como un análisis observacional **RTOG 1008-like** y no como un ensayo aleatorizado ni como una emulación estricta del ensayo objetivo. La exposición debe nombrarse “radioterapia con quimioterapia registrada” frente a “radioterapia sin quimioterapia registrada”; no debe denominarse quimiorradioterapia concurrente.
