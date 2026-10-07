from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

import requests
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "manuscrito_final" / "manuscript.md"
OUT_DIR = ROOT / "output" / "pdf"
OUT_PDF = OUT_DIR / "manuscrito_supervivencia_glandulas_salivales_es.pdf"
TRANSLATED_MD = ROOT / "tmp" / "pdfs" / "manuscript_es.md"
CACHE_FILE = ROOT / "tmp" / "pdfs" / "translation_cache.json"


FIXED = {
    "Predictive modeling of outcomes in salivary gland cancers using machine learning: simulated prospective validation with the RTOG 1008 trial": "Modelización predictiva de resultados en cánceres de glándulas salivales mediante aprendizaje automático: validación prospectiva simulada con el ensayo RTOG 1008",
    "Abstract": "Resumen",
    "Background": "Antecedentes",
    "Methods": "Métodos",
    "Results": "Resultados",
    "Conclusions": "Conclusiones",
    "Introduction": "Introducción",
    "Data source and implementable eligibility": "Fuente de datos y criterios de elegibilidad aplicables",
    "Treatment, endpoint, and estimand": "Tratamiento, desenlace y estimando",
    "Confounding adjustment, positivity, and balance": "Ajuste por confusión, positividad y balance",
    "Bootstrap and internal validation": "Bootstrap y validación interna",
    "Sensitivity and secondary analyses": "Análisis de sensibilidad y secundarios",
    "Cohort selection and observed treatment groups": "Selección de la cohorte y grupos de tratamiento observados",
    "Positivity and covariate balance": "Positividad y balance de covariables",
    "Primary effect estimate": "Estimación del efecto primario",
    "Model performance": "Rendimiento del modelo",
    "Illustrative N=252 simulations": "Simulaciones ilustrativas con N=252",
    "Discussion": "Discusión",
    "Limitations": "Limitaciones",
    "Conclusion": "Conclusión",
    "References": "Referencias",
    "Estimand": "Estimando",
    "RMST difference": "Diferencia de RMST",
    "N category: N0": "Categoría N: N0",
    "N category: N1": "Categoría N: N1",
    "N category: N2": "Categoría N: N2",
    "N category: N3": "Categoría N: N3",
    "−2.3 months": "-2,3 meses",
    "−4.1 months": "-4,1 meses",
    "−5.4 months": "-5,4 meses",
    "IPCW Brier score, cross-validated mean": "Puntuación de Brier con IPCW, media con validación cruzada",
    "Calibration error, cross-validated mean": "Error de calibración, media con validación cruzada",
}

MANUAL = {
    "Whether chemotherapy improves survival when added to radiotherapy in high-risk salivary gland cancer remains unresolved. We developed a falsifiable pre-results model predicting the average survival effect of chemotherapy in advanced T3/T4 disease for future comparison with RTOG 1008.": "No se ha establecido si la adición de quimioterapia a la radioterapia mejora la supervivencia en el cáncer de glándulas salivales de alto riesgo. Desarrollamos un modelo falsable, previo a la publicación de resultados, que predice el efecto promedio de la quimioterapia sobre la supervivencia en enfermedad T3/T4 avanzada, para compararlo en el futuro con RTOG 1008.",
    "We analyzed a standardized SEER-derived cohort of T3/T4 tumors with known N category and radiotherapy. Overall survival was modeled with censoring. The prespecified estimand was the average treatment effect (ATE), expressed as the difference in restricted mean survival time (RMST) through 120 months for RT + chemotherapy versus RT only. Propensity-score weighting and a covariate-adjusted Cox model were used to address measured treatment-selection differences. Both potential outcomes were standardized to the same population, and the full analysis was repeated in 1,000 patient-level bootstrap samples. Internal performance was assessed by five-fold cross-validation.": "Analizamos una cohorte estandarizada derivada de SEER de tumores T3/T4, con categoría N conocida y radioterapia registrada. La supervivencia global se modelizó considerando la censura. El estimando preespecificado fue el efecto promedio del tratamiento (ATE), expresado como la diferencia en el tiempo medio de supervivencia restringida (RMST) hasta 120 meses entre RT más quimioterapia y solo RT. Se utilizaron ponderación por puntuación de propensión y un modelo de Cox ajustado por covariables para abordar las diferencias medidas en la selección del tratamiento. Ambos resultados potenciales se estandarizaron a la misma población y el análisis completo se repitió en 1.000 muestras bootstrap a nivel de paciente. El rendimiento interno se evaluó mediante validación cruzada de cinco pliegues.",
    "Of 4,657 source records, 954 met the eligibility criteria; 283 received RT + chemotherapy and 671 received RT only. Adjusted 10-year RMST was 64.8 months with RT + chemotherapy and 67.1 months with RT only. The ATE was −2.3 months (95% bootstrap CI, −9.6 to +6.0; P=0.580). Adjusted absolute survival differences were −2.2 percentage points at both 5 and 10 years. Median follow-up by reverse Kaplan–Meier was 86 months. Mean cross-validated C-index was 0.641; IPCW Brier scores were 0.215 at 5 years and 0.188 at 10 years.": "De 4.657 registros de origen, 954 cumplieron los criterios de elegibilidad; 283 recibieron RT más quimioterapia y 671 recibieron solo RT. El RMST ajustado a 10 años fue de 64,8 meses con RT más quimioterapia y de 67,1 meses con solo RT. El ATE fue de -2,3 meses (IC del 95% por bootstrap, -9,6 a +6,0; P=0,580). Las diferencias absolutas ajustadas de supervivencia fueron de -2,2 puntos porcentuales tanto a 5 como a 10 años. La mediana de seguimiento mediante Kaplan-Meier inversa fue de 86 meses. El índice C medio con validación cruzada fue de 0,641; las puntuaciones de Brier con IPCW fueron de 0,215 a 5 años y de 0,188 a 10 años.",
    "The model predicted no average overall-survival benefit from adding chemotherapy to radiotherapy in advanced T3/T4 disease. The estimate is consistent with the retrospective evidence against routine chemotherapy intensification and provides a pre-results prediction for prospective testing against RTOG 1008.": "El modelo predijo que la adición de quimioterapia a la radioterapia no aporta un beneficio promedio en la supervivencia global de pacientes con enfermedad T3/T4 avanzada. La estimación concuerda con la evidencia retrospectiva contraria a la intensificación rutinaria con quimioterapia y ofrece una predicción previa a los resultados para su evaluación prospectiva frente a RTOG 1008.",
    "In this RTOG 1008-like analysis of advanced T3/T4 salivary gland cancer, the model predicted no average overall-survival benefit from adding chemotherapy to radiotherapy. After adjustment for censoring and measured treatment-selection differences, the 10-year RMST difference was −2.3 months (95% bootstrap CI, −9.6 to +6.0; P=0.580), and the absolute survival differences at 5 and 10 years were both −2.2 percentage points. The direction and magnitude of these estimates argue against a substantial population-wide benefit from chemotherapy intensification. This prediction is consistent with the current guideline position: postoperative radiotherapy is established for adverse features, whereas routine concurrent chemotherapy remains unsupported outside a clinical trial [1,2]. RTOG 1008 exists precisely because prospective efficacy evidence has been lacking [22–24].": "En este análisis similar a RTOG 1008 de cáncer de glándulas salivales T3/T4 avanzado, el modelo predijo que añadir quimioterapia a la radioterapia no produce un beneficio promedio en la supervivencia global. Tras ajustar por censura y por las diferencias medidas en la selección del tratamiento, la diferencia de RMST a 10 años fue de -2,3 meses (IC del 95% por bootstrap, -9,6 a +6,0; P=0,580), y las diferencias absolutas de supervivencia a 5 y 10 años fueron de -2,2 puntos porcentuales en ambos casos. La dirección y la magnitud de estas estimaciones no respaldan un beneficio sustancial de la intensificación con quimioterapia en el conjunto de la población. Esta predicción coincide con las guías actuales: la radioterapia posoperatoria está establecida ante características adversas, mientras que la quimioterapia concurrente rutinaria no está respaldada fuera de un ensayo clínico [1,2]. RTOG 1008 existe precisamente por la falta de evidencia prospectiva de eficacia [22-24].",
    "The first anchor for interpretation is the postoperative radiotherapy literature. Terhaard et al. identified T3/T4 disease and incomplete resection as adverse factors for recurrence and showed that T and N stage, high grade, and perineural invasion were associated with distant metastasis and survival [5]. Mahmood et al. associated adjuvant radiotherapy with improved survival in high-grade and locally advanced major salivary gland tumors [7]. Schoenfeld et al. reported that postoperative intensity-modulated radiotherapy was well tolerated and achieved high local control [8], while Hosni et al. emphasized that distant metastasis remained a dominant pattern of failure in high-risk patients despite postoperative radiotherapy [9]. The contemporary systematic review by Wang et al. likewise supported the locoregional role of postoperative radiotherapy while underscoring the absence of strong global evidence for adding concurrent chemotherapy [10]. Parotid- and histology-specific series further reinforce the use of postoperative radiotherapy for adverse features such as positive margins, high grade, and T3/T4 disease [33–35]. Taken together, these studies establish radiotherapy as the locoregional backbone of treatment but do not establish that chemotherapy adds a survival benefit.": "El primer punto de referencia para la interpretación es la literatura sobre radioterapia posoperatoria. Terhaard et al. identificaron la enfermedad T3/T4 y la resección incompleta como factores adversos para la recurrencia, y mostraron que los estadios T y N, el alto grado y la invasión perineural se asociaban con metástasis a distancia y supervivencia [5]. Mahmood et al. asociaron la radioterapia adyuvante con una mejor supervivencia en tumores de glándulas salivales mayores de alto grado o localmente avanzados [7]. Schoenfeld et al. informaron que la radioterapia posoperatoria de intensidad modulada fue bien tolerada y logró un elevado control local [8], mientras que Hosni et al. destacaron que la metástasis a distancia seguía siendo un patrón dominante de fracaso en pacientes de alto riesgo pese a la radioterapia posoperatoria [9]. La revisión sistemática contemporánea de Wang et al. también respaldó la función locorregional de la radioterapia posoperatoria y subrayó la ausencia de evidencia global sólida a favor de añadir quimioterapia concurrente [10]. Las series específicas de parótida e histología refuerzan además el uso de radioterapia posoperatoria ante características adversas como márgenes positivos, alto grado y enfermedad T3/T4 [33-35]. En conjunto, estos estudios establecen la radioterapia como base locorregional del tratamiento, pero no demuestran que la quimioterapia añada un beneficio de supervivencia.",
    "The second and more direct anchor is the comparative chemoradiotherapy literature. Early institutional series suggested that concurrent chemotherapy might improve outcomes in selected high-risk patients [11,12]. These reports provided an important biological and clinical rationale for treatment intensification, but their small sample sizes, nonrandomized designs, and susceptibility to treatment-selection bias limited causal interpretation. Subsequent institutional comparisons did not demonstrate a clear overall-survival advantage for chemoradiotherapy [13,16]. In older patients, the SEER-Medicare analysis by Tanvetyanon et al. reported outcomes that did not favor chemotherapy intensification [14], and the large NCDB analysis by Amini et al. found no overall-survival advantage for adjuvant chemoradiotherapy over radiotherapy alone [15]. These larger datasets shifted the balance of evidence away from routine chemotherapy use, although residual confounding remained unavoidable.": "El segundo punto de referencia, más directo, es la literatura comparativa sobre quimiorradioterapia. Las primeras series institucionales sugirieron que la quimioterapia concurrente podría mejorar los resultados en pacientes seleccionados de alto riesgo [11,12]. Estos informes aportaron una justificación biológica y clínica importante para intensificar el tratamiento, pero el reducido tamaño muestral, los diseños no aleatorizados y la susceptibilidad al sesgo de selección limitaron la interpretación causal. Comparaciones institucionales posteriores no demostraron una ventaja clara de la quimiorradioterapia en la supervivencia global [13,16]. En pacientes de mayor edad, el análisis SEER-Medicare de Tanvetyanon et al. comunicó resultados que no favorecieron la intensificación con quimioterapia [14], y el amplio análisis de la NCDB de Amini et al. no encontró una ventaja de supervivencia global de la quimiorradioterapia adyuvante frente a la radioterapia sola [15]. Estos conjuntos de datos más grandes desplazaron el balance de la evidencia en contra del uso rutinario de quimioterapia, aunque la confusión residual siguió siendo inevitable.",
    "More recent studies have largely followed the same direction while suggesting that any benefit may be concentrated in selected biological or clinicopathologic subsets. Kang et al. found no overall- or disease-specific-survival benefit from adding chemotherapy in advanced major salivary gland cancer [19]. Hsieh et al. and Shen et al., however, described possible signals in patients with nodal disease, R2 resection, adenoid cystic carcinoma, or combinations of T3/T4 high-grade tumors and heavy nodal burden [20,21]. Similarly, the propensity-matched adenoid cystic carcinoma study by Hsieh et al. suggested improved locoregional control without a corresponding overall-survival improvement [17]. This distinction is clinically important: a radiosensitizing effect may improve local control without overcoming distant metastatic risk or translating into longer survival. The available evidence therefore does not support the routine addition of chemotherapy across an unselected high-risk population, but it also does not exclude benefit in a biologically enriched subgroup.": "Los estudios más recientes han seguido en gran medida la misma dirección, aunque sugieren que cualquier beneficio podría concentrarse en subgrupos biológicos o clinicopatológicos seleccionados. Kang et al. no encontraron beneficio en la supervivencia global ni específica de la enfermedad al añadir quimioterapia en el cáncer avanzado de glándulas salivales mayores [19]. Sin embargo, Hsieh et al. y Shen et al. describieron posibles señales en pacientes con enfermedad ganglionar, resección R2, carcinoma adenoide quístico o combinaciones de tumores T3/T4 de alto grado y gran carga ganglionar [20,21]. De forma similar, el estudio emparejado por puntuación de propensión de Hsieh et al. sobre carcinoma adenoide quístico sugirió una mejora del control locorregional sin una mejora correspondiente de la supervivencia global [17]. Esta distinción es clínicamente importante: un efecto radiosensibilizador puede mejorar el control local sin superar el riesgo de metástasis a distancia ni traducirse en una supervivencia más prolongada. Por tanto, la evidencia disponible no respalda la adición rutinaria de quimioterapia en una población de alto riesgo no seleccionada, pero tampoco excluye un beneficio en un subgrupo biológicamente enriquecido.",
    "If RTOG 1008 demonstrates a clinically meaningful survival benefit, it will falsify the present prediction and identify dimensions of treatment effect not captured by the registry model. If it does not demonstrate benefit, the result will support the prediction that chemotherapy should not be routinely added to postoperative radiotherapy outside selected subgroups or clinical trials.": "Si RTOG 1008 demuestra un beneficio de supervivencia clínicamente relevante, refutará la predicción actual e identificará dimensiones del efecto del tratamiento que el modelo de registro no captó. Si no demuestra beneficio, el resultado respaldará la predicción de que la quimioterapia no debe añadirse de forma rutinaria a la radioterapia posoperatoria fuera de subgrupos seleccionados o ensayos clínicos.",
    "This work also highlights the distinction between prognostic and predictive modeling. Existing models estimate recurrence, postoperative survival, distant metastasis, or baseline prognosis [25–31]. High risk, however, does not automatically imply chemotherapy benefit. By estimating survival under both treatment strategies in the same population, this analysis directly targets the incremental effect of chemotherapy rather than prognosis alone. Its clinically testable prediction is straightforward: adding chemotherapy to radiotherapy will not improve average overall survival in advanced T3/T4 salivary gland cancer.": "Este trabajo también destaca la diferencia entre la modelización pronóstica y la predictiva. Los modelos existentes estiman la recurrencia, la supervivencia posoperatoria, las metástasis a distancia o el pronóstico basal [25-31]. Sin embargo, un riesgo alto no implica automáticamente un beneficio de la quimioterapia. Al estimar la supervivencia con ambas estrategias de tratamiento en la misma población, este análisis se dirige directamente al efecto incremental de la quimioterapia y no solo al pronóstico. Su predicción clínicamente comprobable es sencilla: añadir quimioterapia a la radioterapia no mejorará la supervivencia global promedio en el cáncer de glándulas salivales T3/T4 avanzado.",
    "The separate cancer-specific cause-specific model estimated a 10-year RMST difference of −3.7 months. It is secondary and does not replace the overall-survival endpoint.": "El modelo secundario de riesgos específicos por causa para la mortalidad por cáncer estimó una diferencia de RMST a 10 años de -3,7 meses. Este análisis es secundario y no sustituye el desenlace de supervivencia global.",
    "Across 1,000 secondary simulations, the mean RMST difference was −2.4 months, with individual simulated estimates ranging from −10.6 to +7.9 months. The average maximum absolute SMD across age, sex, T4, and nodal status was 0.185, demonstrating why a single random sample can appear imbalanced. These simulations are illustrative and are not the primary result.": "En 1.000 simulaciones secundarias, la diferencia media del RMST fue de -2,4 meses, con estimaciones individuales entre -10,6 y +7,9 meses. La media del SMD absoluto máximo para edad, sexo, T4 y estado ganglionar fue de 0,185, lo que demuestra por qué una única muestra aleatoria puede parecer desequilibrada. Estas simulaciones son ilustrativas y no constituyen el resultado primario.",
    "This registry analysis remains susceptible to residual confounding, particularly from pathologic risk factors, performance status, treatment timing, and treatment details not captured in the extract. Diagnosis was the available time origin, treatment concurrency could not be confirmed, and histology-specific estimates were sparse. The Cox model also imposes proportional hazards, and validation was internal. These limitations affect causal certainty and subgroup resolution but do not change the study's prespecified population-level prediction.": "Este análisis de registro sigue siendo susceptible a confusión residual, en particular por factores de riesgo patológicos, estado funcional, momento del tratamiento y detalles terapéuticos no incluidos en el extracto. El diagnóstico fue el origen temporal disponible, no pudo confirmarse la concurrencia de los tratamientos y las estimaciones específicas por histología fueron imprecisas. El modelo de Cox también impone riesgos proporcionales y la validación fue interna. Estas limitaciones afectan la certeza causal y la resolución de los subgrupos, pero no modifican la predicción poblacional preespecificada del estudio.",
    "The model predicts that adding chemotherapy to radiotherapy does not improve average overall survival in advanced T3/T4 salivary gland cancer. The adjusted 10-year RMST difference was −2.3 months, with no favorable effect at 5 or 10 years. This falsifiable pre-results prediction now awaits comparison with RTOG 1008.": "El modelo predice que añadir quimioterapia a la radioterapia no mejora la supervivencia global promedio en el cáncer de glándulas salivales T3/T4 avanzado. La diferencia ajustada del RMST a 10 años fue de -2,3 meses, sin un efecto favorable a 5 ni a 10 años. Esta predicción falsable, formulada antes de conocer los resultados, queda a la espera de su comparación con RTOG 1008.",
}


def clean_translation(text: str) -> str:
    replacements = {
        "radiación": "radioterapia",
        "Radiación": "Radioterapia",
        "quimiorradiación": "quimiorradioterapia",
        "quimio-radioterapia": "quimiorradioterapia",
        "tiempo medio de supervivencia restringido": "tiempo medio de supervivencia restringida",
        "supervivencia media restringida": "supervivencia media restringida",
        "puntaje de propensión": "puntuación de propensión",
        "puntajes de propensión": "puntuaciones de propensión",
        "puntuación Brier": "puntuación de Brier",
        "Kaplan-Meier inverso": "Kaplan-Meier inversa",
        "RT solamente": "solo RT",
        "RT sola": "solo RT",
        "IC del 95 %": "IC del 95%",
        "valor P": "valor de P",
        "Valores P": "Valores de P",
        "supervivencia general": "supervivencia global",
        "punto final": "desenlace",
        "muestras de arranque": "muestras bootstrap",
        "estimaciones de arranque": "estimaciones bootstrap",
        "error estándar de arranque": "error estándar bootstrap",
        "radioterapia grabada": "radioterapia registrada",
        "tiempo de supervivencia no perdido": "tiempo de supervivencia no faltante",
        "antifactual": "contrafactual",
        "covariación": "covariables",
        "covariaciones": "covariables",
        "chi-quare": "chi-cuadrado",
        "nodo positivo": "ganglios positivos",
        "TENSMD habit": "|SMD|",
        "gray": "gris",
        "pestañas": "guiones",
        "línea vertical desgarrada": "línea vertical discontinua",
        "Cuadro ": "Tabla ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = text.replace("*", "").replace("@", "")
    return text


def translate_text(text: str, cache: dict[str, str]) -> str:
    stripped = text.strip()
    if not stripped:
        return text
    if stripped in FIXED:
        return FIXED[stripped]
    if stripped in MANUAL:
        return MANUAL[stripped]
    if stripped in cache:
        return clean_translation(cache[stripped])
    raise KeyError(f"Missing batched translation: {stripped[:80]}")


def populate_translation_cache(lines: list[str], cache: dict[str, str]) -> None:
    segments: list[str] = []
    in_references = False
    for line in lines:
        stripped = line.strip()
        if stripped == "## References":
            in_references = True
            continue
        if in_references or not stripped or stripped.startswith("Authors:"):
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        image_ref = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        candidates: list[str]
        if heading:
            candidates = [heading.group(2)]
        elif image_ref:
            candidates = [image_ref.group(1)]
        elif stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r"\s*:?-{3,}:?\s*", c) for c in cells):
                continue
            candidates = [c for c in cells if c and not re.fullmatch(r"[<>=+−–—\d.,%()\s]+", c)]
        else:
            candidates = [stripped]
        for candidate in candidates:
            if candidate not in FIXED and candidate not in cache and candidate not in segments:
                segments.append(candidate)

    # LibreTranslate accepts an array for q. Keep each request below a conservative payload size.
    batches: list[list[str]] = []
    batch: list[str] = []
    chars = 0
    for segment in segments:
        if batch and chars + len(segment) > 7500:
            batches.append(batch)
            batch, chars = [], 0
        batch.append(segment)
        chars += len(segment)
    if batch:
        batches.append(batch)

    for batch_index, values in enumerate(batches):
        combined = values[0]
        for idx, value in enumerate(values[1:], start=1):
            combined += f"\n\n|||||{idx:04d}|||||\n\n{value}"
        response = requests.post(
            "https://beta.apertium.org/apy/translate",
            data={"q": combined, "langpair": "eng|spa"},
            timeout=240,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Translation service error {response.status_code}: {response.text[:500]}")
        translated_text = response.json()["responseData"]["translatedText"]
        translated = re.split(r"\|{3,}\s*\d{4}\s*\|{3,}", translated_text)
        translated = [item.strip() for item in translated]
        if len(translated) != len(values):
            raise RuntimeError(f"Apertium returned an unexpected batch size: {len(translated)} for {len(values)}")
        for source, target in zip(values, translated):
            cache[source] = clean_translation(target)
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def translate_markdown() -> str:
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    populate_translation_cache(lines, cache)
    result: list[str] = []
    in_references = False
    for line in lines:
        stripped = line.strip()
        if stripped == "## References":
            result.append("## Referencias")
            in_references = True
            continue
        if in_references:
            result.append(line)
            continue
        if not stripped:
            result.append("")
            continue
        if stripped.startswith("Authors:"):
            result.append("Autores:" + stripped[len("Authors:"):])
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            result.append(f"{heading.group(1)} {translate_text(heading.group(2), cache)}")
            continue
        image_ref = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if image_ref:
            result.append(f"![{translate_text(image_ref.group(1), cache)}]({image_ref.group(2)})")
            continue
        if stripped.startswith("|"):
            cells = stripped.strip("|").split("|")
            if all(re.fullmatch(r"\s*:?-{3,}:?\s*", c) for c in cells):
                result.append(line)
            else:
                translated_cells = []
                for cell in cells:
                    value = cell.strip()
                    if not value or re.fullmatch(r"[<>=+−–—\d.,%()\s]+", value):
                        translated_cells.append(value)
                    else:
                        translated_cells.append(translate_text(value, cache))
                result.append("| " + " | ".join(translated_cells) + " |")
            continue
        result.append(translate_text(stripped, cache))
    output = "\n".join(result) + "\n"
    TRANSLATED_MD.write_text(output, encoding="utf-8")
    return output


def esc(text: str) -> str:
    text = html.escape(text, quote=False)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return text


class ManuscriptDoc(BaseDocTemplate):
    def __init__(self, filename: str, title: str):
        super().__init__(
            filename,
            pagesize=LETTER,
            leftMargin=0.78 * inch,
            rightMargin=0.78 * inch,
            topMargin=0.76 * inch,
            bottomMargin=0.72 * inch,
            title=title,
            author="Federico Lorenzo et al.",
            subject="Versión en español del manuscrito científico",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self._decorate))

    def _decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFont("Arial", 7.5)
            canvas.setFillColor(colors.HexColor("#666666"))
            canvas.drawString(self.leftMargin, LETTER[1] - 0.42 * inch, "Modelización predictiva en cánceres de glándulas salivales")
        canvas.setFont("Arial", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawCentredString(LETTER[0] / 2, 0.38 * inch, str(doc.page))
        canvas.restoreState()


def register_fonts() -> None:
    font_dir = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("Arial", str(font_dir / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(font_dir / "arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Italic", str(font_dir / "ariali.ttf")))


def make_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleES", parent=styles["Title"], fontName="Arial-Bold", fontSize=17, leading=20, alignment=TA_CENTER, textColor=colors.black, spaceAfter=12),
        "authors": ParagraphStyle("AuthorsES", parent=styles["Normal"], fontName="Arial", fontSize=9, leading=12, alignment=TA_CENTER, spaceAfter=8),
        "affil": ParagraphStyle("AffilES", parent=styles["Normal"], fontName="Arial-Italic", fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=15),
        "h2": ParagraphStyle("H2ES", parent=styles["Heading2"], fontName="Arial-Bold", fontSize=13, leading=16, textColor=colors.black, spaceBefore=14, spaceAfter=6, keepWithNext=True),
        "h3": ParagraphStyle("H3ES", parent=styles["Heading3"], fontName="Arial-Bold", fontSize=10.5, leading=13, textColor=colors.black, spaceBefore=10, spaceAfter=4, keepWithNext=True),
        "body": ParagraphStyle("BodyES", parent=styles["BodyText"], fontName="Arial", fontSize=9.2, leading=12.4, alignment=TA_JUSTIFY, textColor=colors.black, spaceAfter=6),
        "caption": ParagraphStyle("CaptionES", parent=styles["BodyText"], fontName="Arial", fontSize=7.8, leading=10, alignment=TA_LEFT, textColor=colors.HexColor("#333333"), spaceBefore=4, spaceAfter=9),
        "table": ParagraphStyle("TableES", parent=styles["BodyText"], fontName="Arial", fontSize=7.2, leading=8.8, alignment=TA_LEFT),
        "table_head": ParagraphStyle("TableHeadES", parent=styles["BodyText"], fontName="Arial-Bold", fontSize=7.1, leading=8.5, alignment=TA_CENTER, textColor=colors.white),
        "refs": ParagraphStyle("RefsES", parent=styles["BodyText"], fontName="Arial", fontSize=7.7, leading=9.6, alignment=TA_LEFT, spaceAfter=3),
    }


def table_widths(rows: list[list[str]], width: float) -> list[float]:
    n = len(rows[0])
    if n == 3:
        return [width * 0.52, width * 0.24, width * 0.24]
    if n == 5:
        return [width * 0.31, width * 0.16, width * 0.16, width * 0.26, width * 0.11]
    return [width / n] * n


def build_pdf(markdown: str) -> None:
    register_fonts()
    styles = make_styles()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = ManuscriptDoc(str(OUT_PDF), "Modelización predictiva de resultados en cánceres de glándulas salivales")
    story = []
    lines = markdown.splitlines()
    i = 0
    in_refs = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(esc(line[2:]), styles["title"]))
            story.append(Paragraph("Federico Lorenzo; Agustin Rosich; Jesica Lell; Sergio Aguiar; Valentina Ferreira; Karina Ochandorena; Eduardo Larrinaga; Natalia Gadea; Nicolas Larragueta; Aldo Quarneti", styles["authors"]))
            story.append(Paragraph("¹ Radioterapia, RT International Institute, Montevideo, Uruguay. ² Radioterapia, Unidad Académica de Radioterapia, Montevideo, Uruguay, RT International Institute.", styles["affil"]))
        elif line.startswith("## "):
            heading = line[3:]
            if heading == "Referencias":
                story.append(PageBreak())
                in_refs = True
            story.append(Paragraph(esc(heading), styles["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(esc(line[4:]), styles["h3"]))
        elif line.startswith("Autores:"):
            pass
        elif line.startswith("|"):
            raw_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw_rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            if len(raw_rows) > 1 and all(re.fullmatch(r":?-{3,}:?", c) for c in raw_rows[1]):
                raw_rows.pop(1)
            pdata = []
            for r, row in enumerate(raw_rows):
                style = styles["table_head"] if r == 0 else styles["table"]
                pdata.append([Paragraph(esc(c), style) for c in row])
            table = Table(pdata, colWidths=table_widths(raw_rows, doc.width), repeatRows=1, hAlign="CENTER")
            commands = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244A64")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9D9D9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
            for r in range(1, len(pdata)):
                if r % 2 == 0:
                    commands.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F2F6F8")))
            table.setStyle(TableStyle(commands))
            story.extend([Spacer(1, 4), table, Spacer(1, 9)])
            continue
        else:
            image_ref = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", line)
            if image_ref:
                img_path = SOURCE.parent / image_ref.group(2)
                img = Image(str(img_path))
                max_w, max_h = doc.width, 4.8 * inch
                scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
                img.drawWidth = img.imageWidth * scale
                img.drawHeight = img.imageHeight * scale
                img.hAlign = "CENTER"
                story.append(KeepTogether([img, Paragraph(esc(image_ref.group(1)), styles["caption"])]))
            else:
                style = styles["refs"] if in_refs else styles["body"]
                story.append(Paragraph(esc(line), style))
        i += 1
    doc.build(story)


if __name__ == "__main__":
    translated = translate_markdown()
    build_pdf(translated)
    print(OUT_PDF)
