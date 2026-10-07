"""Build the LaTeX manuscripts in the layout of the Spanish model version.

Usage (from the repository root):
    python scripts/build_pro_latex.py            # figures, .tex and PDFs
    python scripts/build_pro_latex.py --no-pdf   # figures and .tex only

Reads manuscrito_final/manuscript.md (English, PRO text) and
manuscrito_final/manuscript_es.md (Spanish translation of the same text), and
writes into envio_PRO/latex/:
    figures/en/*.pdf, figures/es/*.pdf      vector figures re-rendered from the part 1 model
    manuscript_en_anonymized.tex/.pdf       PRO submission copy (no authors, line numbers)
    manuscript_en.tex/.pdf                  English, with authors and Supplementary Table S1
    manuscrito_es.tex/.pdf                  Spanish, with authors, Table S1 and translated figures

The layout lives in envio_PRO/latex/sga_manuscrito.sty. The .tex files are
generated: edit the markdown sources (or the .sty), then rebuild. PDFs are
compiled with tools/tectonic/tectonic.exe (XeLaTeX, Arial).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common.causal_core as core
from build_pro_submission import AFFILIATIONS, AUTHORS, PART1, RUNNING_TITLE, split_row

LATEX = ROOT / "envio_PRO" / "latex"
TECTONIC = ROOT / "tools" / "tectonic" / "tectonic.exe"

LANGS = {
    "en": {
        "source": ROOT / "manuscrito_final" / "manuscript.md",
        "abstract": "Abstract", "references": "References", "table": "Table", "figure": "Figure",
        "running": RUNNING_TITLE,
        "affiliations": [text for _, text in AFFILIATIONS],
        "supplement": "Supplementary Material",
        "flow_header": ["Selection step", "Remaining, n", "Excluded at step, n"],
        "flow_steps": {
            "Source standardized records": "Source standardized records", "T3/T4 disease": "T3/T4 disease",
            "Known N category": "Known N category", "Radiotherapy recorded": "Radiotherapy recorded",
            "Nonmissing survival time": "Nonmissing survival time",
        },
        "thousands": ",",
        # Same caption and note as 05_Supplementary_Material.docx.
        "flow_caption": "Cohort selection from the standardized SEER-derived extract.",
        "flow_note": "Criteria were applied sequentially. Surgery, postoperative intent of radiotherapy, M stage, "
                     "and diagnosis year are not available in the extract and could not be used as eligibility "
                     "criteria. SEER, Surveillance, Epidemiology, and End Results.",
    },
    "es": {
        "source": ROOT / "manuscrito_final" / "manuscript_es.md",
        "abstract": "Resumen", "references": "Referencias", "table": "Tabla", "figure": "Figura",
        "running": "Modelización predictiva en cánceres de glándulas salivales",
        "affiliations": [
            "Radioterapia, RT International Institute, Montevideo, Uruguay.",
            "Radioterapia, Unidad Académica de Radioterapia, Montevideo, Uruguay, RT International Institute.",
        ],
        "supplement": "Material suplementario",
        "flow_header": ["Paso de selección", "Restantes, n", "Excluidos en el paso, n"],
        "flow_steps": {
            "Source standardized records": "Registros estandarizados de origen", "T3/T4 disease": "Enfermedad T3/T4",
            "Known N category": "Categoría N conocida", "Radiotherapy recorded": "Radioterapia registrada",
            "Nonmissing survival time": "Tiempo de supervivencia disponible",
        },
        "thousands": ".",
        "flow_caption": "Selección de la cohorte a partir del extracto estandarizado derivado de SEER.",
        "flow_note": "Los criterios se aplicaron de forma secuencial. La cirugía, la intención posoperatoria de la "
                     "radioterapia, el estadio M y el año de diagnóstico no están disponibles en el extracto y no "
                     "pudieron utilizarse como criterios de elegibilidad. SEER, Surveillance, Epidemiology, and End Results.",
    },
}
# (output stem, language, anonymized)
VARIANTS = [("manuscript_en_anonymized", "en", True), ("manuscript_en", "en", False), ("manuscrito_es", "es", False)]
FIGURE_WIDTH = {"covariate_balance_love_plot": 0.68}  # fraction of \linewidth; default below
TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


@dataclass
class Table:
    number: str
    caption: str
    header: list[str]
    rows: list[list[str]]
    centered: list[bool]
    note: str = ""


@dataclass
class Figure:
    number: str
    caption: str
    stem: str


def tex(text: str) -> str:
    return "".join(TEX_ESCAPES.get(c, c) for c in text)


# ---------------------------------------------------------------- markdown

def parse(cfg: dict) -> tuple[str, list[tuple]]:
    """Ordered blocks ("h2"|"h3"|"p"|"table"|"figure"|"refs", payload), keeping figures and tables in place."""
    lines = cfg["source"].read_text(encoding="utf-8").splitlines()
    title, blocks, refs = "", [], []
    pending, last_table, in_refs = None, None, False
    fig_re = re.compile(rf"^!\[{cfg['figure']} (\d+)\.\s*(.*)\]\(([^)]+)\)$")
    cap_re = re.compile(rf"^{cfg['table']} (\d+)\.\s*(.*)$")
    i = 0
    while i < len(lines):
        s = lines[i].strip(); i += 1
        if not s or re.match(r"^(Authors|Autores):", s):
            continue
        if s.startswith("# "):
            title = s[2:].strip(); continue
        h = re.match(r"^(#{2,3})\s+(.*)$", s)
        if h:
            text = h.group(2).strip(); last_table = None
            in_refs = len(h.group(1)) == 2 and text == cfg["references"]
            blocks.append(("h2" if len(h.group(1)) == 2 else "h3", text)); continue
        if in_refs:
            ref = re.match(r"^\d+\.\s+(.*)$", s)
            if not ref:
                raise ValueError(f"Unexpected line in references: {s[:60]}")
            refs.append(ref.group(1)); continue
        fig = fig_re.match(s)
        if fig:
            blocks.append(("figure", Figure(fig.group(1), fig.group(2), Path(fig.group(3)).stem)))
            last_table = None; continue
        cap = cap_re.match(s)
        if cap:
            pending = (cap.group(1), cap.group(2)); continue
        if s.startswith("|"):
            block = [s]
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip()); i += 1
            if pending is None:
                raise ValueError(f"Table without a numbered caption near: {s[:60]}")
            last_table = Table(pending[0], pending[1], split_row(block[0]), [split_row(r) for r in block[2:]],
                               [a.endswith(":") for a in split_row(block[1])])
            blocks.append(("table", last_table)); pending = None
            continue
        if last_table is not None and not last_table.note:
            last_table.note = s; last_table = None; continue
        blocks.append(("p", s))
    blocks.append(("refs", refs))
    for kind in ("table", "figure"):
        numbers = [b[1].number for b in blocks if b[0] == kind]
        if numbers != [str(n) for n in range(1, len(numbers) + 1)]:
            raise ValueError(f"{kind} numbers are not sequential in {cfg['source'].name}: {numbers}")
    return title, blocks


# ---------------------------------------------------------------- LaTeX

def table_tex(t: Table) -> list[str]:
    n = len(t.header)
    # Column widths proportional to content, so long labels wrap less than short numbers.
    weights = [max(max((len(r[j]) for r in t.rows), default=0), 0.55 * len(t.header[j]),
                   max(len(w) for w in t.header[j].split()), 6) for j in range(n)]
    factors = [n * w / sum(weights) for w in weights]
    cols = "|" + "|".join(f">{{\\hsize={f:.3f}\\hsize}}{'C' if t.centered[j] and j else 'L'}"
                          for j, f in enumerate(factors)) + "|"
    out = [r"\begin{table}[H]"]
    if t.number.startswith("S"):
        out.append(r"\renewcommand\thetable{S\arabic{table}}\setcounter{table}{0}")
    out += [rf"\caption{{{tex(t.caption)}}}\label{{tab:{t.number}}}", r"\begin{sgatable}",
            rf"\begin{{tabularx}}{{\linewidth}}{{{cols}}}", r"\hline",
            r"\rowcolor{sgaheader}" + " & ".join(rf"\sgahead{{{tex(c)}}}" for c in t.header) + r" \\ \hline"]
    out += [" & ".join(tex(c) for c in row) + r" \\ \hline" for row in t.rows]
    out += [r"\end{tabularx}", r"\end{sgatable}"]
    if t.note:
        out.append(rf"\tablenote{{{tex(t.note)}}}")
    out += [r"\end{table}", ""]
    return out


def figure_tex(f: Figure, lang: str) -> list[str]:
    width = FIGURE_WIDTH.get(f.stem, 0.76)
    return [r"\begin{figure}[H]", r"\centering",
            rf"\includegraphics[width={width}\linewidth]{{figures/{lang}/{f.stem}.pdf}}",
            rf"\caption{{{tex(f.caption)}}}\label{{fig:{f.number}}}", r"\end{figure}", ""]


def supplement_table(cfg: dict) -> Table:
    flow = pd.read_csv(PART1 / "outputs" / "tables" / "cohort_flow.csv")
    fmt = lambda v: f"{v:,}".replace(",", cfg["thousands"])
    rows = [[cfg["flow_steps"][r.Step], fmt(r.Remaining_N), fmt(r.Excluded_at_step_N)] for r in flow.itertuples()]
    return Table("S1", cfg["flow_caption"], cfg["flow_header"], rows, [False, True, True], cfg["flow_note"])


def document(lang: str, anonymized: bool) -> str:
    cfg = LANGS[lang]
    title, blocks = parse(cfg)
    options = ",".join(o for o, on in (("spanish", lang == "es"), ("anonymized", anonymized)) if on)
    authors = ", ".join(rf"{tex(name)}\textsuperscript{{{aff}}}" for name, aff in AUTHORS)
    affiliations = " ".join(rf"\textsuperscript{{{n}}}{tex(text)}" for n, text in enumerate(cfg["affiliations"], 1))
    pdfauthor = "" if anonymized else ", ".join(name for name, _ in AUTHORS)
    out = [
        f"% Generated by scripts/build_pro_latex.py from {cfg['source'].relative_to(ROOT).as_posix()}; do not edit by hand.",
        r"\documentclass[10pt]{article}",
        rf"\usepackage[{options}]{{sga_manuscrito}}" if options else r"\usepackage{sga_manuscrito}",
        rf"\hypersetup{{pdftitle={{{tex(title)}}},pdfauthor={{{tex(pdfauthor)}}}}}",
        rf"\runningtitle{{{tex(cfg['running'])}}}",
        rf"\manuscriptauthors{{{authors}}}",
        rf"\manuscriptaffiliations{{{affiliations}}}",
        "", r"\begin{document}", "", rf"\maketitleblock{{{tex(title)}}}", "",
    ]
    first_section = True
    for kind, payload in blocks:
        if kind == "h2":
            if payload == cfg["references"]:
                out.append(r"\clearpage")
            elif not first_section:
                out.append(r"\FloatBarrier")  # keep each section's figures and tables inside it, as in the model
            first_section = False
            out += [rf"\section{{{tex(payload)}}}", ""]
        elif kind == "h3":
            out += [rf"\subsection{{{tex(payload)}}}", ""]
        elif kind == "p":
            out += [tex(payload), ""]
        elif kind == "table":
            out += table_tex(payload)
        elif kind == "figure":
            out += figure_tex(payload, lang)
        elif kind == "refs":
            out += [r"\begin{sgareferences}"] + [rf"\item {tex(r)}" for r in payload] + [r"\end{sgareferences}", ""]
    if not anonymized:
        # PRO receives Table S1 as a separate file; the full versions carry it as an appendix.
        out += [r"\clearpage", rf"\section{{{tex(cfg['supplement'])}}}", ""] + table_tex(supplement_table(cfg))
    out += [r"\end{document}", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- figures, compile

def build_figures() -> None:
    """Re-render the part 1 figures as vector PDF in each language (English is identical to Figures/*.tif)."""
    spec = importlib.util.spec_from_file_location("part1", PART1 / "train_seer_model.py")
    part1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(part1)
    d, _ = part1.load_data(PART1 / "data" / "raw" / "ExportadaSEER_Estandarizada.csv")
    e = core.estimate(d)
    tables = PART1 / "outputs" / "tables"
    saved = pd.read_csv(tables / "adjusted_survival_curves.csv")
    if not (saved.sort_values(["Treatment", "Month"]).Survival.values.round(10)
            == e["curves"].sort_values(["Treatment", "Month"]).Survival.values.round(10)).all():
        raise RuntimeError("Re-estimated survival curves differ from the saved part 1 outputs.")
    boot, smd = pd.read_csv(tables / "bootstrap.csv"), pd.read_csv(tables / "balance.csv")
    for lang in LANGS:
        core.plots(LATEX / "figures" / lang, e, boot, smd, formats=("pdf",), lang=lang)


def compile_pdf(stem: str) -> Path:
    result = subprocess.run([str(TECTONIC), f"{stem}.tex"], cwd=LATEX, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        noise = ("accessing absolute path", "Fontconfig error")
        log = "\n".join(l for l in (result.stdout + result.stderr).splitlines() if not l.startswith(noise))
        raise RuntimeError(f"tectonic failed for {stem}.tex:\n{log}")
    return LATEX / f"{stem}.pdf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true", help="Write figures and .tex files without compiling.")
    args = ap.parse_args()
    build_figures()
    for stem, lang, anonymized in VARIANTS:
        (LATEX / f"{stem}.tex").write_text(document(lang, anonymized), encoding="utf-8")
        print(f"{stem}.tex" if args.no_pdf else compile_pdf(stem).relative_to(ROOT))


if __name__ == "__main__":
    main()
