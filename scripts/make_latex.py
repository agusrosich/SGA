from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


TABLE_CAPTIONS = {
    "simulated_trial_population_characteristics.csv": "Simulated trial population characteristics.",
    "primary_in_silico_trial_result.csv": "Primary in-silico trial result.",
    "simulated_trial_arm_characteristics.csv": "Simulated trial arm outcome characteristics.",
    "key_postoperative_radiotherapy_studies.csv": "Key postoperative radiotherapy studies.",
}


AUTHORS_LATEX = (
    r"\fontsize{10.2}{11.8}\selectfont "
    r"Federico Lorenzo\textsuperscript{1,2}, "
    r"Agustin Rosich\textsuperscript{1,2}, "
    r"Jesica Lell\textsuperscript{1,2}, "
    r"Sergio Aguiar\textsuperscript{1}, "
    r"Valentina Ferreira\textsuperscript{1}, "
    r"Karina Ochandorena\textsuperscript{1,2}\\"
    r"Eduardo Larrinaga\textsuperscript{1}, "
    r"Natalia Gadea\textsuperscript{1}, "
    r"Nicolas Larragueta\textsuperscript{1}, "
    r"Aldo Quarneti\textsuperscript{1}\\[0.45em]"
    r"\fontsize{8.0}{9.3}\selectfont "
    r"\textsuperscript{1}Radiotherapy, RT International Institute, Montevideo, Uruguay\\"
    r"\textsuperscript{2}Radiotherapy, Unidad Academica de Radioterapia, Montevideo, Uruguay, RT International Institute"
)


def tex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return "".join(LATEX_SPECIALS.get(char, char) for char in text)


def inline_markup(text: str) -> str:
    text = tex_escape(text)
    text = re.sub(r"\[([0-9,\-]+)\]", r"\\cite{\1}", text)
    text = text.replace("+/-", r"$\pm$")
    return text


def cite_list(raw: str) -> str:
    keys: list[str] = []
    for part in raw.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            keys.extend(str(num) for num in range(int(start), int(end) + 1))
        else:
            keys.append(part)
    return ",".join(f"ref{key}" for key in keys)


def fix_citations(text: str) -> str:
    return re.sub(r"\\cite\{([0-9,\-]+)\}", lambda m: rf"\cite{{{cite_list(m.group(1))}}}", text)


def make_tabular(csv_path: Path, table_num: int) -> str:
    rows = list(csv.reader(csv_path.open(newline="", encoding="utf-8")))
    if not rows:
        return ""

    headers = rows[0]
    body = rows[1:]
    caption = TABLE_CAPTIONS.get(csv_path.name, csv_path.stem.replace("_", " ").title())
    label = f"tab:{table_num}"
    col_count = len(headers)

    if csv_path.name == "primary_in_silico_trial_result.csv":
        row = body[0]
        rows_latex = "\n".join(
            rf"{tex_escape(headers[idx])} & {tex_escape(row[idx])} \\"
            for idx in range(len(headers))
        )
        return rf"""
\begin{{table}}[t]
\centering
\caption{{{tex_escape(caption)}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{p{{0.57\columnwidth}}p{{0.31\columnwidth}}}}
\toprule
Measure & Value \\
\midrule
{rows_latex}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

    if csv_path.name == "simulated_trial_arm_characteristics.csv":
        short_headers = ["Arm", "N", "Age", "T4", "N+", "Stage IV", "Median OS", "Mean OS", "Event"]
        rows_latex = "\n".join(
            " & ".join(tex_escape(cell) for cell in row) + r" \\"
            for row in body
        )
        return rf"""
\begin{{table*}}[t]
\centering
\caption{{{tex_escape(caption)}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
{" & ".join(short_headers)} \\
\midrule
{rows_latex}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""

    if csv_path.name == "simulated_trial_population_characteristics.csv":
        colspec = r"p{0.32\textwidth}p{0.18\textwidth}p{0.18\textwidth}p{0.18\textwidth}"
        size = r"\scriptsize"
        rows_latex = "\n".join(
            " & ".join(tex_escape(cell) for cell in row) + r" \\"
            for row in body
        )
        return rf"""
\begin{{table*}}[t]
\centering
\caption{{{tex_escape(caption)}}}
\label{{{label}}}
{size}
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{0.82}}
\begin{{tabular}}{{{colspec}}}
\toprule
{" & ".join(tex_escape(h) for h in headers)} \\
\midrule
{rows_latex}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""

    if csv_path.name == "key_postoperative_radiotherapy_studies.csv":
        keep = ["Author_Year", "N", "Comparison", "Primary_Endpoint", "Main_Finding", "Reference"]
        indices = [headers.index(name) for name in keep]
        colspec = r"p{0.13\textwidth}p{0.06\textwidth}p{0.18\textwidth}p{0.13\textwidth}p{0.37\textwidth}p{0.06\textwidth}"
        rows_latex = "\n".join(
            " & ".join(tex_escape(row[i]) for i in indices) + r" \\"
            for row in body
        )
        return rf"""
\begin{{table*}}[t]
\centering
\caption{{{tex_escape(caption)}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{0.9}}
\begin{{tabular}}{{{colspec}}}
\toprule
Author & N & Comparison & Endpoint & Main finding & Ref. \\
\midrule
{rows_latex}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""

    colspec = "l" + "r" * (col_count - 1)
    rows_latex = "\n".join(
        " & ".join(tex_escape(cell) for cell in row) + r" \\"
        for row in body
    )
    env = "table*" if col_count > 5 else "table"
    return rf"""
\begin{{{env}}}[t]
\centering
\caption{{{tex_escape(caption)}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{{colspec}}}
\toprule
{" & ".join(tex_escape(h) for h in headers)} \\
\midrule
{rows_latex}
\bottomrule
\end{{tabular}}
\end{{{env}}}
"""


def km_curve(group: pd.DataFrame) -> list[tuple[float, float]]:
    times = sorted(float(value) for value in group["Simulated_OS_Months"].dropna())
    n_at_risk = len(times)
    survival = 1.0
    points = [(0.0, 1.0)]
    counts = Counter(times)
    for time in sorted(counts):
        events = counts[time]
        before = survival
        survival *= 1.0 - events / n_at_risk
        points.append((time, before))
        points.append((time, survival))
        n_at_risk -= events
    return points


def write_plot_data(source_dir: Path, output_dir: Path) -> None:
    plot_dir = output_dir / "latex_data"
    plot_dir.mkdir(parents=True, exist_ok=True)

    patients = pd.read_csv(source_dir.parent / "outputs" / "in_silico_salivary_gland_trial" / "simulated_trial_patient_level_predictions.csv")
    for arm, filename in [("Chemoradiation", "km_chemoradiation.dat"), ("Radiation alone", "km_radiation_alone.dat")]:
        points = km_curve(patients.loc[patients["Simulated_Arm"] == arm])
        with (plot_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            handle.write("time survival\n")
            for time, survival in points:
                handle.write(f"{time:.3f} {survival:.5f}\n")

    boot = pd.read_csv(source_dir.parent / "outputs" / "in_silico_salivary_gland_trial" / "bootstrap_delta_os_distribution.csv")
    boot[["Delta_OS_Months"]].to_csv(plot_dir / "bootstrap_delta_os_distribution.csv", index=False)

    stage_counts = (
        patients.groupby(["Simulated_Arm", "Stage_Group"]).size().reset_index(name="N")
    )
    stage_counts.to_csv(plot_dir / "stage_distribution.csv", index=False)


def figure_latex(fig_num: int) -> str:
    if fig_num == 1:
        return r"""
\begin{figure*}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.82\textwidth,
    height=0.42\textwidth,
    xlabel={Predicted overall survival (months)},
    ylabel={Survival probability},
    xmin=0, xmax=120,
    ymin=0, ymax=1.02,
    axis background/.style={fill=SoftGray},
    grid=both,
    major grid style={white},
    minor grid style={white!65!SoftGray},
    legend pos=north east,
]
\addplot+[thick, no markers, const plot, color=SoftBlue] table[x=time,y=survival] {latex_data/km_chemoradiation.dat};
\addlegendentry{Chemoradiotherapy}
\addplot+[thick, dashed, no markers, const plot, color=SoftCoral] table[x=time,y=survival] {latex_data/km_radiation_alone.dat};
\addlegendentry{Radiotherapy alone}
\end{axis}
\end{tikzpicture}
\caption{Kaplan--Meier curves of predicted overall survival in the in-silico randomized trial.}
\label{fig:km}
\end{figure*}
"""
    if fig_num == 2:
        return r"""
\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=\columnwidth,
    height=0.72\columnwidth,
    xlabel={Median OS difference (months)},
    ylabel={Frequency},
    ybar interval,
    xtick={-10,-5,0,5,10},
    xticklabel style={font=\scriptsize},
    axis background/.style={fill=SoftGray},
    grid=major,
    major grid style={white},
]
\addplot+[hist={bins=32, data min=-10, data max=10}, fill=JournalTeal!28, draw=JournalTeal]
table[y=Delta_OS_Months] {latex_data/bootstrap_delta_os_distribution.csv};
\addplot+[thick, dashed, no markers, color=SoftCoral] coordinates {(0,0) (0,85)};
\end{axis}
\end{tikzpicture}
\caption{Bootstrap distribution of the median overall survival difference, defined as chemoradiotherapy minus radiotherapy alone.}
\label{fig:bootstrap}
\end{figure}
"""
    return r"""
\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=\columnwidth,
    height=0.72\columnwidth,
    ybar,
    bar width=10pt,
    symbolic x coords={Stage III,Stage IV},
    xtick=data,
    ylabel={Patients},
    legend pos=north west,
    axis background/.style={fill=SoftGray},
    grid=major,
    major grid style={white},
]
\addplot+[fill=SoftBlue!70, draw=SoftBlue] coordinates {(Stage III,47) (Stage IV,79)};
\addlegendentry{Chemoradiotherapy}
\addplot+[fill=SoftCoral!70, draw=SoftCoral] coordinates {(Stage III,38) (Stage IV,88)};
\addlegendentry{Radiotherapy alone}
\end{axis}
\end{tikzpicture}
\caption{Stage distribution by simulated treatment arm.}
\label{fig:stage}
\end{figure}
"""


def convert_abstract(markdown_path: Path) -> str:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    in_abstract = False

    for line in lines:
        stripped = line.strip()
        if stripped == "## Abstract":
            in_abstract = True
            continue
        if in_abstract and stripped.startswith("## "):
            break
        if not in_abstract or not stripped:
            continue
        if stripped.startswith("### "):
            title = stripped[4:]
            output.append(rf"\textbf{{{tex_escape(title)}.}}")
            continue
        output.append(fix_citations(inline_markup(stripped)) + r"\par")

    return "\n".join(output)


def convert_body(markdown_path: Path) -> str:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    in_refs = False
    in_abstract = False
    table_num = 0
    pending_table_title = ""
    pending_figure_title = ""

    for line in lines:
        stripped = line.strip()
        if stripped == "## Abstract":
            in_abstract = True
            continue
        if in_abstract:
            if stripped.startswith("## "):
                in_abstract = False
            else:
                continue
        if stripped == "## References":
            in_refs = True
            output.append(r"\begin{thebibliography}{35}")
            continue
        if in_refs:
            match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
            if match:
                output.append(rf"\bibitem{{ref{match.group(1)}}} {inline_markup(match.group(2))}")
            continue
        if not stripped:
            output.append("")
            continue
        if stripped.startswith("Authors:") or stripped.startswith("Affiliations:"):
            continue
        if stripped.startswith("# "):
            continue
        if stripped.startswith("## "):
            title = stripped[3:]
            output.append(rf"\section{{{tex_escape(title)}}}")
            continue
        if stripped.startswith("### "):
            title = stripped[4:]
            if title in {"Background", "Objective", "Methods", "Results", "Conclusions"}:
                output.append(rf"\textbf{{{tex_escape(title)}.}}")
            elif title.startswith("Table "):
                pending_table_title = title
            elif title.startswith("Figure "):
                pending_figure_title = title
            else:
                output.append(rf"\subsection{{{tex_escape(title)}}}")
            continue
        table_ref = re.search(r"`tables/([^`]+\.csv)`", stripped)
        if table_ref:
            table_num += 1
            output.append(make_tabular(markdown_path.parent / "tables" / table_ref.group(1), table_num))
            pending_table_title = ""
            continue
        image_ref = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if image_ref:
            fig_match = re.search(r"Figure\s+(\d+)", pending_figure_title or image_ref.group(1))
            fig_num = int(fig_match.group(1)) if fig_match else 1
            output.append(figure_latex(fig_num))
            pending_figure_title = ""
            continue
        output.append(fix_citations(inline_markup(stripped)) + "\n")

    output.append(r"\end{thebibliography}")
    return "\n".join(output)


def build(markdown_path: Path, output_path: Path) -> None:
    base_dir = markdown_path.parent
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_plot_data(base_dir, output_path.parent)
    body = convert_body(markdown_path)
    abstract = convert_abstract(markdown_path)
    title = tex_escape(markdown_path.read_text(encoding="utf-8").splitlines()[0].lstrip("# "))
    tex = rf"""\documentclass[10pt,twocolumn]{{article}}
\usepackage[letterpaper,margin=0.68in,columnsep=0.22in]{{geometry}}
\usepackage{{newtxtext,newtxmath}}
\usepackage{{microtype}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\usepackage{{xcolor}}
\usepackage{{titling}}
\usepackage{{tikz}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{statistics}}
\pgfplotsset{{compat=1.18}}
\definecolor{{JournalTeal}}{{HTML}}{{517C7A}}
\definecolor{{JournalGold}}{{HTML}}{{D7B83F}}
\definecolor{{SoftBlue}}{{HTML}}{{3F6FAE}}
\definecolor{{SoftCoral}}{{HTML}}{{CF6F66}}
\definecolor{{SoftGray}}{{HTML}}{{F2F4F4}}
\captionsetup{{font=small,labelfont={{bf,color=JournalTeal}}}}
\pretitle{{\begin{{center}}\color{{JournalTeal}}\Large\bfseries}}
\posttitle{{\par\end{{center}}\vspace{{-0.45em}}}}
\preauthor{{\begin{{center}}}}
\postauthor{{\par\end{{center}}\vspace{{-0.65em}}\noindent\textcolor{{JournalGold}}{{\rule{{\textwidth}}{{0.7pt}}}}\vspace{{0.75em}}}}
\setlength{{\parindent}}{{1em}}
\setlength{{\parskip}}{{0pt}}
\renewcommand{{\baselinestretch}}{{0.98}}
\title{{{title}}}
\author{{{AUTHORS_LATEX}}}
\date{{}}
\begin{{document}}
\twocolumn[{{
\maketitle
\vspace{{-1.1em}}
\begin{{center}}
\begin{{minipage}}{{0.92\textwidth}}
\small
\textbf{{Abstract}}\par\vspace{{0.35em}}
{abstract}
\end{{minipage}}
\end{{center}}
\vspace{{0.9em}}
}}]
{body}
\end{{document}}
"""
    output_path.write_text(tex, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/make_latex.py manuscript.md output.tex")
    build(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
