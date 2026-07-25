"""Convert the revised manuscript DOCX directly to a self-contained LaTeX file."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from docx import Document
from docx.document import Document as DocumentType
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn


def escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "−": "--", "–": "--", "—": "---", "≥": r"$\geq$", "≤": r"$\leq$",
        "±": r"$\pm$", "×": r"$\times$", "→": r"$\rightarrow$",
        "“": "``", "”": "''", "‘": "`", "’": "'",
    }
    return "".join(replacements.get(char, char) for char in text)


def inline(paragraph: Paragraph) -> str:
    chunks: list[str] = []
    for run in paragraph.runs:
        value = escape(run.text)
        if not value:
            continue
        if run.bold and run.italic:
            value = rf"\textbf{{\textit{{{value}}}}}"
        elif run.bold:
            value = rf"\textbf{{{value}}}"
        elif run.italic:
            value = rf"\textit{{{value}}}"
        chunks.append(value)
    return "".join(chunks) if chunks else escape(paragraph.text)


def iter_blocks(parent: DocumentType):
    for child in parent.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def extract_image(paragraph: Paragraph, document: DocumentType, assets: Path, number: int) -> str | None:
    blips = paragraph._p.xpath(".//a:blip")
    if not blips:
        return None
    rel_id = blips[0].get(qn("r:embed"))
    if not rel_id:
        return None
    part = document.part.related_parts[rel_id]
    suffix = Path(part.partname).suffix or ".png"
    name = f"figure_{number:02d}{suffix}"
    (assets / name).write_bytes(part.blob)
    return name


def table_latex(table: Table) -> str:
    rows = [[escape(cell.text.strip()) for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    columns = len(rows[0])
    spec = "@{}" + "X" * columns + "@{}"
    wide = columns >= 4
    environment = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    output = [rf"\begin{{{environment}}}[t]", r"\centering", r"\scriptsize" if wide else r"\small", rf"\begin{{tabularx}}{{{width}}}{{{spec}}}", r"\toprule"]
    for index, row in enumerate(rows):
        padded = row[:columns] + [""] * max(0, columns - len(row))
        line = " & ".join(padded) + r" \\"
        if index == 0:
            line = " & ".join(rf"\textbf{{{value}}}" for value in padded) + r" \\"
        output.append(line)
        if index == 0:
            output.append(r"\midrule")
    output.extend([r"\bottomrule", r"\end{tabularx}", rf"\end{{{environment}}}"])
    return "\n".join(output)


def convert(docx_path: Path, tex_path: Path, assets: Path) -> None:
    document = Document(docx_path)
    assets.mkdir(parents=True, exist_ok=True)
    blocks = list(iter_blocks(document))
    title = next((block.text.strip() for block in blocks if isinstance(block, Paragraph) and block.style.name == "Title"), docx_path.stem)
    if title == docx_path.stem:
        title = next((block.text.strip() for block in blocks if isinstance(block, Paragraph) and block.style.name.startswith("Heading 1")), title)

    body: list[str] = []
    image_number = 0
    in_abstract = False
    for block in blocks:
        if isinstance(block, Table):
            body.append(table_latex(block))
            continue
        text = block.text.strip()
        image_name = extract_image(block, document, assets, image_number + 1)
        if image_name:
            image_number += 1
            body.extend([
                r"\begin{figure}[H]", r"\centering",
                rf"\includegraphics[width=\columnwidth]{{assets/{image_name}}}",
                r"\end{figure}",
            ])
            continue
        if not text or text == title:
            continue
        style = block.style.name
        value = inline(block)
        if style.startswith("Heading 2"):
            if text == "Abstract":
                in_abstract = True
                body.append(r"\begin{abstract}")
            else:
                if in_abstract:
                    body.append(r"\end{abstract}")
                    in_abstract = False
                body.append(rf"\section{{{escape(text)}}}")
        elif style.startswith("Heading 3"):
            if in_abstract:
                body.append(rf"\textbf{{{escape(text)}.}}")
            else:
                body.append(rf"\subsection{{{escape(text)}}}")
        elif style.startswith("Heading 4"):
            body.append(rf"\subsubsection{{{escape(text)}}}")
        elif style.startswith("List Bullet"):
            body.append(rf"\begin{{itemize}}\item {value}\end{{itemize}}")
        else:
            body.append(value + r"\par")
    if in_abstract:
        body.append(r"\end{abstract}")

    author_line = next((b.text.strip() for b in blocks if isinstance(b, Paragraph) and ";" in b.text and "Authors:" not in b.text), "")
    author_line = re.sub(r"\^\d+(?:,\d+)*", "", author_line)
    author_tex = r" \and ".join(escape(part.strip()) for part in author_line.split(";") if part.strip())
    preamble = rf"""\documentclass[10pt,twocolumn]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{tabularx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{microtype}}
\usepackage[hidelinks]{{hyperref}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.45em}}
\setlength{{\columnsep}}{{0.24in}}
\title{{{escape(title)}}}
\author{{{author_tex}}}
\date{{}}
\begin{{document}}
\maketitle
"""
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(preamble + "\n".join(body) + "\n\\end{document}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("tex", type=Path)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    assets = args.assets or args.tex.parent / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    convert(args.docx, args.tex, assets)


if __name__ == "__main__":
    main()
