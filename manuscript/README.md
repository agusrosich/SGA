# Manuscript Workspace

This folder contains the manuscript and reporting materials for the
censoring-aware observational causal survival analysis.

## Files

- `manuscript.md`: main manuscript draft in Markdown.
- `puntos_a_cumplir.md`: methodological revision checklist.
- `data_availability_and_emulation_limits.md`: auditable mapping of available
  and unavailable target-trial variables.
- `tables/`: trial-focused tables copied from the analysis output.
- `references/`: place bibliography files or literature review notes here.
- `output/`: generated DOCX files.
- `assets/`: figures or supplementary files.

## Build DOCX

Run from the project root:

```bat
scripts\build_manuscript_docx.bat
```

The generated file will be:

```text
manuscript\output\salivary_gland_causal_survival_manuscript.docx
```

## Run the analysis

From the repository root:

```bat
scripts\run_in_silico_trial.bat 1000 1000
```

The arguments are the number of patient-level bootstrap iterations and the
number of secondary N=252 simulations. Primary outputs are written to
`outputs/causal_survival/`.

The primary analysis uses the full eligible cohort, overall-survival censoring,
stabilized IPTW, an adjusted weighted Cox model, and a 10-year RMST ATE. The
N=252 simulations are secondary and illustrative. The treatment field does not
establish concurrent chemoradiotherapy, and the source extract is insufficient
for a strict RTOG 1008 target-trial emulation.
