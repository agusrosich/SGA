# Manuscript Workspace

This folder is for writing the paper from the in-silico salivary gland cancer trial.

## Files

- `manuscript.md`: main manuscript draft in Markdown.
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
manuscript\output\salivary_gland_in_silico_trial_manuscript.docx
```

The manuscript is intentionally centered on the in-silico trial result. The predictive model is described as a tool for generating a future-validatable prediction, not as the main clinical claim.
