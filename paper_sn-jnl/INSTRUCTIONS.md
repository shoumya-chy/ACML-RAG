# Compiling the Springer (sn-jnl) version

The ACML journal track publishes in Springer's *Machine Learning* journal, which
requires the Springer Nature `sn-jnl` LaTeX template. This folder contains a
ready `main.tex` formatted for that template, plus `refs.bib` and all figures.

## Easiest path (Overleaf)
1. Open Overleaf and start the **"Springer Nature LaTeX Template"** (gallery), or
   download the template zip from Springer's author pages.
2. Replace the template's `main.tex` with the `main.tex` in this folder.
3. Upload `refs.bib` and all `*.png` figures into the project root.
4. Compile. Reference style is set to `sn-basic` (author-year), matching
   Machine Learning. If the journal specifies a different style, change the
   class option on line `\documentclass[sn-basic]{sn-jnl}` (e.g. `sn-mathphys-num`).

## Notes / spots that may need a one-line tweak in the template
- Theorem environments (`proposition`, `assumption`, `remark`) are declared in the
  preamble. If sn-jnl reports one is already defined, delete that `\newtheorem` line.
- ORCIDs (Anmita Das 0009-0000-1401-0494; Shoumya Chowdhury 0009-0005-5552-1094)
  go into the Springer submission system metadata; they are noted as a comment in
  `main.tex`.
- Citations use `\citep`/`\citet` (natbib-style), which `sn-basic` supports.

The content, results, tables, and figures are identical to the compiled
`paper/ncsrag_journal.pdf` in the repository root.
