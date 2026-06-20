# Knowing When Not to Answer

Code, data, and analysis for the paper **"Knowing When Not to Answer: Noise-Conditional
Hallucination Risk Estimation and Shift-Robust Selective Generation for Retrieval-Augmented
Language Models"** (Anmita Das, Shoumya Chowdhury, Sushanta Paul).

We study selective generation for retrieval-augmented generation (RAG) under
retrieval-noise distribution shift. The repository releases the full controlled
two-regime testbed, the naturalistic SQuAD validation, the noise-conditional risk
estimator, and the **risk-binned Mondrian conformal risk control** procedure, together
with every result file and the figure-generation code.

**Repository:** https://github.com/shoumyac/Noise-Conditional-Hallucination-RAG

## Repository structure

```
paper/                Manuscript source and compiled PDF
  ncsrag_journal.tex  LaTeX source
  refs.bib            Bibliography
  ncsrag_journal.pdf  Compiled paper
  figures/            All figures
code/                 Experiment, analysis, and figure code
  build_world_v2.py         Build the controlled two-regime fact world
  build_world_domain2.py    Second independent domain (companies/elements)
  run_experiment_v2.py      Generation across the typed-noise grid
  run_semantic_entropy.py   Semantic-entropy baseline
  run_llm_judge.py          LLM-judge relabeling
  stage1_cache.py           Cache out-of-fold risk-estimator predictions
  stage2_report.py          Conformal-under-shift + baseline report (controlled)
  naturalistic_report.py    Naturalistic per-condition + conformal report
  nat_noise_report.py       Natural-noise (no injection) analysis
  make_figures.py           Regenerate all paper figures
  analyze_*.py              Additional diagnostics and sensitivity sweeps
  colab/                    Colab Pro+ notebooks (large models, naturalistic, judge)
data/
  controlled/         Controlled-grid generations (8 generators, 80M to 9B)
  naturalistic/       SQuAD real-retrieval (BM25/dense) and natural-noise generations
  judged/             LLM-judge-relabeled files + human_annotation_sheet.csv
  reports/            Per-model JSON reports (AUROC, SE gap, conformal)
docs/                 Result summaries (multi-family, naturalistic, decomposition)
```

Files ending in `_partial.json` are resumable checkpoints and can be ignored.

## Data schema

Each `results_*.json` is a list of generation records with fields:
`qid, regime, cond, question, gold, answer, label` (correct / hallucination / abstain),
retrieval descriptors (`ret_mean, ret_top1, ret_margin, ret_min, ret_std, has_ctx`),
model-internal features (`lp_mean, lp_min, ent_mean, ent_max, ent_first, ans_len`),
and `se` (discrete semantic entropy). `judged_*.json` additionally carry `label_em`
(exact-match label) alongside the judge `label`.

## Reproducing the analysis

```bash
pip install -r requirements.txt
# controlled six-family panel
for t in qwen2_5_1_5b_instruct qwen2_5_3b_instruct phi_3_5_mini_instruct \
         mistral_7b_instruct_v0_3 qwen2_5_7b_instruct gemma_2_9b_it; do
  python code/stage1_cache.py $t && python code/stage2_report.py $t
done
# naturalistic
for t in squad_bm25_qwen2_5_7b_instruct squad_dense_qwen2_5_7b_instruct \
         squad_bm25_mistral_7b_instruct_v0_3 squad_dense_mistral_7b_instruct_v0_3; do
  python code/stage1_cache.py $t && python code/naturalistic_report.py $t
done
python code/nat_noise_report.py     # natural-noise
python code/make_figures.py         # regenerate figures
```
(Place the data files where the scripts expect them, or adjust the `DATA` path at the
top of each script.)

## Regenerating data

The large-model generations were produced on Colab Pro+ (A100) using the notebooks in
`code/colab/`. See each notebook header for instructions.

## License

Code released under the MIT License (see LICENSE). Data are derived from publicly
available, openly licensed models and the SQuAD dataset.

## Citation

```bibtex
@article{das2026knowing,
  title={Knowing When Not to Answer: Noise-Conditional Hallucination Risk Estimation
         and Shift-Robust Selective Generation for Retrieval-Augmented Language Models},
  author={Das, Anmita and Chowdhury, Shoumya and Paul, Sushanta},
  year={2026},
  note={Code and data: \url{https://github.com/shoumyac/Noise-Conditional-Hallucination-RAG}}
}
```
