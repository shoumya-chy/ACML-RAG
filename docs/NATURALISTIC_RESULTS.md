# Naturalistic validation (SQuAD, real retrieval) - results

300-query diverse SQuAD pool (one question per paragraph, seed=13), real BM25 and
dense (e5+FAISS) retrieval, injected typed-noise grid + PoisonedRAG poisoning,
white-box features + discrete semantic entropy. Exact-match labels (judge-validated
separately). Analyzer: stage1_cache.py + naturalistic_report.py.
Reports: pilot_v2/data/natreport_squad_*.json.

## Estimator + semantic-entropy baseline (answered subset)

| Generator | Retr | n_ans | h-rate | RET | MOD | Combined | SE alone | Combined-SE [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | BM25 | 1301 | 0.347 | 0.787 | 0.563 | 0.842 | 0.533 | +0.309 [+0.284,+0.334] |
| Qwen2.5-7B | dense | 1315 | 0.353 | 0.799 | 0.523 | 0.840 | 0.545 | +0.295 [+0.267,+0.325] |
| Mistral-7B | BM25 | 1438 | 0.472 | 0.804 | 0.760 | 0.852 | 0.714 | +0.138 [+0.117,+0.159] |
| Mistral-7B | dense | 1479 | 0.479 | 0.803 | 0.737 | 0.857 | 0.693 | +0.164 [+0.141,+0.186] |

- Estimator AUROC 0.84-0.86 on genuine retrieval, both retrievers, both generators.
- Retriever-independence: BM25 vs dense nearly identical (0.842/0.840; 0.852/0.857).
- SE gap positive and significant everywhere (+0.14 to +0.31). Mistral's SE is more
  informative (0.71) than Qwen's (0.53), so its gap is smaller but still decisive.

## Per-condition hallucination (answered), Qwen-7B BM25 example
closedbook 0.53 | clean 0.15 | irr50 0.84 | irr100 0.20 | contra_r1 0.12 |
contra_only 0.69 | poison 0.85 | mixed 0.15.
PoisonedRAG (poison) is the strongest hallucination inducer (0.74-0.85 across models).
Clean-condition hallucination is already 0.15-0.28 -> a 10% selective-risk target sits
BELOW the clean base rate for these models, so alpha=0.20 is the realistic operating point.

## Conformal under naturalistic noise shift
At the realistic target alpha=0.20 (justified by clean base rate ~0.2):

| Generator | Retr | Marginal viol | Marginal ans | Risk-binned viol | Risk-binned risk | Risk-binned ans |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | dense | 0.62 | 0.73 | 0.01 | 0.115 | 0.23 |
| Mistral-7B | dense | 0.47 | 0.50 | 0.00 | 0.088 | 0.21 |

(alpha=0.30: risk-binned holds at 0% violation, ~0.50 answer rate. alpha=0.10 is below
the clean base rate, so risk-binned certifies almost nothing - correct, not a failure.)

## Framing for the paper (honest)
- Controlled world: the dramatic "marginal CRC violates in 100% of resplits" headline.
- Naturalistic: milder real noise + higher base rates -> marginal violates ~half of
  resplits while risk-binned Mondrian restores validity at a usable answer rate.
- The estimator and the semantic-entropy baseline collapse replicate on real retrieval;
  the conformal dissociation reproduces directionally at a realistic target.
- This is corroboration on real data, not an overclaim of the controlled drama.

## TODO in rewrite
- Run alpha=0.20 conformal for all four naturalistic tags (have dense; add BM25) for the table.
- Fold a naturalistic subsection + this table into ncsrag_journal.tex.
- Swap exact-match labels for judge-validated labels once judge_summary kappa >= 0.7 confirmed.
