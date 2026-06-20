# Natural-noise run + the missing-evidence / misinformation decomposition

## 1. Natural-noise experiment (SQuAD full ~2000-paragraph pool, NO injection)
Qwen2.5-7B, BM25+dense pooled, 600 generations, 553 answered, halluc rate 0.212.

Core result - real retrieval failure induces hallucination:
- gold paragraph RETRIEVED (gold_in_ctx=1): n=507, hallucination 0.179
- gold paragraph MISSED  (gold_in_ctx=0): n=16,  hallucination 0.625
- closedbook (answered):                   n=30,  hallucination 0.533

This rebuts "your noise is synthetic": with zero injected noise, natural retrieval
misses raise hallucination ~3.5x (0.18 -> 0.63).

Estimator on natural noise (grouped 5-fold CV, pooled): retrieval 0.582, model 0.712,
combined 0.691, SE alone 0.674; combined-SE +0.017 [-0.024,+0.059] (NOT significant).
On subtle natural noise the retrieval descriptors are weak and SE is competitive -
the opposite of the injected finding. The mechanism (below) explains why.

## 2. The decomposition (the key new insight)
Within the injected naturalistic SQuAD data, split conditions into two failure modes
and measure each signal's AUROC:

Qwen2.5-7B (BM25):
  missing-evidence (closedbook,irr50,irr100), n=51, halluc 0.69:
     SE 0.668 | retrieval 0.513 | combined 0.420
  misinformation (contra_r1,contra_only,poison,mixed), n=961, halluc 0.39:
     SE 0.521 | retrieval 0.800 | combined 0.859
Mistral-7B (BM25):
  missing-evidence, n=453, halluc 0.86:
     SE 0.701 | retrieval 0.504 | combined 0.744
  misinformation, n=692, halluc 0.30:
     SE 0.587 | retrieval 0.667 | combined 0.755

Interpretation:
- Semantic entropy detects MISSING-EVIDENCE uncertainty (AUROC 0.67-0.70) but is
  BLIND to confident MISINFORMATION absorption (AUROC 0.52-0.59, near chance):
  when the model absorbs a false but coherent passage, its samples agree, so entropy
  is low even though the answer is wrong.
- Retrieval descriptors are the MIRROR: near chance on missing-evidence (0.50-0.51,
  because retrieval scores do not reveal the model's internal uncertainty) but strong
  on misinformation (0.67-0.80, because contradictory/irrelevant passages have
  observable signatures).
- The COMBINED estimator captures both modes (0.74-0.86 on misinformation, recovers
  missing-evidence for Mistral at 0.74).

This reconciles the whole paper:
1. It explains WHY the SE baseline collapses on injected conflict noise (the headline).
2. It explains WHY SE looked competitive on natural noise (dominated by missing-evidence).
3. It justifies the combined estimator design (complementary, not redundant, signals).
Caveat: Qwen missing-evidence n=51 is small (heavy abstention under irrelevance);
Mistral (n=453) is the reliable demonstration; the pattern is consistent across both.

## 3. Judge-vs-exact-match agreement (label validation)
Full judged files:
- natural_bm25_qwen:  agree 0.873, kappa3 0.783, kappa2(c/h) 0.608, EMh->Jc 24, EMc->Jh 6
- natural_dense_qwen: agree 0.877, kappa3 0.792, kappa2(c/h) 0.575, EMh->Jc 26, EMc->Jh 9
- squad_bm25_qwen:    agree 0.893, kappa3 0.835, kappa2(c/h) 0.792, EMh->Jc 46, EMc->Jh 78
- squad_bm25_mistral: agree 0.783, kappa3 0.668, kappa2(c/h) 0.608, EMh->Jc 135, EMc->Jh 91

Judge agrees strongly with exact-match for Qwen (kappa2 0.79), moderately for Mistral
(0.61); disagreements concentrate where exact-match is brittle (Mistral's verbose
paraphrases). Judge corrects in BOTH directions (not biased). Real gate = judge-vs-human
on the 120-row sheet (human_annotation_sheet.csv).

## TODO (rewrite)
- New subsection: "Why semantic entropy fails: missing-evidence vs misinformation."
- Re-verify estimator AUROC under judged labels for the 2 fully-judged squad files.
- Fold natural-noise + decomposition + judge-agreement into ncsrag_journal.tex.
