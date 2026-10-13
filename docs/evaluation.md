# Evaluation before major training

## Split and governance

Create train/validation/private test partitions before normalization that learns statistics, augmentation, synthetic expansion or teacher calls. Group parent formulas, variants, duplicate source descriptions, material aliases and odor-pair reversals. Use source-held-out and molecular-scaffold-held-out slices in addition to the practical known-palette formula-family split. Report these tasks separately: a molecule split is not a new-formula evaluation.

Keep a permanent hidden sensory test set with a custodian and versioned manifest. Public fixtures exercise code only. The development team receives aggregate scores at phase gates; repeated threshold tuning on the hidden set is prohibited. Track general-model pretraining contamination as unresolved for public benchmarks and prioritize new private sensory observations.

## Measurements

| Requirement | Measurement |
|---|---|
| Description → formula | Blind sensory brief-match ratings and paired preference against retrieval and small/general-LM baselines; infeasible outputs remain failures. |
| Formula → description | Attribute MAE/macro-F1, unsupported-attribute rate and blinded assessor match; fluency reported separately. |
| Ingredient validity | Registered/permitted ID rate, including adversarial nonexistent or deprecated IDs. |
| Formula validity | Strict JSON, units, uniqueness, exact mass balance and constraint status; report each separately and abstentions. |
| Attribute preservation | Per-concept error and negation/intensity/temporal slices; no single unvalidated embedding score as truth. |
| Expert preference | Win/tie/loss, expert/session clustered confidence interval and inter-rater agreement. |
| Formula diversity | Valid unique compositions, concentration-aware distances and sensory diversity at matched brief quality. |
| Reconstruction | Independent human-anchored perceptual error in both cycles; exact formula match secondary. |
| Substitution | Blinded original/substitute similarity, availability and deterministic feasibility constraints. |
| Accord understanding | Held-out accord families, minimal-pair briefs and compositional edits. |
| Mixture prediction | Pair and ≥3-material tests at known ratios; improvement over weighted-additive baseline, including nonlinear and temporal slices. |
| Quantization | Paired BF16/Q8/Q4 quality differences with confidence bounds and all validity checks. |
| Inference speed | Warm and cold p50/p95 end-to-end latency, prefill/decode separately, tokens/s and formulas/s; fixed hardware/batch/length. |
| Active compute | Activated weights per step plus measured/estimated FLOPs, executed layers, router/cache overhead and total request cost. |
| Memory | Peak accelerator allocation and process RSS, load time, disk size, KV/recurrent cache and optional draft model. |

Language perplexity is only a diagnostic. Primary product outcome is olfactory usefulness under a fixed feasible palette and context.

## Human protocol

Initial pilot: 20 briefs, two candidates per brief, at least three independent expert ratings, randomized A/B labels and controlled dilution/substrate. Measure at agreed times, provisionally 0, 30, 120 and 360 minutes. Counterbalance order, rest between samples, include repeated references and log batch/preparation details. These are design choices for perfumer review, not validated universal protocol constants.

Collect text-only feasibility ratings separately from physical sensory ratings. Record both experts and nonexpert target users when the goal involves consumer perception. A pilot estimates variation and rater agreement; it cannot establish a robust performance claim. Plan confirmatory sample size from the pilot's clustered variance and predeclare the minimum meaningful preference improvement.

## Predeclared advancement gates

These are proposed engineering gates, not measured results or universal standards.

- Data gate: documented use rights, resolved IDs/units, grouped splits, adequate real observations and no detected train/test family overlap. Otherwise stop training and repair data.
- E0 gate: [specified separately](first-experiment.md); synthetic execution never passes the scientific gate.
- Shared-space gate: beat constant and retrieval/additive baselines in both directions on validation; reproduce over three seeds for learned neural runs and retain uncertainty intervals.
- Generative gate: 100% emitted candidates pass structural/registry checks; infeasible or uncertain constraints produce an explicit non-pass status. On a predeclared adversarial suite, zero accepted unknown IDs or malformed formulas.
- Human gate: confirmatory expert-preference improvement with lower clustered 95% bound above chance, or predefined noninferiority at lower compute. Ties are reported; aggregate at brief/session level.
- Optimization gate: ≥20% lower measured median end-to-end latency or peak memory, no validity regression, and ≤0.02 absolute degradation on a normalized primary quality measure with an appropriate confidence bound. Check p95 as well. Never accept a router just because exit rate is high.
- Custom-model gate: demonstrate a quality/size/latency/control advantage over a strong open-weight model and the small dense baseline at matched data, palette and validation. Record total training and deployment cost. If no advantage, retain the simpler stack.

## Required controls

Random/shuffled pairs, train-mean attributes, nearest-neighbor retrieval, additive mixture encoding, text-only LM, no-molecular-features, no-cycle-loss, no-preferences, fixed-depth dense versus MoE, and one optimization at a time. Evaluate latent collapse and independent sensory calibration. Do not optimize a learned evaluator and then use that same evaluator as the sole success criterion.
