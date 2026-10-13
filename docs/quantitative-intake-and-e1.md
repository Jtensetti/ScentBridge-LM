# Quantitative intake and E1 — 2026-10-13

## Evidence correction

The user supplied `ScentBridge_traningsdata_med_mangder.zip` and three loose companion files. All three companions match their archive counterparts byte for byte. Archive SHA-256: `828c7be3283e63bf9715be2c74c54761f49a6ab161874f2b2b89a7c180dd1a36`. All 35 listed file checksums match.

In response to a question explicitly covering the 39 starting accords, all 4,992 proportioned variants and their descriptions, the user answered **“Allt”**. This supersedes the archive's obsolete untested claims for these registered formulas. The derived data records `sensory_validated: true` together with `sensory_evidence: user_attestation` and a versioned attestation. Original formula generation remains rule based; subsequent approval does not change that history. Dates of individual trials, protocol, panel size and numeric sensory ratings remain unspecified.

Approval is not automatically propagated to newly altered compositions in arithmetic tasks. Exact matches against the registered formula catalog inherit its status; other changes carry `not_documented_for_modified_formula`. Generated preference pairs remain excluded: arithmetic correctness and old evidence labels are not human sensory preferences. The older qualitative source intake retains its historical classification.

## Local exports and checks

The import produces original byte-preserved files, corrected formula records, 117 approved description texts, corrected SFT with lineage IDs, and the 2,000-row starter subset with restored lineage. Three obsolete trial-status clauses were removed from derived descriptions; original wording is retained in the source files. Unchanged historical statements about the source having no quantities refer to the original qualitative reference, not the new approved formula catalog.

| Split | Formulas | Starting accords | Families | SFT tasks |
|---|---:|---:|---:|---:|
| Train | 3,584 | 28 | 15 | 71,680 |
| Validation | 768 | 6 | 3 | 15,360 |
| Test | 640 | 5 | 3 | 12,799 |
| Total | 4,992 | 39 | 21 | 99,839 |

No accord family crosses splits. There are no duplicate compositions. The 2,000 starter records are a subset, not additional independent observations. All 99,839 corrected outputs retain the original arithmetic values. The supplied validator was reviewed and rerun on originals (output redirected to a separate report), including 9,984 original preference pairs. Its `empirical_olfactory_samples: 0` is obsolete **source metadata**, not the corrected intake status. Fifteen source math tests and eleven repository tests passed, including finite-difference checks of the learned model's gradients.

## E1: trained shared-embedding retrieval baseline

The numeric sensory labels required by E0 are absent. E1 therefore learns paired text/formula associations directly rather than filling in fictional sensory ratings. It uses training-only word/character-trigram TF-IDF, square-root material fractions, two trainable tanh projections and symmetric contrastive loss. The 32-dimensional latent is a learned correspondence space, **not a calibrated perceptual scale**.

NumPy 2.3.4, seed 20261013; 32,864 trained parameters. Each update samples one approved paraphrase and one proportion variant per starting accord, avoiding false-negative pairs between variants sharing the same description. All 3,584 training formulas are seen. The run completes 600 updates; checkpoint 150 is selected by validation bidirectional retrieval after every variant has been used. No test query labels or scores are used for selection. Local CPU training took about 1.1 seconds; no cloud training job was launched.

Decoding ranks 39 approved baseline catalog entries. The catalog includes held-out candidates as inference inputs; their formulas/descriptions do not participate in gradient training or vocabulary fitting. Formula-to-text outputs retrieve stored descriptions. Text-to-formula outputs retrieve stored compositions. This does not generate new formulas or prose. All variants of one starting accord share three descriptions, so exact proportion recovery is not identifiable from those texts.

| Query split | Text → correct starting accord | Formula → correct starting accord |
|---|---:|---:|
| Training | 100% (84 descriptions) | 100% (3,584 formulas) |
| Validation | 5.56% (18 descriptions) | 16.67% (768 formulas) |
| Test | **0% (15 descriptions)** | **0% (640 formulas)** |

Uniform random selection from 39 candidates gives 2.56% expected accuracy. The test set has only three independent families, not 640 independent sensory experiments. Four materials in each held-out split are unseen during training. Some descriptions explicitly name ingredients. These results demonstrate memorized associations and failed transfer to new families; they are not evidence of perfumery competence. Do not tune further on this test set and continue calling it untouched.

## Reproduce locally

Requires Python and the repository's pinned NumPy dependency. Use private/local paths for data and weights.

```sh
python experiments/import_quantitative.py --archive /path/to/ScentBridge_traningsdata_med_mangder.zip --out /path/to/approved-data --confirmed-all
python experiments/e1_paired_retrieval.py --data /path/to/approved-data --out /path/to/e1-model
python experiments/e1_paired_retrieval.py --model /path/to/e1-model --text "Amber"
python -m unittest discover -s tests -v
```

`--confirmed-all` requires an explicit user statement covering every formula and description; it must not be used on a future unrelated archive without equivalent confirmation. Weights (`weights.npz`), vocabulary, training palette and local candidate catalog (`model.json`), exact metrics and learning history (`training-report.json`) are saved. Unknown ingredient IDs are rejected by the query interface; reported similarity scores are not calibrated probabilities.

## Next model work

Broaden independent accord-family coverage and record variant-specific sensory differences, then test whether a pretrained Swedish-capable text encoder and ingredient descriptors improve transfer. Keep the mass calculator deterministic. The corrected SFT is available for a later language-model experiment; it was not consumed by E1 and should not be described as an already completed 99,839-example language-model fine-tune. Source distribution rights remain separate from the user's training authorization and sensory approval; this change publishes code and aggregate findings, not the underlying reference material or weights.
