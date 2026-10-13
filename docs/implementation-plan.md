# Hardware and staged implementation

## Hardware plan

These are planning envelopes, not measured fit guarantees. No hardware purchase or cloud allocation is needed for this review. Profile short sequences first (512–1,024 tokens, batch 1).

| Stage | Initial budget | Decision |
|---|---|---|
| Registry, E0 and data audit | Existing CPU, 8–16 GB RAM; E0 caps the pilot at 5,000 rows | No GPU dependency. Measure runtime and dataset size. |
| 0.6B adapter baseline | One existing 12–24 GB GPU; 32 GB host RAM; Linux/WSL reference environment | Start BF16/LoRA, checkpointing if needed. This is an estimate; record a fit trial before long training. |
| Mixture encoders / molecular features | CPU precomputation or same GPU; isolated OpenPOM environment | Batch and cache molecular/receptor features; avoid recomputing per request. |
| 2–5B custom MoE full training | Conditional Hopper/SM100 allocation, likely distributed; no fixed node count yet | PithTrain requires compatible architecture. Establish parameter, optimizer, activation and expert-dispatch memory from a small run before sizing. |
| Quantized local inference | Target 16–32 GB system RAM and optional 8–16 GB GPU | Weights, attention/recurrent cache, grammar, registry, heads and draft all count. CPU fallback benchmark required. |

Weight storage lower bound is `parameters × bits / 8`. A 2–5B model uses 1–2.5 GB for ideal 4-bit weights before scales, unquantized modules, cache and runtime buffers. Sparse activation reduces arithmetic but does not remove the need to store or fetch inactive experts. Full mixed-precision Adam training can require roughly 12–20 bytes per parameter before activations (24–100 GB across 2–5B), depending on optimizer/master-weight choices and sharding. These are arithmetic estimates, not PithTrain benchmarks.

For each run log hardware, driver, runtime versions, dtype, sequence length, batch, warm-up, wall time, peak VRAM/RSS, total/active parameters and tokens/formulas processed. Measure effective training tokens/s and estimate remaining GPU-hours from actual steps; no defensible dollar budget exists before this measurement and a chosen provider.

## Phases and exit conditions

| Phase | Deliverables | Exit gate / fallback |
|---|---|---|
| 0 — this review | Source review, architecture, schema, evaluation, CPU reference experiment | Review recorded; smoke execution only. Actual data rights and sensory corpus remain unresolved. |
| 1 — ScentBridge 1.0 | Registry/ontology, dataset manifests, source-specific importers, observation contracts, deduplication, baseline space | Rights-cleared pilot; E0 on real observations. If too few paired formulas exist, collect them instead of scaling a model. |
| 2 — bidirectional benchmark | Qwen3-0.6B adapters, retrieval and template baselines, grammar and semantic validator | Reproducible two-direction evaluation; fixed data and inference budgets. |
| 3 — shared representation | Contextual set encoder, supervised attributes, residual latent, optional frozen molecular features | Beats additive/text-only controls on grouped observations; uncertainty calibrated. |
| 4 — expert feedback | Blinded rating workflow, consent/access records, preferences and corrections, ranking/DPO experiment | Holdout sensory benefit, not only an improved learned reward. |
| 5 — small specialist | Dense size sweep, MoE top-k experiments, separate dynamic-depth and hybrid architecture trials | Custom design earns a Pareto improvement and has a demonstrated local export route. Otherwise keep dense. |
| 6 — inference | Q8/Q4 comparisons, optional native MTP or EAGLE, llama.cpp companion runtime | Quality and validity gates pass, real request latency/memory improve. |
| 7 — justification | Matched comparisons to strong general open-weight models and documented total cost | Publish strengths, failures and domain coverage. Retire custom architecture if it offers no useful advantage. |

No stage automatically authorizes a costly training run. Sequencing is evidence-driven rather than calendar-driven; sensory collection and data access are likely to dominate the critical path.

## Target repository layout

```text
README.md
docs/                      # current review, design, provenance snapshot
experiments/               # E0 is present; subsequent experiments versioned
tests/                     # meaningful E0 integrity tests
src/scentbridge/           # planned, not yet implemented
  registry/ ontology/ ingest/ validation/
  representations/ models/ training/ inference/ evaluation/
schemas/                   # planned versioned JSON Schemas
configs/                   # planned pinned experiment configurations
data/manifests/            # planned rights-cleared source references only
reports/                   # local/generated metrics; ignored by default
```

Raw formulas, sensory identities, hidden labels, downloaded checkpoints and third-party archives are excluded from public Git. Use restricted storage for private evaluation; commit only manifest hashes and aggregate reports approved for release. Dataset publication permissions are separate from permission to train.

## Immediate work after the review

1. Agree the pilot palette and attribute vocabulary with a perfumer; determine which formula/description observations can be used.
2. Resolve dataset-specific terms and create manifests. Import a small eligible subset and inspect mappings manually.
3. Prepare grouped train/validation input for E0 and record the real-data result, including negative results.
4. Only after that gate, implement the production registry validator and neural benchmark. Keep the E0 reference unchanged for comparison.
