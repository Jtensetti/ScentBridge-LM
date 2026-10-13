# ScentBridge-LM

A compact olfactory model for **odor description → perfume formula** and **perfume formula → odor description**, connected through a shared, nonlinguistic olfactory representation.

Project repository: [Jtensetti/parfumeLM](https://github.com/Jtensetti/parfumeLM).

## Status — 2026-10-13

**ScentBridge-LM 0.1 is a locally trained generative language model**, based on Qwen3-0.6B, with standalone merged weights and a Swedish local application. It supports grounded text/formula responses using 39 starting accords and exact lookup of 4,992 approved formulas, with deterministic mass calculations. See [the language-model release](docs/language-model-0.1.md) for training, evaluation and scope. Model files and the application bundle are delivered locally; this repository contains the reproducible code.

The earlier E1 experiment learned a small text/formula retrieval embedding but achieved **0% on unseen test families**. That negative result remains valid: the new release uses catalog grounding and does not claim a solution to blind olfactory prediction. See [the quantitative intake and E1 report](docs/quantitative-intake-and-e1.md).

A [reference accord intake](docs/reference-data-intake.md) has now been audited against the user-supplied original ZIP: 22 accord concepts, 19 variants, 48 ingredient entries and 40 qualitative reference records. The reviewed local dataset preserves modifier details lost in the original flat export. It contains no personal sensory tests or quantified formulas and remains separate from training and the public repository.

The later quantitative archive is a separate intake: 4,992 formulas and 99,839 generated instruction examples. The user explicitly confirmed that **all formulas, proportions and descriptions were smell-tested and approved**, superseding their old untested flags. This is recorded as user-reported evidence; original files are preserved, and no missing numerical ratings or test protocols are invented. Corrected data and model weights are stored locally; this repository contains code and aggregate results.

## Read the plan

1. [Original project brief](docs/project-brief.md)
2. [Review of all 17 proposed projects](docs/architecture-review.md)
3. [Architecture, component diagram and training pipeline](docs/architecture.md)
4. [ScentBridge 1.0 data contracts](docs/data-schema.md)
5. [Evaluation and human sensory protocol](docs/evaluation.md)
6. [Hardware, repository layout and staged delivery](docs/implementation-plan.md)
7. [First executable experiment and decision criteria](docs/first-experiment.md)
8. [Inspected upstream revisions](docs/source-snapshot.json)

## Initial decision

Begin with a versioned material registry, licensed observations, a shared perceptual space and deterministic formula validation. Establish a CPU baseline before a 0.6B language benchmark. Add sparse MoE, dynamic depth and specialized kernels only when controlled experiments demonstrate a benefit.

The first model stack is Python + NumPy for the CPU baseline, then PyTorch + Transformers/PEFT for a small language benchmark. TRL is the selected post-training interface when needed; do not layer Soup and Unsloth on top by default. XGrammar enters with generative formulas. Keep OpenPOM in a separate environment if its molecular features improve results.

The implemented language-model baseline uses PyTorch/Transformers/PEFT with local LoRA training and checked JSON output. The interface retrieves approved catalog references and visibly falls back to the catalog if model output deviates. Free generation remains available separately through `python -m scentbridge.cli --raw`; it is not catalog-validated.

## Run the initial smoke experiment

```sh
python -m venv .venv
# Activate .venv for your shell before the next commands.
python -m pip install -r experiments/requirements.txt
python experiments/e0_shared_space.py --smoke
```

See the experiment document for real-data inputs. The program prints an explicit `smoke_only` status on fabricated data. Real sensory validation and a permanent private test set remain required.

The repository's own distribution license is not selected yet. Upstream code, weights and datasets retain their respective terms; their licenses are documented separately.
