# E0: supervised shared attribute-space baseline

## Question and scope

Can two very small encoders map odor descriptions and known-palette formulas into the **same human-rated attribute coordinates**, better than predicting the training mean? This tests whether the available paired observations contain usable signal before allocating GPU training.

The executable reference is [e0_shared_space.py](../experiments/e0_shared_space.py). It fits two ridge regressions: normalized text counts → attributes, and square-root mass fractions → attributes. Formula decoding retrieves the nearest training-catalog formula in that shared space; reverse decoding renders predicted attribute ratings into text. It does not generate novel formulas, model chemistry, predict safety, handle temporal context or demonstrate a learned residual latent. Those limitations are intentional controls for later experiments.

## Execute

Python ≥3.11, NumPy 2.3.4. The initial check was run with Python 3.13. Commands from repository root:

```sh
python -m pip install -r experiments/requirements.txt
python experiments/e0_shared_space.py --smoke
python -m unittest discover -s tests -v
python experiments/e0_shared_space.py --data /path/to/cleared-pilot.json
```

The last command requires real prepared input; that file is not supplied. The smoke command creates deterministic fictional materials and observations in memory. It makes no downloads or paid calls. Metrics print as JSON; redirect them into restricted experiment storage if inputs are private.

## Input contract

One JSON object contains `attributes` (ordered names), `material_ids` (ordered registry palette), `rights_status` (`cleared`), `evidence` (`human_observed` or `synthetic`), and `train` / `validation` arrays. Each row contains a unique `id`, a leakage `group` (formula family/source lineage), `text`, `formula_ppm` (ID-to-integer mapping totaling 1,000,000) and `ratings` (complete values in [0,1] in declared attribute order).

```json
{
  "attributes": ["floral", "sweet"],
  "material_ids": ["MAT_00000001", "MAT_00000002"],
  "rights_status": "cleared",
  "evidence": "human_observed",
  "train": [{"id":"r1","group":"family1","text":"floral, low sweetness",
    "formula_ppm":{"MAT_00000001":700000,"MAT_00000002":300000},
    "ratings":[0.8,0.2]}],
  "validation": [{"id":"r2","group":"family2","text":"sweet floral",
    "formula_ppm":{"MAT_00000001":300000,"MAT_00000002":700000},
    "ratings":[0.4,0.8]}]
}
```

This example illustrates the format, not an adequate dataset. E0 needs at least five training and two validation rows. Require identical measurement context and a known palette for this pilot. Missing attributes must not be imputed from validation; select a complete-label subset and report exclusion counts. Production models will use masks instead.

The program checks record IDs, split families, exact formula overlap, known palette, integer ppm totals and finite ratings. Its `cleared` flag is an assertion supplied by the data preparer; it does not independently determine legal rights or human origin. Save full source manifests, context, duplicate audit and exclusion counts alongside the matrix export.

## Outputs and success/failure

Output includes dataset SHA-256, seed/environment, dimensions, wall time, text→attribute and formula→attribute MAE, train-mean MAE, relative improvements, group-bootstrap 95% intervals on the paired error improvement, catalog-retrieval attribute error, and one example output in each direction. The vocabulary and ridge weights use training rows only. Hyperparameter `alpha=1` is fixed; later tuning belongs to a separate development split.

Scientific advancement requires all of:

1. Rights-cleared human observations, independently audited grouping, common context and reliable attribute ratings.
2. At least 100 validation observations from 30 independent groups. This is a screening floor, not a statistical-power guarantee.
3. At least 10% relative MAE reduction against the train mean for **both** encoders, and each group's paired bootstrap lower 95% improvement bound above zero.
4. Catalog retrieval quality and failures inspected by a perfumer; output validity does not establish sensory equivalence.

The program marks only the **numeric** gate. A human/data audit is still necessary before phase advancement. If either direction fails, inspect data/label context and compare retrieval controls. Do not escalate to MoE in response to poor data. Synthetic input always returns `smoke_only`; insufficient real input returns `insufficient_evidence`. There is no hidden-test result in this delivery.

A numerical pass supports the next neural representation experiment only. It cannot establish novel-formula quality, perfume usefulness or commercial readiness. Word-count features deliberately expose limitations such as negation and metaphor; use their failure slices to motivate the small language encoder.
