# Reference accord intake — 2026-10-13

The user supplied [a shared conversation](https://chatgpt.com/share/6acde69b-1dec-83ed-886f-563cc7bc33f8) titled **Strukturera parfymdataset** and subsequently clarified that its contents are reference material without personal sensory tests.

The original JSON and ZIP download controls were attempted. Both reported an upload-status retrieval failure. The generated source files have therefore **not** been downloaded, inspected or verified. A separate local dataset was reconstructed from the visible factual ingredient lists and modifier relationships; it must not be described as the original dataset.

## Audited interpretation

| Item | Count |
|---|---:|
| Parent accord concepts | 22 |
| Qualitative combinations, including three apple variants | 24 |
| Additional modifier/variant records | 16 |
| Reference records | 40 |
| Source material labels before identity resolution | 50 |
| Possible alias pairs awaiting review | 2 |
| Human sensory observations | 0 |
| Complete quantified formulas | 0 |
| E0-eligible records | 0 |

The earlier assistant's reported 48 normalized ingredients is consistent with merging two candidate alias pairs, but the unavailable original JSON has not been checked. Its 19 variants can be reconciled as three apple variants plus 16 additional variants. These are provenance observations, not validation of that assistant's transformations.

## Intake rules

- Evidence is `reference_untested`; user-tested is false. No measured ratings or experimental context are inferred.
- Missing amounts, units, concentrations and sensory ratings remain null. Never fill equal proportions to make the input fit E0.
- Provisional `REF_MAT_...` IDs represent source mentions; they cannot enter the production material palette until resolved.
- Modifier relationships retain a base-record link and operation. Ambiguous modifications remain ambiguous rather than silently becoming complete recipes.
- All records share a single source leakage group. Translations and variants are not independent test observations.
- Original author/publication and data-use rights are unresolved. Training, E0 admission and raw-data redistribution remain disabled.
- The full literary prose, full translation, and the previous assistant's unverified chemistry/regulatory claims were excluded from the reconstruction.

Only this intake note and a non-content manifest are committed to the public repository. The local reconstructed files are delivered separately. No reference records were added to training or hidden evaluation, and no model was trained on this material.

## Next step

Recover the original files through a direct attachment and compare their identities, modifiers and schema with the reconstruction. Establish original-source provenance and allowed uses. If cleared, qualitative association learning can be a separate task. The E0 paired-observation experiment still needs quantified formulas and context-matched human attribute ratings.

See [the intake manifest](../data/manifests/reference-accord-intake.json) and [E0's input contract](first-experiment.md).
