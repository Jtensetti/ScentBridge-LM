# Reference accord intake — original archive audited

Updated 2026-10-13. The user supplied `doftackord_svenska_dataset.zip` after the original download controls in [Strukturera parfymdataset](https://chatgpt.com/share/6acde69b-1dec-83ed-886f-563cc7bc33f8) failed. All four original files have now been read and checked. The user explicitly clarified that this is **reference material without personal sensory tests**.

## Confirmed inventory

| Item | Count |
|---|---:|
| Original archive files | 4 |
| Parent accord concepts | 22 |
| Nested variants | 19 |
| Ingredient index entries | 48 |
| Flat reference records | 40 |
| Matched earlier reconstructed records | 40 |
| Earlier material labels mapped into the archive index | 50 |
| Personal sensory observations | 0 |
| Complete quantified formulas | 0 |
| E0-eligible records | 0 |

The 40 records comprise 21 nonempty parent combinations plus 19 variants. The earlier reconstruction represented the three apple variants as direct combinations, giving 24 combinations plus 16 modifiers. These are two organizations of the same references. Two alias merges account for 50 source labels mapping to 48 archive ingredient IDs; this confirms the archive's normalization, not independent chemical identity.

## Checks performed

ZIP CRC, member names, UTF-8 decoding, duplicate JSON keys, unique ingredient/record IDs, ingredient referential integrity, nested-to-flat ingredient parity, one-to-one reconstruction crosswalk, missing-quantity preservation and training exclusion checks passed. Local copies of the four originals match their archive bytes. Member hashes and the archive hash are in [the source manifest](../data/manifests/reference-accord-intake.json).

## Findings and reviewed representation

1. A qualitative trace amount in one nested variant was omitted from the original flat export. It is restored as text, with numeric amount still null.
2. Three variety-dependent modifiers lost their optionality flags during flattening. These flags and separate modifier relationships are restored.
3. One dried-fruit variant is labeled as an addition by the archive, while the visible English source does not unambiguously specify its operation. The reviewed record preserves both the archive claim and an `unspecified_modification` assessment.
4. Flattened labels combine parent context and variant names. They remain source annotations, not measured odor attributes or independent observations.
5. Archive fields named `verified_cas` and `verified_chemistry_notes` are retained as source claims. This intake does not independently verify chemistry or regulatory status.

The reviewed local dataset has 40 records with source IDs, base-record links, original modifier rules, reviewed relations, qualitative amount, optionality, missing numerical measurements and one common leakage group. `REF_ARC_MAT_...` identifiers use a separate namespace; the earlier `REF_MAT_...` identifiers are mapped through a crosswalk rather than silently reassigned. Neither namespace is a production material registry.

## Admission and publication status

Evidence remains `reference_untested`; user-tested is false. Missing weights, concentration, test conditions and sensory ratings remain null. No source variant is promoted to a safe, validated or complete formula. Original publication/author and data-use rights remain unresolved, so training/E0 admission and public raw-data redistribution remain excluded.

The public repository includes this note and a non-content manifest. A separate local deliverable contains the unchanged originals, reviewed JSON/JSONL, material and record crosswalks, and audit results. No model was trained on this material.

## Next work

Establish original-source provenance and permitted uses. Review source identity mappings against an actual material registry. If cleared, qualitative association learning can be evaluated separately. The [E0 experiment](first-experiment.md) still requires quantified formulas and context-matched human attribute ratings; the recovered archive does not supply those observations.
