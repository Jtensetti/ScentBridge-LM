# ScentBridge 1.0 data contracts

Design version `1.0-draft`. JSONL is the interchange format; use Parquet for analytical batches and SQLite for registry lookup when volume warrants it. No graph database or vector service is required initially.

## Entity contracts

Every record has `schema_version`, immutable ID, revision, provenance IDs and creation/update timestamps. A correction creates a new revision, not a reused identity. Units, measurement method and uncertainty accompany measured quantities.

| Entity | Required content and relationships |
|---|---|
| `Source` | URL/DOI, snapshot/checksum, author/rights holder, acquisition date, code/data/weight license separately, permitted training and redistribution uses, review status, restrictions and attribution. Unresolved rights block inclusion in a release/training manifest. |
| `Molecule` | Internal ID, canonical structure where available, stereochemistry, InChIKey, CAS aliases, molecular descriptors and feature artifact references. Keep unknown and conflicting identifiers explicit. |
| `Material` | Stable `MAT_########` ID, names/aliases, type (single substance/natural/base/stock/carrier), supplier/grade/lot, constituent links with fractions and uncertainty, availability and deprecated/replacement relations. Never silently equate CAS with supplier identity. |
| `OntologyConcept` | Stable `ODR_...` ID, preferred label, language, synonyms, definitions, parent/related concepts and version. Separate odor family, intensity, affect and metaphor; do not force a strict tree. |
| `PropertyObservation` | Subject ID, property, value/unit, context, assay method, source, assessor class and uncertainty. Includes volatility, odor strength, diffusion, tenacity and physical metadata. Conflicting observations coexist. |
| `Description` | Text/language, subject, concept annotations including polarity/degree, observer class (consumer/perfumer/synthetic), context, parent/source IDs and teacher lineage when generated. |
| `Formula` | ID/revision, ingredients by material ID, canonical integer mass ppm, basis, carrier, stock identity, preparation context, finished-product dosage if known, formula-family ID and provenance. |
| `Accord` | Concept and optional formula-family links; do not assume every semantic accord has one recipe. Versioned embedding artifacts and supporting observations. |
| `SensoryObservation` | Formula/material/accord subject, attributes with scales and masks, assessor/session, blinded sample ID, concentration, carrier, substrate, temperature, elapsed time, repetitions and uncertainty. |
| `Interaction` | Unordered material set, ratios, context, observed outcome/residual, evidence class and uncertainty. Missing ratio does not become 50:50. |
| `Constraint` | Material/constituent scope, prohibition/max/min, units and concentration basis, product category, jurisdiction, source/version/effective dates and review state. Unknown limit is never zero or unlimited. |
| `Preference` | Prompt/candidate IDs, presentation order, assessor, choice/tie/neither, physical-versus-text evaluation, corrections, substitutions, confidence and consent/access scope. |
| `EmbeddingArtifact` | Subject IDs, encoder/checkpoint SHA, dimension, normalization, ontology/context version, training manifest hash and array checksum. Never mix vectors from different encoder revisions. |
| `SplitManifest` | Record/family IDs, split, grouping rationale, source hashes, creation seed, embargo/access, augmentation lineage and excluded duplicates. Hidden sensory labels live outside this public repository. |

## Formula transport example

These IDs and amounts are schematic and do not identify an actual perfume.

```json
{
  "schema_version": "1.0-draft",
  "formula_id": "FORM_DEMO_001",
  "registry_version": "pilot-001",
  "basis": "mass_fraction_as_supplied",
  "total_ppm": 1000000,
  "ingredients": [
    {"material_id": "MAT_00000001", "amount_ppm": 120000},
    {"material_id": "MAT_00000002", "amount_ppm": 80000},
    {"material_id": "MAT_00000003", "amount_ppm": 800000}
  ],
  "finished_product_dose_ppm": null,
  "constraint_status": "not_evaluated"
}
```

Material 3 could be a carrier only if the registry says so. `amount_ppm` means parts of the supplied material per million total parts by mass; it is not active-neat concentration. Do not accept volume units without a versioned density conversion and its conditions.

## Validation layers

1. JSON grammar/schema: strict object keys; known enum values; 1–16 unique ingredients; integer positive ppm bounded by 1,000,000; no NaN/Infinity, strings for amounts or duplicate JSON keys.
2. Registry: every ID resolves in the pinned registry, is active and allowed in the requested palette. A regex alone is insufficient.
3. Arithmetic: exact sum 1,000,000; no implicit remainder. Expand stock/natural constituent fractions only where known; track uncertain composition.
4. Contextual constraints: convert concentrate to finished-product exposure where the context allows; apply versioned category/constituent rules; return `pass`, `fail` or `indeterminate` with reasons. Structural validity never implies safe or compliant use.
5. Model output grounding: declared attributes match decoder evidence; reject unsupported factual claims about material properties.

The first executable baseline uses a deliberately narrower matrix input contract described in [E0](first-experiment.md). Its numeric columns must be prepared from this registry and observation system. E0 is not the production validator.

## Dataset ingestion

For each Pyrfume archive inspect its manifest and source terms, retain original IDs and units, resolve molecular identity without dropping stereochemistry, and map odor terms into versioned concepts. Record unmapped/ambiguous labels. Keep source-specific annotation scales and sensory protocols; calibrate mappings using overlaps rather than concatenating incomparable ratings.

Good Scents-derived pairs, OpenPOM datasets and Pyrfume archives may overlap. Deduplicate molecules, source descriptions, unordered pairs and formula families across them before splitting. Keep commercial formulas private unless their rights expressly permit publishing. No public corpus currently reviewed here has been established as sufficient paired, measured perfume-formula training data.
