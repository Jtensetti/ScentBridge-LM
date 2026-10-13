# ScentBridge 0.2: a brief becomes a new recipe

The main product is a recipe designer for a non-perfumer. A request such as
“jag vill att det ska lukta aska och eld” must produce a new recipe, with
materials, proportions, weighing instructions and ways to revise the result.
The approved catalog remains a separate reference function.

Version 0.1 did not implement this product: its language-model prompts supplied
the complete reference formula and description. Its excellent copy accuracy was
not evidence of odor-to-formula design ability. This is a correction of scope,
not a claim that the earlier retrieval problem has been solved.

## Implemented path

1. A locally fine-tuned Qwen3-0.6B interprets a Swedish brief into a sparse map
   of odor direction strengths, exclusions and unsupported concepts. JSON schema
   decoding restricts IDs. Neither a target formula nor the target intent is
   present in its prompt.
2. Explicit phrases and negations are checked independently. The small encoder
   still hallucinates extra facets in free-form cases. Therefore explicit supported
   wording is authoritative, unrequested additions are discarded, and unfamiliar
   content requires clarification. This conservative vocabulary gate limits
   automatic interpretation of metaphors. Corrections are identified in the UI.
   The interface shows the resulting interpretation and permits weight changes.
3. A small trained formula decoder maps a sparse intent vector to mass
   proportions, followed by deterministic constraints and scaling. It supports 39 catalog starting accords and 12 additional odor
   directions. **Its 156 nonzero coefficients are fitted from synthetic recipe pairs derived
   from authored design hypotheses. It is a learned distillation of those rules,
   not a perceptual mixture model learned from new sensory observations.** Intent weights do not
   represent measured odor intensity, and mass blending does not guarantee the
   resulting perceptual balance.
4. The accord option uses the requested modules. The perfume-draft option adds
   a disclosed woody/floral/musk scaffold while honoring supported exclusions.
   A complete numerical recipe is produced, but no ready-for-sale or skin-use
   status is inferred.
5. An exact decimal calculator produces a 1% active blotter-trial recipe from
   specified stocks (default 10% w/w, guaiacol 1% w/w). Actual stock strengths
   can be changed; the carrier is recalculated and impossible dilution plans
   are rejected. These are mathematical requirements for prepared solutions,
   not experimentally established solubility or stability specifications.
6. The user can export a full recipe, a weighing CSV, and a linked sensory
   observation. Notes are not automatically converted into approval or training.

All new outputs have `generated_not_smell_tested` status. A user-approved parent
does not confer approval on a new mixture. Material-source evidence and recipe
lineage are retained separately. Unknown concepts can require clarification;
the system does not claim unlimited olfactory coverage.

## Material expansion

The user explicitly allowed suggestions beyond the supplied palette. Five
materials were added: Iso E Super, guaiacwood oil, guaiacol, vetiveryl acetate
and Galaxolide. Descriptors are grounded in primary sources:

- [IFF Iso E Super](https://www.iff.com/scent/ingredients-compendium/iso-e-super/): woody/amber character.
- [IFF Galaxolide](https://www.iff.com/scent/ingredients-compendium/galaxolide-undiluted/): clean musk.
- [IFRA fragrance ingredient glossary, April 2020](https://d3t14p1xronwr0.cloudfront.net/docs/ifra-fragrance-ingredient-glossary-april-2020.pdf), credited to IFRA, pages 63 and 99: guaiacwood oil, guaiacol and vetiveryl acetate descriptors.

The glossary is a vocabulary source, not current IFRA concentration limits.
The 0.2% guaiacol ceiling in the composer is an authored design guard, **not an
IFRA limit or a safety certification**. The ash direction is a proposed dry
wood/earth interpretation; no source is claimed to validate that mixture as ash.
Sources were inspected on 2026-10-13. Stock products, grades, supplier documents
and final application still matter for a physically manufactured perfume.

## Training and reproducibility

`experiments/prepare_creative.py` creates 755 training, 61 validation and 12 test
examples for intent extraction. The supervision is authored synthetic language
data, with supplied descriptions used as concept examples; it is not 755 new
sensory observations. Six pairs of odor directions are absent from the training
pair templates, including ash/smoke. The free-form application acceptance suite
is separate from the training file.

Local training uses the same pinned base model and environment as 0.1, but a
fresh LoRA adapter is trained from the original base weights. The old copy
adapter is not continued. Training uses assistant-only loss, 240 optimizer
steps, accumulation 4, rank 8 and learning rate 0.0002. The final checkpoint
choice and measured results belong in the release manifest.

```powershell
python experiments/prepare_creative.py --catalog PRIVATE/catalog.json --out PRIVATE/creative-data
python experiments/train_lm.py --base PRIVATE/Qwen3-0.6B --data PRIVATE/creative-data --out PRIVATE/creative-training --steps 240 --eval-every 80 --lr 0.0002 --max-length 1536
python experiments/evaluate_creative.py --model PRIVATE/scentbridge-q8.gguf --catalog PRIVATE/catalog.json --data PRIVATE/creative-data --out PRIVATE/evaluation.json
python -m unittest discover -s tests -v
```

This release is a working product prototype. A strong general recipe model
still requires independently evaluated novel mixtures and measured feedback.
Passing software checks or language-intent tests cannot establish that a new
recipe smells good or matches a creative brief in real sensory testing.


## 6,000 synthetic description/recipe pairs

The user explicitly requested scaling up paired synthetic descriptions and
recipes using existing olfactory knowledge. `synthetic_recipe_pairs.py` creates
6,000 unique descriptions and 6,000 unique formulas. Every record contains its
intent vector, exact percentages, parent IDs and hypothetical status. It also
exports 12,000 forward/reverse SFT instructions. Those direct language-model
instructions are **prepared, not trained** in this release.

The deployed formula specialist is trained on 4,816 pairs using nonnegative
projection of a least-squares fit, with 51 input directions and 156 nonzero
coefficients. Validation has 583 pairs and test has 601. All ratios for an
unordered set of directions belong to a single split; no composition group
crosses splits. The test mean absolute component error is 0.00002456 percentage
points. This near-exact recovery is expected: the synthetic targets were made
by additive mass mixing. It measures reconstruction of that prior, not genuine
olfactory prediction. The original approved observations are never relabeled.

## Measured language and application results

| Check | Base Qwen3-0.6B | New local LoRA |
| --- | ---: | ---: |
| Exact intent on 12 reserved pair prompts | 0/12 | 8/12 |
| Correct facet/exclusion sets on those prompts | 1/12 | 8/12 |
| Initial application regression cases before coverage guards | 7/12 | 4/12 |
| Current guarded application regression cases | not rerun | 12/12 |

The last row is a development regression suite used to repair the application,
not an untouched scientific test. The poor ungated application result is the
reason for the conservative interpretation guard; schema correctness alone was
insufficient. The encoder was trained on all 755 training examples, selected at
step 240 using all 61 validation examples (loss 0.05608), and exported to Q8.
Training took about 874 seconds with peak allocated GPU memory about 2,549 MiB.
No sensory evaluation of the new recipes has taken place.

The local release retains initial failed application results, corrected results,
training report, synthetic corpus manifest and fitted decoder. Only code and
aggregate evidence are published in the repository.
