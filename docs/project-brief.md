# ScentBridge-LM

Build a technical plan for a new small language model for perfumery.

The model must translate in both directions:

- Natural-language odor description → perfume formula.
- Perfume formula → natural-language odor description.

Do not treat this as a normal chatbot project.

The main goal is to build a small domain model for olfaction.

The model must use only the compute that is necessary for the current task.

The system must use sparse models, routing, dynamic depth, structured decoding, quantization, and specialist components where these methods improve performance.

## Core idea

Do not make natural language the internal representation of smell.

Create an intermediate olfactory representation.

Use this architecture:

```text
Natural language
      ↓
Language encoder
      ↓
Olfactory latent space
      ↓
Formula decoder
      ↓
Perfume formula
```

The reverse path must also work:

```text
Perfume formula
      ↓
Material and mixture encoder
      ↓
Olfactory latent space
      ↓
Language decoder
      ↓
Natural-language odor description
```

The olfactory latent space is the central component.

Both directions must use the same olfactory representation.

## ScentBridge

Create ScentBridge 1.0 as the knowledge and representation layer.

ScentBridge must contain:

- An odor ontology.
- Material identifiers.
- Material metadata.
- Molecular information where available.
- Human odor descriptions.
- Professional perfumer descriptions.
- Perceptual attributes.
- Material embeddings.
- Accord embeddings.
- Formula embeddings.
- Mixture observations.
- Volatility data.
- Odor strength.
- Diffusion.
- Tenacity.
- Temporal development.
- Known material interactions.
- Safety and usage constraints.
- Empirical evaluation data.

Do not put all factual knowledge in the language model weights.

Use external structured data when this is more efficient.

## Formula representation

Do not make the model generate raw ingredient names when this is not necessary.

Use stable material tokens.

Example:

```text
<MAT_00847> 0.120
<MAT_00192> 0.075
<MAT_03117> 0.004
```

Each material token must map to structured data.

Example:

```json
{
  "id": "MAT_00847",
  "name": "Phenethyl alcohol",
  "cas": "...",
  "embedding": [],
  "odor_profile": [],
  "volatility": null,
  "strength": null,
  "constraints": []
}
```

The model must learn olfactory relationships.

The model must not waste capacity on database lookup tasks.

## Training strategy

Use paired and unpaired data.

Paired data can contain:

```text
odor description ↔ formula
```

Unpaired data can contain:

- Odor descriptions.
- Perfume formulas.
- Material descriptions.
- Accord descriptions.
- Molecular odor data.
- Human sensory evaluations.

Use back-translation where useful.

Use cycle consistency.

Example:

```text
description
→ predicted formula
→ reconstructed description
```

The reconstructed description must preserve the important olfactory properties of the original description.

Also test:

```text
formula
→ predicted odor representation
→ reconstructed formula
```

Do not require exact formula reconstruction when several formulas can produce similar perceptual results.

Use a perceptual loss or latent-space similarity where this is more correct.

## Human preference data

Add a workflow for expert perfumers.

The expert must be able to compare two outputs.

Example:

```text
Prompt:
"Cold mineral rose with a metallic edge. Low sweetness."

Candidate A:
...

Candidate B:
...
```

The expert selects the better result.

Use this data for preference optimization.

Also collect:

- Corrections.
- Attribute ratings.
- Similarity ratings.
- Material substitutions.
- Accord corrections.
- Formula feasibility judgments.

## Target model

Do not assume that a large dense model is necessary.

Investigate a sparse specialist model.

A useful initial research target is:

```text
Total parameters: 2B-5B
Active parameters per token: 300M-800M
Sparse MoE: yes
Top-k routing: small
Dynamic depth: yes
Hybrid attention/state-space architecture: evaluate
Multi-token prediction: evaluate
Quantized local inference: required
```

These values are research targets.

Do not treat them as fixed requirements.

Measure the smallest model that can meet the evaluation targets.

## Candidate specialists

Investigate whether the model should contain different experts for:

- Language and perception.
- Raw materials.
- Accords.
- Formula structure.
- Mixture interactions.
- Volatility and evaporation.
- Diffusion.
- Intensity.
- Temporal development.
- Semantic odor concepts.

Do not add an expert unless evaluation shows that it is useful.

## Projects to evaluate

Use the following projects as building blocks, references, or implementation sources.

Do not combine them without analysis.

### Model architecture

**Qwen3.8-Flash-Next**

Study:

- Sparse MoE.
- Low active parameter count.
- Gated DeltaNet.
- Sparse attention.
- N-gram embeddings.
- Multi-token prediction.

Use these concepts where they improve the ScentBridge-LM architecture.

### Small efficient model architecture

**Gemma 4**

Study:

- Small MoE configurations.
- Per-layer embeddings.
- Methods that increase model capacity without equal increases in inference compute.

### Training from scratch

**PithTrain**

Evaluate it as the main training framework for a custom sparse MoE model.

Prefer simple code that an AI coding agent can inspect and modify.

### Fine-tuning

**Soup**

Use it where useful for:

- SFT.
- LoRA.
- QLoRA.
- Embedding training.
- Preference training.
- Evaluation.

### Efficient fine-tuning

**Unsloth**

Evaluate it for low-memory fine-tuning and rapid experiments.

### Post-training

**Hugging Face TRL**

Evaluate:

- SFT.
- DPO.
- KTO.
- GRPO.
- Distillation.

Use only the methods that match the data.

### Training efficiency

**Liger Kernel**

Evaluate optimized training kernels.

**FlashAttention-4**

Evaluate it for attention workloads.

### Dynamic compute

**TIDE**

Study token-level or layer-level early exit.

Simple tokens should not use the same compute as difficult tokens.

Test whether dynamic depth improves the compute-to-quality ratio.

### Inference

**vLLM**

Use it for efficient inference experiments.

**EAGLE / speculative decoding**

Evaluate speculative decoding.

Also evaluate native multi-token prediction if the base architecture supports it.

### Structured output

**XGrammar 2**

Use constrained generation for formulas.

The generated output must follow a strict schema.

The system must reject:

- Invalid material identifiers.
- Invalid units.
- Impossible numeric formats.
- Invalid JSON.
- Invalid structural syntax.

Use deterministic validation after decoding.

### Local deployment

**llama.cpp / GGUF**

Use it as a target for local deployment if the architecture is compatible.

Test low-bit quantization.

The final system should be able to run locally on ordinary high-end consumer hardware if possible.

## Olfactory projects and datasets

### Pyrfume

Use Pyrfume as a source of standardized olfactory datasets.

Map relevant datasets into the ScentBridge schema.

### OpenPOM

Use OpenPOM as a reference and possible pretrained component for molecular odor representation.

Investigate whether its Principal Odor Map can initialize or constrain part of the ScentBridge latent space.

### odor-pair

Use odor-pair data to study interactions between odor molecules.

Test whether pairwise data improves mixture prediction.

### Microsoft Olfaction

Study the published models and data that connect:

```text
molecule
→ olfactory receptor activity
→ perceptual properties
```

Use this as biological information where it improves the model.

Do not assume that receptor prediction alone is sufficient for perfume perception.

## Teacher models

Use a larger open-weight model as a teacher during development.

Use the teacher for:

- Synthetic training examples.
- Data normalization.
- Description expansion.
- Formula explanation.
- Error classification.
- Dataset cleaning.
- Distillation.

Do not allow synthetic data to become the only source of truth.

Human sensory data must have higher value.

Expert perfumer data must have higher value than synthetic data when the two sources conflict.

## Evaluation

Create an evaluation suite before major training.

Keep a permanent hidden test set.

Measure at least:

1. Description → formula quality.
2. Formula → description quality.
3. Ingredient validity.
4. Formula validity.
5. Olfactory attribute preservation.
6. Expert preference.
7. Formula diversity.
8. Reconstruction consistency.
9. Material substitution quality.
10. Accord understanding.
11. Mixture prediction.
12. Quantization loss.
13. Inference speed.
14. Active compute per token.
15. Memory use.

Do not use language-model perplexity as the main measure of success.

The main metric is olfactory usefulness.

## Research principle

The main research problem is not language generation.

The main research problem is the relationship between:

```text
molecules
materials
mixtures
formulas
human perception
olfactory concepts
language
```

Keep this structure explicit.

## Development method

Work in phases.

### Phase 1

Create the data model and ScentBridge ontology.

Create material identifiers.

Import existing public olfactory datasets.

Create a baseline latent space.

### Phase 2

Create a bidirectional baseline.

Use an existing small open-weight model.

Train:

```text
description → formula
formula → description
```

Use this model only as a benchmark.

### Phase 3

Train the olfactory representation.

Connect:

- Molecular data.
- Material data.
- Human descriptions.
- Formula data.
- Mixture data.

Test whether the latent space contains meaningful odor relationships.

### Phase 4

Add expert preference data.

Train preference models.

### Phase 5

Build the small specialist architecture.

Evaluate sparse MoE and dynamic compute.

Distill from larger teacher models where useful.

### Phase 6

Optimize inference.

Add:

- Quantization.
- Speculative decoding.
- Structured output.
- Local deployment.

### Phase 7

Compare the custom model against strong open-weight general models.

The custom model must justify its existence.

It must be smaller, faster, more controllable, or more accurate in the perfume domain.

Prefer all four.

## First task

Do not start implementation immediately.

First produce a technical architecture review.

For each project in this prompt:

1. Inspect the current repository and documentation.
2. State what part is useful.
3. State what part is not useful.
4. Check the license.
5. Check hardware requirements.
6. Check model compatibility.
7. Check maintenance status.
8. Identify integration risks.

Then propose the smallest coherent first stack.

Do not make a stack of technologies only because they are available.

Remove redundant components.

Prefer simple architecture.

After this review, produce:

1. The final proposed architecture.
2. A component diagram.
3. A training pipeline.
4. A data schema.
5. An evaluation design.
6. A hardware plan.
7. A repository structure.
8. A staged implementation plan.
9. The first executable experiment.
10. Clear criteria for whether the experiment succeeds or fails.

The objective is not to make a perfume chatbot.

The objective is to create a compact olfactory language model that can learn a shared representation of smell and translate between human odor concepts and perfume formulas.