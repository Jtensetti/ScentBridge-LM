# ScentBridge-LM 0.1

ScentBridge now has an actual generative language-model fine-tune, exported as standalone safetensors weights, plus a local Swedish interface. This replaces neither the historical E0/E1 experiments nor their negative results: the release is a **catalog-grounded language model**, not a demonstrated perceptual model for arbitrary unseen perfumes.

## Model and training

The base is [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), Apache-2.0, pinned at `c1899de289a04d12100db370d81485cdf75e47ca`. Its downloaded weight-file SHA-256 was independently checked against the Hub's LFS metadata. The merged model has 596,049,920 parameters, in BF16. The original file stores a duplicate tied output embedding; the exported safetensors removes that duplicate through the supported save API.

Local LoRA training updates 2,293,760 parameters: rank 8, alpha 16, dropout 0.05, q/k/v/o attention projections. PyTorch 2.9.1+cu128, Transformers 4.57.1, PEFT 0.17.1, Accelerate 1.11.0; the full environment is pinned in `configs/requirements-local.lock.txt`. Use [PyTorch's documented CUDA installation](https://pytorch.org/get-started/previous-versions/#v291) and the pinned stack for reproduction.

The run used the existing NVIDIA RTX A3000 Laptop GPU (6 GB). It performed 320 optimizer updates with four accumulated single-example batches per update. All 1,232 selected training examples were consumed; 101,758 assistant tokens contributed to the loss. The training data contains 424,161 total tokens before repeated sampling, no truncated/excluded examples, and 28 starting accords from 15 families. Validation has 84 examples from three distinct families; the stable balanced checkpoint-selection subset contains 32. The test file was never opened by training.

Examples comprise 672 description-to-formula, 224 formula-to-description and 336 arithmetic tasks. Eight composition variants per training accord reduce near-duplicate volume; all three approved paraphrases are used in the description-to-formula direction. Reverse examples use the approved first description and are not repeated under identical inputs. Family separation is inherited from the quantitative intake. The 70-example test reuses the predefined family holdout from E1; it is not a newly collected sensory trial.

For paired tasks, the approved reference composition and description are provided in the prompt. This deliberately trains faithful contextual output and a stable JSON contract. Therefore success on these tasks measures grounded generation, **not discovery of the right odor or ingredients without reference information**. The arithmetic tasks contain the problem rather than its answer.

Assistant-only loss masks every prompt token. Checks verified all 1,232 examples, the end-of-answer token, and absence of truncation. Training uses BF16, SDPA, gradient checkpointing, gradient clipping, AdamW, warm-up and cosine decay. The selected checkpoint is step 320: token-weighted validation loss fell from 1.10137 to 0.10323. The run took 552.65 seconds after initial loading/evaluation and peaked at 2,031.97 MiB allocated CUDA memory. No cloud training resources were used.

## Local application

The application runs on `127.0.0.1:8765` and works offline after installation. It provides:

- Swedish description-to-formula and formula-to-description flows, with an explicit accord selector when lookup is ambiguous.
- The 39 baseline accords and exact lookup of all 4,992 user-approved formulas.
- Exact decimal mass calculations for a requested batch size, separate from model-generated arithmetic.
- A generated JSON response checked against the approved composition, name and description. If any field deviates, the UI clearly identifies its fallback to the verified catalog entry.
- Explicit rejection of unknown ingredients, invalid totals, nonfinite numbers and unsupported compositions. A changed formula does not inherit approval merely because its ingredients are familiar.
- CSV recipe export. No API key or external inference service is needed.

Catalog selection uses transparent lexical lookup and approved names/descriptions, not the failed E1 learned retrieval model. Exact source wording lookup is a software check, not evidence of semantic generalization. For experimental free generation, `scentbridge.cli --raw` exposes the language model without retrieval and marks the result `catalog_validated: false`. Those outputs are outside the verified application contract.

## Reproduce

Prepare the quantitative intake first, then run these commands from the repository root. Data, weights and reports remain in local/private paths.

```sh
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r configs/requirements-local.lock.txt
python experiments/prepare_lm.py --data /path/to/approved-data --out /path/to/lm-data
python experiments/train_lm.py --base /path/to/pinned-qwen --data /path/to/lm-data --out /path/to/training --steps 320
python experiments/export_lm.py --base /path/to/pinned-qwen --training /path/to/training --data /path/to/lm-data --approved-data /path/to/approved-data --out /path/to/release
python experiments/evaluate_lm.py --base /path/to/pinned-qwen --data /path/to/lm-data --out /path/to/base-test.json --per-task 99
python experiments/evaluate_lm.py --base /path/to/release/model --data /path/to/lm-data --out /path/to/scentbridge-test.json --per-task 99
python -m unittest discover -s tests -v
```

Download the base model at the pinned revision with its tokenizer and license. The trainer does not download or execute remote model code. The LoRA adapter is also retained, with its base-model identity and revision recorded; the merged model needs no adapter assembly at runtime. Model/data distribution remains local in this delivery, while code and aggregate results are in GitHub.

## Limits

The 39 starting accords do not establish generalization across perfumery. Many variants share descriptions; the model has no new numerical sensory measurements, receptor model, temporal evaporation model or learned calibrated olfactory space. The raw language model may hallucinate or miscalculate. The finished application is deliberately bounded to the approved catalog and deterministic calculations. Novel formulation quality still requires a separate sensory evaluation, rather than interpreting JSON validity as olfactory success.

## Final evaluation and runtime export

The selected merged model was converted with llama.cpp **b11429** to Q8_0 GGUF (approximately 633 MB). Windows binaries were verified against the published release SHA-256 digests. The native engine avoids the slow Transformers generation path observed on this Windows host. The application bundle contains a Windows CPU engine; this machine preferentially uses the separately cached CUDA 12.4 engine. Both routes were exercised.

| Same 70 test inputs, no application fallback | Base Qwen3 Q8 | ScentBridge Q8 | ScentBridge BF16/Transformers |
|---|---:|---:|---:|
| Valid JSON | 36/70 | 70/70 | 70/70 |
| Exact grounded catalog answers | 35/40 | 40/40 | 40/40 |
| Exact full arithmetic answers | 0/30 | 8/30 | 8/30 |
| Median generation wall time | 2.045 s | 0.378 s | 2.979 s |

Outputs are scored before any catalog fallback. Exact means complete parsed JSON equality, including string formatting and field names; arithmetic failures include wrong numbers and schema/format errors. The base model often used prose instead of the requested JSON. Inputs/decoding budgets are matched, but generated output lengths differ. All 40 paired tests include the reference fields in context: these results do not overturn E1's failed blind-family generalization. Q8 retained BF16's aggregate accuracy on this test; this does not establish universal quantization equivalence.

A CPU-native amber query also produced the correct response in 3.58 seconds. Native CUDA amber was 0.43 seconds in a separate smoke test. Timings exclude engine startup and vary with hardware/load. Generation statistics and unmodified raw outputs are included in the local evaluation directory. The earlier interrupted BF16 base evaluation is not used in the comparison.

Application checks cover both directions, exact scaling (20/80 amber at 25 g gives 5/20 g), malformed percentages, unregistered variants, invalid batches, local-origin restrictions and explicit catalog fallback. All 21 repository unit tests passed. Exact lookup recognizes all 117 stored description strings; that is a catalog integrity check, not an independent language benchmark. The UI was exercised in the browser with actual model responses.

For native evaluation, pass a GGUF path to `evaluate_lm.py`; set `SCENTBRIDGE_LLAMA_BINARY` to the verified llama-server binary if it is not in the bundled/cached location. No extra Python packages are needed for native serving. `model/` remains available for Transformers, training and alternate deployment. Free raw generation is exposed separately and remains unvalidated.

For conversion, install `requirements-conversion.txt`, obtain the [pinned llama.cpp b11429 source and official Windows binaries](https://github.com/ggml-org/llama.cpp/releases/tag/b11429), verify their published asset hashes, then run:

```sh
python experiments/export_native.py --release /path/to/release --llama-source /path/to/llama.cpp-b11429 --cpu-runtime /path/to/verified-cpu-runtime
```

On this host, the CUDA archives were `llama-b11429-bin-win-cuda-12.4-x64.zip` and `cudart-llama-bin-win-cuda-12.4-x64.zip`; SHA-256 digests and download URLs are retained in the local native-engine manifest. The portable bundle uses the CPU archive, so installing the Python ML stack is optional for inference. The Windows launch/stop scripts start a hidden loopback process and shut down its owned native child cleanly.
