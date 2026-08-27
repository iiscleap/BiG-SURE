# OKVQA Visual QA

This folder is the OKVQA-only visual QA workflow in BiG-SURE. It is scoped to the 200 examples in `../data/visual_qa/okvqa/subset.csv` and three vision-language models:

- `llava-hf/llava-v1.6-mistral-7b-hf`
- `mistralai/Pixtral-12B-2409`
- `Qwen/Qwen3-VL-8B-Instruct`

The launchers run seed `10` only. The numbers reported in the paper aggregate all five experimental seeds: `10`, `20`, `30`, `40`, and `50`.

## Environments

LLaVA and Qwen3-VL generation, plus every evaluation and compute stage, use the regular `snne` Conda environment. Pixtral generation alone requires `snne_pixtral`, created from:

```bash
conda env create -f ../environments/snne_pixtral.yml
```

For a Pixtral-only generation pass, activate `snne_pixtral` and temporarily leave only `mistralai/Pixtral-12B-2409` in the launcher's `MODELS` array. The launchers default to the active environment's `python`; `PYTHON_BIN` can override it explicitly.

Generation uses the `bigsure-okvqa` W&B project and stores local run files in `outputs/wandb/`. Run `wandb login` for online tracking or set `WANDB_MODE=offline`. `WANDB_PROJECT` and `WANDB_ENTITY` can override the project and team. Downstream stages use deterministic copies under `outputs/responses/`, so they do not depend on W&B run IDs.

## Data And Layout

The downloadable dataset lives outside this task directory:

```text
../data/visual_qa/okvqa/
  subset.csv                     # 200 OKVQA examples
  metadata.csv                   # original-question metadata
  perturbations.csv              # image perturbation metadata
  rephrased_perturbations.csv    # rephrased question/image-perturbation pairs
  images/                        # 200 original images
  images_perturbed/              # 1,400 perturbed images
```

- `inference/`: generation launchers.
- `evaluation/`: VQA accuracy calculation.
- `baselines/`: individual baseline, entailment, BiG-SURE, and consolidation launchers.
- `outputs/`: responses, entailments, BiG-SURE outputs, and consolidated tables.
- `snne/`: implementation code used by the launchers.

The launchers derive all code and data paths from their own locations. For the command sequence below, change to the task folder with a checkout-relative path:

```bash
cd BiG-SURE/visual_qa_okvqa
```

## Generation Modes

### 1. Vanilla And Stochastic Sampling

```bash
bash inference/run_vanilla_generation.sh
```

For each original question-image pair, `snne/generate_okvqa_answers.py` creates one greedy response, three low-temperature responses (`--low_temp 0.1`), and ten high-temperature stochastic responses (`--temperature 1.0`).

Each run writes `validation_generations.pkl`, `uncertainty_measures.pkl`, and `experiment_details.pkl` below `outputs/responses/vanilla/<model>_seed<seed>/`. W&B receives the same artifacts. The high-temperature responses supply the standard uncertainty baselines.

`validation_generations.pkl` is the canonical model-output artifact. For each question ID it contains `most_likely_answer`, three `low_temp_responses`, and ten high-temperature `responses`. The generation-time `uncertainty_measures.pkl` only records the ordered question IDs; semantic cluster IDs are derived from the ten stochastic responses when a standard baseline first needs them.

After every generation command, the launcher reloads the saved pickle and validates its schema. Vanilla runs must contain 200 complete records; rephrased-perturbed runs must contain 7,000 complete records. Evaluation validates the vanilla pickle again and requires exact question-ID coverage in `vqa_accuracy.json`.

### 2. Rephrased-Perturbed Sampling

```bash
bash inference/run_perturbed_generation.sh
```

`snne/generate_okvqa_rephrased_perturbed.py` generates one greedy response and ten temperature-`1.0` responses for every row in `rephrased_perturbations.csv`, using the corresponding image in `images_perturbed/`. The CSV has 35 rows per original question: five text rephrasings for each of seven image perturbations. It writes to `outputs/responses/perturbed/`.

The greedy perturbed response is retained for inspection; BiG-SURE uses the ten stochastic responses. Perturbed responses are not used by the standard baselines.

## Evaluation

After vanilla generation, compute official VQA accuracy for the greedy answers:

```bash
bash evaluation/run_vqa_accuracy.sh
```

For every vanilla run this writes `vqa_accuracy.json` next to `validation_generations.pkl`. Every standard baseline, entailment precompute, and BiG-SURE launcher consumes this JSON. Continuous official VQA accuracy is converted to a binary correctness label with `vqa_acc >= 0.5` for AUROC.

## Standard Baselines

Run each baseline independently after vanilla generation and evaluation:

```bash
bash baselines/run_snne.sh
bash baselines/run_kle.sh
bash baselines/run_graph_baselines.sh
bash baselines/run_blackbox_semantic_entropy.sh
```

Each launcher requires both files:

```text
outputs/responses/vanilla/<model>_seed<seed>/validation_generations.pkl
outputs/responses/vanilla/<model>_seed<seed>/vqa_accuracy.json
```

Every baseline uses the same artifact loader. If `uncertainty_measures.pkl` or semantic cluster IDs are absent, whichever baseline is run first computes and caches them. SNNE, KLE, graph baselines, and black-box semantic entropy can therefore be run independently and in any order.

The derived `uncertainty_measures.pkl` and `embedding_and_similarity.pkl` caches store the ordered question IDs from `validation_generations.pkl`. A stale cache from another seed or generation run is detected and recomputed automatically. No pickle copying, renaming, or W&B run-directory lookup is required. Launchers print the exact missing canonical artifact and fail if no complete run is available.

- `run_snne.sh`: SNNE semantic/entailment and lexical uncertainty from vanilla answers.
- `run_kle.sh`: KLE uncertainty from the vanilla answer-similarity kernel.
- `run_graph_baselines.sh`: degree, eigen, eccentricity, and lexical-graph measures over vanilla answers.
- `run_blackbox_semantic_entropy.sh`: black-box semantic entropy and DSE-style uncertainty from semantic answer clusters.

Baseline CSVs are written to the established task-root directories: `snne_results/`, `kle_results/`, `graph_baseline_results/`, and `blackbox_se_results/`.

## BiG-SURE

BiG-SURE must precompute entailments after both vanilla and rephrased-perturbed generation:

```bash
bash baselines/run_precompute_okvqa_entailments.sh
```

This calls `snne/precompute_entailments_vqa.py` for each model and seed. Before loading the NLI model it requires exact question-ID coverage, exactly three low-temperature vanilla answers, 35 perturbed/rephrased variants per question, and ten stochastic answers per variant. It then compares the three vanilla answers against all 350 perturbed/rephrased answers, uses `vqa_accuracy.json` for labels, and writes a versioned archive:

```text
outputs/entailments/<model>_seed<seed>_perturbed.npz
```

Then run the BiG-SURE spectral-energy method:

```bash
bash baselines/run_bigsure_spectral_energy.sh
```

The launcher uses `entail_prob + min`, entropy-confidence weighting, and Jensen-Shannon divergence. It reads each vanilla response pickle and entailment `.npz`, verifies their question IDs and low-temperature responses, subsamples ten of the 350 high-temperature columns deterministically, then writes per-run artifacts under `outputs/spectral_energy/`. If an older archive is rejected, rerun the precompute command above.

## Consolidation

After the desired baseline and BiG-SURE runs finish:

```bash
bash baselines/run_consolidate_results.sh
```

This reads all available seed CSVs from the four baseline directories, the per-seed official accuracy JSONs, and per-seed summaries under `outputs/spectral_energy/`. It averages each method across available seeds and writes:

```text
outputs/consolidated/consolidated_vqa_pivot.csv
```

## Run Order

```bash
bash inference/run_vanilla_generation.sh
bash inference/run_perturbed_generation.sh
bash evaluation/run_vqa_accuracy.sh
bash baselines/run_snne.sh
bash baselines/run_kle.sh
bash baselines/run_graph_baselines.sh
bash baselines/run_blackbox_semantic_entropy.sh
bash baselines/run_precompute_okvqa_entailments.sh
bash baselines/run_bigsure_spectral_energy.sh
bash baselines/run_consolidate_results.sh
```
