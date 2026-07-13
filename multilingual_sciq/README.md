# Multilingual SciQ

This folder contains the multilingual SciQ slice of BiG-SURE. It is intentionally scoped to:

- Dataset: SciQ only.
- Questions: 300 original SciQ questions from `sciq_rephrased_300.json`.
- Languages used in this workflow: `en`, `zh`, `ja`, `fr`.
- Models: `apertus` and `aya`.
- Pipeline stages: inference, Gemini evaluation, standard uncertainty baselines, and BiG-SURE spectral energy.

The raw JSONs also contain `th`, but this consolidated workflow uses the four-language setting above.

## Folder Layout

- `inference/`: model generation code and launchers.
- `evaluation/`: Gemini correctness labeling for greedy generations.
- `baselines/`: semantic entropy, SNNE, KLE, graph baselines, entailment precompute, BiG-SURE spectral energy, and consolidation.
- `outputs/`: precomputed SciQ responses, Gemini/Claude evaluation files, and multilingual entailment `.npz` files for the two-model setup.
- `results/`: precomputed baseline, BiG-SURE spectral-energy, and consolidated result CSVs.
- `prompt/`: prompt templates used by `inference/inference.py`.
- `requirements.txt`: Python dependencies inherited from the original multilingual setup.

The data lives outside this task folder:

- `../data/multilingual_sciq/sciq/mling_sciq.json`
- `../data/multilingual_sciq/sciq/sciq_rephrased_300.json`
- `../data/multilingual_sciq/sciq/sciq_examples_multilingual.json`

## Data

The main files are:

- `mling_sciq.json`: full multilingual SciQ source file.
- `sciq_rephrased_300.json`: the 300-question experimental set plus rephrased questions.
- `sciq_examples_multilingual.json`: examples/support data from the original workflow.

`sciq_rephrased_300.json` contains:

- 300 original SciQ questions.
- 1500 rephrased question entries.
- Multilingual question/answer fields. This workflow uses `en`, `zh`, `ja`, and `fr`.

## Precomputed Artifacts

Inference and entailment precompute are expensive. This folder includes the final artifacts needed to reproduce the reported baseline and BiG-SURE numbers without rerunning generation from scratch.

Included response/evaluation artifacts:

- `outputs/sciq/inference/apertus_sciq_infer_vanilla_seed*/`
- `outputs/sciq/inference/apertus_sciq_infer_sampling_seed*/`
- `outputs/sciq/inference/apertus_sciq_infer_rephrased_sampling_seed*/`
- `outputs/sciq/inference/aya_sciq_infer_vanilla_seed*/`
- `outputs/sciq/inference/aya_sciq_infer_sampling_seed*/`
- `outputs/sciq/inference/aya_sciq_infer_rephrased_sampling_seed*/`

These folders contain the available `generate.json`, `generate0.3.json`, `generate_with_accuracy.json`, and `generate_with_accuracy_claude.json` files copied from the original March run. The standard baseline scripts consume the evaluated vanilla files plus stochastic sampling files. BiG-SURE consumes evaluated vanilla files plus rephrased-sampling files.

One bundled evaluation is incomplete: `apertus_sciq_infer_vanilla_seed50/` does not contain `generate_with_accuracy.json`. Run Step 3 for that seed before attempting a complete five-seed Gemini-based reproduction.

Included entailments:

- `outputs/entailments/sciq_apertus_rephrased_sampling_seed10.npz`
- `outputs/entailments/sciq_apertus_rephrased_sampling_seed20.npz`
- `outputs/entailments/sciq_apertus_rephrased_sampling_seed40.npz`
- `outputs/entailments/sciq_apertus_rephrased_sampling_seed50.npz`
- `outputs/entailments/sciq_aya_rephrased_sampling_seed10.npz`
- `outputs/entailments/sciq_aya_rephrased_sampling_seed20.npz`
- `outputs/entailments/sciq_aya_rephrased_sampling_seed30.npz`
- `outputs/entailments/sciq_aya_rephrased_sampling_seed40.npz`

The original artifact folder did not contain `sciq_apertus_rephrased_sampling_seed30.npz` or `sciq_aya_rephrased_sampling_seed50.npz`. To recompute those two missing entailment files, run:

```bash
bash baselines/run_precompute_multilingual.sh
```

Included completed result folders:

- `results/semantic_entropy/`
- `results/snne/`
- `results/kle/`
- `results/graph_baselines/`
- `results/spectral_energy/`
- `results/consolidated/`

If you only want to inspect or reproduce tables from existing outputs, start from `results/consolidated/`. If you want to rerun metrics without expensive model inference, start from Step 4 below using the existing `outputs/` files.

## Inference Modes

All three inference modes are implemented in `inference/inference.py`.

### 1. Vanilla

`--inference_mode vanilla`

For each question and language, the script generates:

- 1 greedy answer.
- 3 low-temperature samples.

The greedy answer is the one evaluated by Gemini for correctness labels.

### 2. Sampling

`--inference_mode sampling`

For each original question and language, the script generates high-temperature stochastic samples. These outputs feed the standard uncertainty baselines.

### 3. Rephrased Sampling

`--inference_mode rephrased_sampling`

For each rephrased question and language, the script generates:

- 1 greedy answer.
- 10 high-temperature samples by default.

These perturbation/rephrasing-aware generations feed BiG-SURE spectral energy.

## Run Order

The launchers derive all code and data paths from their own locations. For the sequence below, change to the task folder with a checkout-relative path:

```bash
cd BiG-SURE/multilingual_sciq
```

### Step 1: Generate Vanilla And Sampling Outputs

Skip this step if using the bundled `outputs/sciq/inference/` artifacts.

Inference is tracked in the `bigsure-multilingual-sciq` W&B project, with local files under `outputs/wandb/`. Run `wandb login` for online tracking or set `WANDB_MODE=offline`; `WANDB_PROJECT` and `WANDB_ENTITY` are optional overrides. The generated JSON remains the canonical input to evaluation and baselines.

Run vanilla and standard stochastic sampling for both models:

```bash
bash inference/run_vanilla_and_sampling.sh
```

This runs:

- Models: `apertus`, `aya`
- Dataset: `sciq`
- Modes: `vanilla`, `sampling`
- Seeds: `10 20 30 40 50`
- Languages: `en zh ja fr`

Outputs are written under:

```text
outputs/sciq/inference/
```

### Step 2: Generate Rephrased Sampling Outputs

Skip this step if using the bundled `outputs/sciq/inference/` artifacts.

Run perturbation/rephrasing-aware stochastic generation:

```bash
bash inference/run_rephrased_sampling.sh
```

This runs `rephrased_sampling` for both `apertus` and `aya` over the same seed set. These outputs are used by BiG-SURE.

### Step 3: Run Gemini Evaluation

Skip this step if using the bundled `generate_with_accuracy.json` files.

Set your Gemini key, then evaluate the greedy answer from each vanilla run:

```bash
export GOOGLE_API_KEY=...
bash evaluation/run_gemini_evaluation.sh
```

For each vanilla `generate.json`, this writes:

```text
generate_with_accuracy.json
```

The baseline and BiG-SURE scripts use those Gemini labels as the correctness target.

### Step 4: Run Standard Baselines

After vanilla, sampling, and Gemini evaluation are complete:

```bash
bash baselines/run_semantic_entropy.sh
bash baselines/run_snne.sh
bash baselines/run_kle.sh
bash baselines/run_graph_baselines.sh
```

This computes:

- Semantic entropy.
- SNNE.
- KLE.
- Graph baselines.

Results are written under:

```text
results/semantic_entropy/
results/snne/
results/kle/
results/graph_baselines/
```

### Step 5: Precompute Entailments For BiG-SURE

Skip this step for seeds whose `.npz` files already exist under `outputs/entailments/`.

After rephrased sampling is complete:

```bash
bash baselines/run_precompute_multilingual.sh
```

This mirrors the original `March_ARR_2026/code/snne/run_precompute_multilingual.sh` flow, but uses local BiG-SURE paths. It reads evaluated vanilla outputs and rephrased-sampling outputs, then writes multilingual entailment `.npz` files under:

```text
outputs/entailments/
```

### Step 6: Run BiG-SURE Spectral Energy

After entailment precompute is complete:

```bash
bash baselines/run_bigsure_spectral_energy.sh
```

This runs:

```text
compute_spectral_energy_multilingual.py
```

with Gemini labels and the `entail_prob + min` configuration.

Results are written under:

```text
results/spectral_energy/
```

### Step 7: Consolidate Results

```bash
bash baselines/run_consolidate_results.sh
```

This writes consolidated CSV summaries under:

```text
results/consolidated/
```

## Important Entry Points

Inference:

- `inference/inference.py`
- `inference/run_vanilla_and_sampling.sh`
- `inference/run_rephrased_sampling.sh`

Evaluation:

- `evaluation/gemini_evaluate_generations.py`
- `evaluation/run_gemini_evaluation.sh`

Baselines and BiG-SURE:

- `baselines/compute_multilingual_semantic_entropy.py`
- `baselines/compute_multilingual_snne.py`
- `baselines/compute_multilingual_kle.py`
- `baselines/compute_multilingual_graph_baselines.py`
- `baselines/precompute_entailments_multilingual.py`
- `baselines/run_precompute_multilingual.sh`
- `baselines/compute_spectral_energy_multilingual.py`
- `baselines/consolidate_multilingual_results.py`
