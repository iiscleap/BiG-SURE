# Multilingual SciQ

This folder contains the multilingual SciQ slice of BiG-SURE. New runs are intentionally scoped to:

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

Inference and entailment precompute are expensive. This folder includes historical response, evaluation, entailment, and result artifacts from the original March run. The corrected pipeline below validates a stricter artifact contract.

Included response/evaluation artifacts:

- `outputs/sciq/inference/apertus_sciq_infer_vanilla_seed*/`
- `outputs/sciq/inference/apertus_sciq_infer_sampling_seed*/`
- `outputs/sciq/inference/apertus_sciq_infer_rephrased_sampling_seed*/`
- `outputs/sciq/inference/aya_sciq_infer_vanilla_seed*/`
- `outputs/sciq/inference/aya_sciq_infer_sampling_seed*/`
- `outputs/sciq/inference/aya_sciq_infer_rephrased_sampling_seed*/`

These folders contain the available `generate.json`, `generate0.3.json`, `generate_with_accuracy.json`, and `generate_with_accuracy_claude.json` files copied from the original March run. The standard baseline scripts consume the evaluated vanilla files plus stochastic sampling files. BiG-SURE consumes evaluated vanilla files plus rephrased-sampling files.

The historical vanilla and standard-sampling files contain 295 aligned questions, not 300. The old inference loader removed five target questions (`1`, `4`, `6`, `11`, and `12`) because they occurred in its few-shot prefix. Existing standard-baseline result folders reproduce that historical 295-question run. New inference chooses few-shot examples outside the evaluation set and produces all 300 questions.

One bundled evaluation is incomplete: `apertus_sciq_infer_vanilla_seed50/` does not contain `generate_with_accuracy.json`. Run Step 3 for that seed before attempting a complete five-seed Gemini-based reproduction.

Two bundled rephrased-sampling files are partial JSONL checkpoints: Apertus seed 30 has 101 of 1,500 records and Aya seed 50 has 257 of 1,500. Resume Step 2 for those runs before entailment precompute. Resume logic uses question IDs, so historical JSON arrays and JSONL checkpoints are both handled safely.

Included entailments:

- `outputs/entailments/sciq_apertus_rephrased_sampling_seed10.npz`
- `outputs/entailments/sciq_apertus_rephrased_sampling_seed20.npz`
- `outputs/entailments/sciq_apertus_rephrased_sampling_seed40.npz`
- `outputs/entailments/sciq_apertus_rephrased_sampling_seed50.npz`
- `outputs/entailments/sciq_aya_rephrased_sampling_seed10.npz`
- `outputs/entailments/sciq_aya_rephrased_sampling_seed20.npz`
- `outputs/entailments/sciq_aya_rephrased_sampling_seed30.npz`
- `outputs/entailments/sciq_aya_rephrased_sampling_seed40.npz`

The bundled `.npz` files use the legacy per-rephrase layout and are retained only with the historical result tables. The corrected BiG-SURE flow rejects them automatically. Recompute version-2 entailments for the desired model and seeds with:

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

If you only want to inspect the historical tables, start from `results/consolidated/`. You can rerun standard baselines from Step 4 with the internally aligned 295-question response files. A corrected 300-question reproduction requires new inference, evaluation, and version-2 entailment precompute.

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
- Seed: `10`
- Languages: `en zh ja fr`

The launch scripts run seed `10` only. The numbers reported in the paper aggregate all five experimental seeds: `10`, `20`, `30`, `40`, and `50`.

Outputs are written under:

```text
outputs/sciq/inference/
```

Generation checkpoints are newline-delimited JSON records stored in `generate.json`. Resume, evaluation, and compute code accept both this JSONL representation and historical JSON-array files. Before loading uncertainty models, standard baselines validate unique and identically ordered question IDs, four vanilla outputs, ten sampling outputs, matching probability counts, and complete Gemini labels.

### Step 2: Generate Rephrased Sampling Outputs

Skip this step if using the bundled `outputs/sciq/inference/` artifacts.

Run perturbation/rephrasing-aware stochastic generation:

```bash
bash inference/run_rephrased_sampling.sh
```

This runs `rephrased_sampling` for both `apertus` and `aya` over the same seed set. Each rephrased record contains one greedy answer followed by ten stochastic answers. Only the ten stochastic answers are used by BiG-SURE.

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

KLE constructs all pairwise entailment edges for one question as batches. The
multilingual mDeBERTa adapter accepts both scalar pairs and these batched pairs,
and `batch_size` can be adjusted in `MultilingualEntailmentDeberta.check_implication`
if GPU memory is limited.

Results are written under:

```text
results/semantic_entropy/
results/snne/
results/kle/
results/graph_baselines/
```

### Step 5: Precompute Entailments For BiG-SURE

Skip this step only for archives that the launcher reports as current. Legacy archives are detected by schema version and recomputed.

After rephrased sampling is complete:

```bash
bash baselines/run_precompute_multilingual.sh
```

The precompute stage validates five rephrases per original question and all four languages before loading mDeBERTa. For each original question and language it compares three low-temperature vanilla answers against 50 stochastic answers (five rephrases times ten samples), recording the paraphrase index of every column. It writes version-2 multilingual entailment archives under:

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
