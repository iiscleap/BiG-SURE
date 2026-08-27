# Text QA: TriviaQA And SVAMP

This is the Text QA slice of BiG-SURE. It contains only TriviaQA and SVAMP, evaluated with the local SQuAD metric. Gemini evaluation is not used.

Supported models:

- `Meta-Llama-3.1-8B-Instruct`
- `Llama-3.3-70B-Instruct-4bit`
- `Qwen2.5-72B-Instruct-4bit`
- `Qwen2.5-7B-Instruct`
- `gemma-3-12b-it`
- `gemma-3-27b-it`

The data is deliberately outside this code folder at `../data/text_qa/`: 400 TriviaQA examples and 300 SVAMP examples, with matching rephrased CSVs for each task.

## Workflow

The launchers are location-independent and run seed `10` only. Activate `snne`, then run these commands from this folder or invoke the scripts by path from anywhere. The numbers reported in the paper aggregate all five experimental seeds: `10`, `20`, `30`, `40`, and `50`.

Generation uses the `bigsure-text-qa` W&B project and stores local run files in `outputs/wandb/`. Run `wandb login` for online tracking or set `WANDB_MODE=offline`. `WANDB_PROJECT` and `WANDB_ENTITY` can override the project and team.

1. Generate original-question responses:

```bash
bash scripts/generate/generate_qa.sh
```

For each original question this produces one greedy response, three low-temperature (`0.1`) responses, and ten stochastic (`1.0`) responses. The greedy response is scored with SQuAD and becomes the correctness target for all later metrics.

2. Compute standard baselines from the original-question stochastic responses:

```bash
bash scripts/compute/run_semantic_entropy.sh
bash scripts/compute/run_snne.sh
bash scripts/compute/run_kle.sh
bash scripts/compute/run_graph_baselines.sh
```

All four launchers read the vanilla run paths recorded in `config/runs.tsv`. They use the ten temperature-`1.0` responses and the greedy SQuAD label, and write to `semantic_entropy_results/`, `snne_results/`, `kle_results/`, and `graph_baseline_results/`. Missing semantic clusters and pairwise similarities are computed on the first run and cached beside the generation pickle.

3. Generate rephrased-question stochastic responses:

```bash
bash scripts/generate/generate_rephrased_qa.sh
```

This uses the rephrased CSV for the same task and creates one greedy response plus ten temperature-`1.0` responses for every rephrased question. The greedy rephrased response is retained in the pickle; BiG-SURE and its ablations use the ten stochastic responses.

4. Precompute DeBERTa bidirectional entailment probabilities:

```bash
bash scripts/compute/run_precompute_entailments.sh
```

Before loading DeBERTa, the command requires exact original/rephrased question-ID coverage, three low-temperature original answers, five rephrasings per question, and ten stochastic answers per rephrasing. It writes one versioned compressed `.npz` file per task/model/seed to `outputs/entailments/`, comparing the three original low-temperature answers against all 50 rephrased high-temperature answers.

5. Run BiG-SURE:

```bash
bash scripts/compute/run_bigsure.sh
```

This runs `snne/compute_spectral_energy_weighted_all_modes_rephrased.py` with SQuAD correctness, DeBERTa entailment probabilities, `entail_prob + min`, entropy-confidence weighting, and Jensen-Shannon divergence. It verifies that the supplied generations still match the versioned entailment archive, then deterministically samples ten of its 50 high-temperature columns. Results are written to `outputs/spectral_energy/`. Rerun precompute if a legacy archive is rejected.

6. Run the strawman ablations:

```bash
bash scripts/compute/run_strawman_ablations.sh
```

This calls `snne/compute_strawman_baselines.py` on exactly the same rephrased responses, entailment files, and vanilla SQuAD labels as BiG-SURE. It writes to `outputs/strawman_ablations/`.

`config/runs.tsv` is shared by every compute stage. Original generation records `vanilla_run_dir`; rephrased generation adds `rephrased_run_dir` to the same task/model/seed row. Standard baselines require only the vanilla path. Entailment precompute, BiG-SURE, and strawman ablations require both paths.

The TriviaQA generator is explicitly given `../data/text_qa/triviaqa/llama_validation_trivia_qa.csv`; it no longer falls back to a randomly sampled validation subset. This keeps original and rephrased generations aligned.

## Retained Code

- `scripts/generate/`: original and rephrased generation.
- `scripts/compute/`: four standard baseline launchers, entailment precompute, BiG-SURE, and strawman ablations.
- `snne/compute_semantic_entropy.py`: black-box semantic entropy baseline.
- `snne/compute_snne.py`: SNNE baseline.
- `snne/compute_kle.py`: kernel language entropy baseline.
- `snne/compute_graph_baselines.py`: graph uncertainty baselines.
- `snne/compute_spectral_energy_weighted_all_modes_rephrased.py`: BiG-SURE implementation.
- `snne/compute_strawman_baselines.py`: ablation implementation.
- `snne/uncertainty/`: shared model, data, metric, and NLI runtime code required by the retained entry points.
