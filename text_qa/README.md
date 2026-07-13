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

The launchers are location-independent and use seeds `10 20 30 40 50`. Activate `snne`, then run these commands from this folder or invoke the scripts by path from anywhere.

Generation uses the `bigsure-text-qa` W&B project and stores local run files in `outputs/wandb/`. Run `wandb login` for online tracking or set `WANDB_MODE=offline`. `WANDB_PROJECT` and `WANDB_ENTITY` can override the project and team.

1. Generate original-question responses:

```bash
bash scripts/generate/generate_qa.sh
```

For each original question this produces one greedy response, three low-temperature (`0.1`) responses, and ten stochastic (`1.0`) responses. The greedy response is scored with SQuAD and becomes the correctness target for all later metrics.

2. Generate rephrased-question stochastic responses:

```bash
bash scripts/generate/generate_rephrased_qa.sh
```

This uses the rephrased CSV for the same task and creates ten temperature-`1.0` samples for every rephrased question. These responses are used only by BiG-SURE and its ablations.

3. Precompute DeBERTa bidirectional entailment probabilities:

```bash
bash scripts/compute/run_precompute_entailments.sh
```

The command writes one compressed `.npz` file per task/model/seed to `outputs/entailments/`. Each archive compares the original low-temperature answers against rephrased high-temperature answers.

4. Run BiG-SURE:

```bash
bash scripts/compute/run_bigsure.sh
```

This runs `snne/compute_spectral_energy_weighted_all_modes_rephrased.py` with SQuAD correctness, DeBERTa entailment probabilities, `entail_prob + min`, entropy-confidence weighting, and Jensen-Shannon divergence. Results are written to `outputs/spectral_energy/`.

5. Run the ablations:

```bash
bash scripts/compute/run_strawman_ablations.sh
```

This calls `snne/compute_strawman_baselines.py` on exactly the same rephrased responses, entailment files, and vanilla SQuAD labels as BiG-SURE. It writes to `outputs/strawman_ablations/`.

`config/runs.tsv` is the run manifest shared by stages 3-5. Both generation scripts create or update it automatically, with one row per task/model/seed and the corresponding local W&B directories. The compute launchers stop with a clear error when the manifest is empty and skip only incomplete rows or missing entailment files, which permits partial reruns.

The TriviaQA generator is explicitly given `../data/text_qa/triviaqa/llama_validation_trivia_qa.csv`; it no longer falls back to a randomly sampled validation subset. This keeps original and rephrased generations aligned.

## Retained Code

- `scripts/generate/`: original and rephrased generation.
- `scripts/compute/`: entailment precompute, BiG-SURE, and strawman ablations.
- `snne/compute_spectral_energy_weighted_all_modes_rephrased.py`: BiG-SURE implementation.
- `snne/compute_strawman_baselines.py`: ablation implementation.
- `snne/uncertainty/`: shared model, data, metric, and NLI runtime code required by the retained entry points.
