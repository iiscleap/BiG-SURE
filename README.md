# BiG-SURE

Official implementation of the EMNLP 2026 main-conference paper, **BiG-SURE: Bipartite Graph Spectral Energy for Uncertainty and Reliability Estimation of LLMs**.

BiG-SURE is the consolidated codebase for uncertainty estimation with stochastic generations, standard baselines, and BiG-SURE spectral energy across three task settings:

- Text QA: TriviaQA and SVAMP.
- Visual QA: OKVQA.
- Multilingual QA: SciQ in English, Chinese, Japanese, and French.

The provided launch scripts run seed `10` only to keep the default reproduction manageable. The numbers reported in the paper aggregate all five experimental seeds: `10`, `20`, `30`, `40`, and `50`.

This README only covers environment setup and data placement. Use the task README for generation, evaluation, baseline, entailment, and BiG-SURE run instructions:

- [Text QA](text_qa/README.md)
- [OKVQA Visual QA](visual_qa_okvqa/README.md)
- [Multilingual SciQ](multilingual_sciq/README.md)

## Score a New Dataset or Model from CSV

`run_uncertainty.py` computes per-example BiG-SURE and baseline uncertainty
scores from model generations stored in a CSV. A 100-example SciQ/Apertus input
is provided in `demo_input_100.csv`.

```bash
conda activate snne
python run_uncertainty.py \
  --input demo_input_100.csv \
  --output demo_scores.csv \
  --measures all \
  --input_aug True \
  --device cuda
```

Set `--input_aug False` to compute BiG-SURE from stochastic generations on the
direct question instead of the paraphrased-question generations. In that mode,
`sampled_responses` supplies the ten high-temperature responses and
`rephrased_responses`/`rephrase_ids` are not required.

The required CSV column is `id`. Other columns are JSON arrays and are needed
according to the selected measure: `sampled_responses`,
`sampled_probabilities`, `low_temperature_responses`, `rephrased_responses`,
and `rephrase_ids`. With the default `--input_aug True`, the paper configuration
stores 50 `rephrased_responses`: ten responses for each of five values in the
aligned `rephrase_ids` array. An optional binary `correct` column enables
error-AUROC evaluation.

### Multimodal Input

Place images beside the CSV in an `images/` directory and add an `image_path`
column containing paths relative to the CSV. For example:

```text
multimodal_input/
  input.csv
  images/
    example_001.jpg
    example_002.jpg
```

```bash
python run_uncertainty.py \
  --input multimodal_input/input.csv \
  --output multimodal_scores.csv \
  --measures all \
  --multimodal True \
  --input_aug True \
  --image_augs blur,noise \
  --device cuda
```

The direct runner scores precomputed model generations; it does not load a VLM
or create responses from the images. A multimodal CSV must therefore provide
the responses generated from each augmented image in `augmented_responses`
(or `rephrased_responses`), with aligned augmentation names in
`image_augmentation_ids` (or `augmentation_ids`). Allowed names are `contrast`,
`blur`, `rotate`, `shift`, `noise`, `masking`, and `bw`. When text paraphrasing
is enabled, `rephrase_ids` must also align with those arrays.

`--image_augs all` is the default. The runner selects exactly ten stochastic
responses per direct or paraphrased input and balances them over the requested
image augmentations. Thus `--image_augs blur,noise` selects five responses from
each augmentation. If ten is not evenly divisible by the number selected, the
earlier augmentations receive one additional response. The CSV must contain at
least the required number of precomputed responses for every selected
augmentation and input group.

Arguments:

- `--input`: input CSV path.
- `--output`: per-example output CSV path.
- `--measures`: `all` or a comma-separated selection of `bigsure`,
  `semantic_entropy`, `predictive_entropy`, `num_semantic_sets`,
  `lexical_similarity`, `graph_degree`, `graph_eigenvalue`, and `snne`.
- `--nli-model`: Hugging Face model name or local NLI checkpoint.
- `--device`: `auto`, `cpu`, or `cuda`.
- `--batch-size`: NLI batch size; defaults to `128`.
- `--max-rows`: optionally score only the first N rows for a quick test.
- `--input_aug`: `True` (default) uses paraphrased input generations; `False`
  uses direct-question stochastic generations.
- `--multimodal`: `False` by default; set to `True` for image-backed examples.
- `--image_augs`: comma-separated image augmentations used in multimodal mode;
  defaults to `all`. The plural `--input_augs` spelling is accepted as an alias;
  it is distinct from the singular boolean `--input_aug` option.

BiG-SURE requires low-temperature answers plus grouped responses generated from
rephrased or perturbed prompts. The paper configuration uses three
low-temperature answers and ten stochastic answers for each of five rephrases.
Larger output values indicate greater uncertainty. When `correct` contains both
binary classes, the script also writes `<output-name>_summary.csv` with AUROC for
detecting errors.

## Environment Setup

Use the existing `snne` Conda environment for Text QA, multilingual SciQ, and the LLaVA/Qwen3-VL OKVQA runs. If it must be recreated, use `environments/snne.yml`. Install PyTorch first with the command from the [PyTorch selector](https://pytorch.org/get-started/locally/), then install the repository dependencies:

```bash
# Existing shared environment
conda activate snne

# Only when recreating it on a new machine:
# conda env create -f environments/snne.yml

# Install the CUDA-compatible torch and torchvision build selected for this machine,
# then install the remaining repository dependencies.
pip install -r requirements.txt
```

The dependency versions in `requirements.txt` match the validated shared stack, including `transformers==4.57.0`, which is required by multilingual inference.

### Pixtral Exception

Pixtral generation requires its separate `snne_pixtral` environment because it uses the pinned vLLM/Mistral stack in `environments/snne_pixtral.yml` (`vllm`, `mistral-common`, Torch 2.4, and `transformers 4.45.2`). Create it once with:

```bash
conda env create -f environments/snne_pixtral.yml
```

Use `snne_pixtral` only for the Pixtral entries in the OKVQA generation launchers. All OKVQA evaluation, baseline, entailment, and BiG-SURE scripts use the regular `snne` environment.

Authenticate with Hugging Face before generating from gated models such as Llama or Gemma:

```bash
huggingface-cli login
```

Generation is tracked with Weights & Biases in separate projects for each task. Log in for online tracking, or select offline mode before running any generation launcher:

```bash
# Online
wandb login

# Or fully local/offline
export WANDB_MODE=offline
```

The default projects are `bigsure-text-qa`, `bigsure-okvqa`, and `bigsure-multilingual-sciq`. Override a project with `WANDB_PROJECT`, and set `WANDB_ENTITY` when runs must belong to a specific team. W&B files stay inside each task's `outputs/wandb/` directory in both online and offline modes.

Multilingual SciQ Gemini evaluation is optional. It is the only stage requiring an API key:

```bash
export GOOGLE_API_KEY=...
```

### SNNE Module Paths

`snne` is intentionally source-imported rather than installed as one global Python package. Text QA and OKVQA contain task-local packages with the same name but different model and data adapters; installing both into one environment would cause the later installation to overwrite the former.

Use the supplied task launchers, which set `PYTHONPATH` correctly. For direct Python execution from the repository root, use only one of these path configurations at a time:

```bash
# Text QA
export PYTHONPATH="$PWD/text_qa:${PYTHONPATH:-}"

# OKVQA Visual QA
export PYTHONPATH="$PWD/visual_qa_okvqa:${PYTHONPATH:-}"

# Multilingual SciQ: shared Text QA SNNE helpers plus multilingual utilities
export PYTHONPATH="$PWD/text_qa:$PWD/multilingual_sciq/baselines:${PYTHONPATH:-}"
```

The shared Text QA package provides the KLE and common uncertainty helpers used by the multilingual baseline scripts. The OKVQA package is a visual-language fork with image-aware generators and VQA evaluation support. Multilingual SciQ uses its own JSON generation and evaluation code, then imports the shared Text QA KLE/uncertainty helpers through its launchers.

Verify the environment without downloading models:

```bash
PYTHONPATH="$PWD/text_qa" python -c "from snne.kle.core import vn_entropy; print('text SNNE OK')"
PYTHONPATH="$PWD/visual_qa_okvqa" python -c "from snne.uncertainty.utils.metric_utils import get_metric; print('visual SNNE OK')"
PYTHONPATH="$PWD/text_qa:$PWD/multilingual_sciq/baselines" python -c "import multilingual_utils; from snne.kle.core import vn_entropy; print('multilingual helpers OK')"
```

## Data Installation

The separately distributed data bundle must be extracted as `data/` at the repository root. See [data/README.md](data/README.md) for its contents and placement instructions. The resulting layout is:

```text
BiG-SURE/
  data/
    text_qa/
      triviaqa/
      svamp/
    visual_qa/
      okvqa/
        images/
        images_perturbed/
    multilingual_sciq/
      sciq/
  text_qa/
  visual_qa_okvqa/
  multilingual_sciq/
```

The task launchers use these canonical paths:

- `data/text_qa/triviaqa/` and `data/text_qa/svamp/`
- `data/visual_qa/okvqa/`
- `data/multilingual_sciq/sciq/`

No data symlinks or path edits are required. Every maintained launcher derives the repository and data roots from its own location. Run a launcher from any working directory; use `PYTHON_BIN=/path/to/python` only when selecting a Python executable without activating its Conda environment first.

## Acknowledgements and Code Provenance

This repository is substantially based on and adapted from [SNNE](https://github.com/BigML-CS-UCLA/SNNE), the official implementation of *Beyond Semantic Entropy: Boosting LLM Uncertainty Quantification with Pairwise Semantic Similarity*. We gratefully acknowledge the authors for making that work available. BiG-SURE reuses and extends SNNE's answer-generation pipeline, uncertainty-quantification utilities, semantic-similarity infrastructure, and baseline implementations. This repository adds the task-specific Text QA, multilingual SciQ, and OKVQA workflows; artifact validation and entailment precomputation; and the BiG-SURE spectral-energy method.

SNNE itself builds on several open-source projects, whose contributions are also part of this codebase's technical lineage:

- [Semantic Uncertainty](https://github.com/jlko/semantic_uncertainty), which informed the SNNE repository structure and semantic-uncertainty pipeline.
- [UQ-NLG](https://github.com/zlin7/UQ-NLG), from which SNNE adapted its graph-based uncertainty baselines.
- [LM-Polygraph](https://github.com/IINemo/lm-polygraph), from which SNNE adapted summarization and translation components.

We thank the authors and maintainers of SNNE and these upstream projects for releasing their code. Please retain this acknowledgement when redistributing derived versions of BiG-SURE, and follow the license and citation requirements of the relevant upstream projects.

### SNNE Citation

If you use the SNNE-derived components in this repository, please cite the original paper:

```bibtex
@article{nguyen2025beyond,
  title={Beyond Semantic Entropy: Boosting LLM Uncertainty Quantification with Pairwise Semantic Similarity},
  author={Nguyen, Dang and Payani, Ali and Mirzasoleiman, Baharan},
  journal={In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2025}
}
```
