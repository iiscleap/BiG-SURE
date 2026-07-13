# BiG-SURE Data

This folder contains the dataset payload for the consolidated BiG-SURE codebase. It is designed to be uploaded to Drive separately from the source code.

## Layout

- `text_qa/triviaqa/`: TriviaQA original and rephrased CSVs.
- `text_qa/svamp/`: SVAMP original and rephrased CSVs.
- `visual_qa/okvqa/`: OKVQA 200-sample subset with metadata, perturbation CSVs, `subset.csv`, blank image, representative examples, 200 original images, and 1400 referenced perturbed images.
- `multilingual_sciq/sciq/`: multilingual SciQ JSONs and rephrased/example files.

## Setup

After downloading this folder, place it at:

```bash
BiG-SURE/data
```

The maintained task launchers read these canonical paths directly; no symlinks are required.
