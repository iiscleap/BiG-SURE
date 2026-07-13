# OKVQA Representative Set

This folder contains 5 representative original datapoints from the OKVQA dataset,
along with all their image perturbations and question rephrasings.

## Structure

Each sample folder contains:
- `README.md` — Human-readable summary of the datapoint
- `info.json` — Machine-readable metadata
- `images/` — Original image + all perturbed variants

## Samples

| # | Question ID | Question Type | Original Question | Answer |
|---|-------------|---------------|-------------------|--------|
| 1 | 4689545 | Other | What brand makes the gaming system shown? | nintendo |
| 2 | 3125245 | Cooking and Food | What flavour of cake is this? | vanilla |
| 3 | 1929325 | Sports and Recreation | What areas of the united states would this sport be common in? | north |
| 4 | 4496345 | Vehicles and Transportation | What city was this picture taken in? | moscow |
| 5 | 3040445 | Objects, Material and Clothing | How high off the ground are these traffic lights? | 15 feet |

## Perturbation Types

Each original image has 7 perturbation types applied:
- **contrast** — Contrast adjustment
- **blur** — Gaussian blur
- **rotate** — Image rotation
- **shift** — Spatial shift
- **noise** — Added noise
- **masking** — Region masking
- **bw** — Black & white conversion

Each perturbation has 5 rephrased versions of the original question.
