#!/usr/bin/env python3
"""Create a representative set folder with 5 original datapoints and their perturbations/rephrasings."""

import csv
import os
import shutil
import json
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parents[2])
OKVQA_DIR = os.path.join(BASE_DIR, "snne/okvqa")
OUTPUT_DIR = os.path.join(OKVQA_DIR, "representative_set")

# Read metadata.csv
metadata = []
with open(os.path.join(OKVQA_DIR, "metadata.csv"), "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        metadata.append(row)

print(f"Total original datapoints: {len(metadata)}")

# Pick 5 diverse datapoints from different question types
question_types_seen = set()
selected = []
# Prioritize diverse question types
for row in metadata:
    qtype = row["question_type"]
    if qtype not in question_types_seen and len(selected) < 5:
        selected.append(row)
        question_types_seen.add(qtype)
    if len(selected) == 5:
        break

print(f"\nSelected {len(selected)} datapoints:")
for s in selected:
    print(f"  question_id={s['question_id']}, type={s['question_type']}, question={s['question']}")

# Read rephrased_perturbations.csv
perturbations = []
with open(os.path.join(OKVQA_DIR, "rephrased_perturbations.csv"), "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        perturbations.append(row)

print(f"\nTotal perturbation rows: {len(perturbations)}")

# Build lookup: question_id -> list of perturbation rows
selected_qids = {s["question_id"] for s in selected}
perturb_by_qid = {}
for row in perturbations:
    qid = row["question_id"]
    if qid in selected_qids:
        perturb_by_qid.setdefault(qid, []).append(row)

# Create output directory
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# For each selected datapoint, create a subfolder
for idx, orig in enumerate(selected, 1):
    qid = orig["question_id"]
    # Create folder named by index and short description
    folder_name = f"sample_{idx}_qid_{qid}"
    sample_dir = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Copy original image
    orig_img_rel = orig["image_path"]  # e.g. snne/okvqa/images/OKVQA_...jpg
    orig_img_src = os.path.join(BASE_DIR, orig_img_rel)
    orig_img_name = os.path.basename(orig_img_rel)
    
    images_dir = os.path.join(sample_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    if os.path.exists(orig_img_src):
        shutil.copy2(orig_img_src, os.path.join(images_dir, orig_img_name))
        print(f"  Copied original: {orig_img_name}")
    else:
        print(f"  WARNING: Original image not found: {orig_img_src}")
    
    # Copy perturbed images and gather info
    perturb_rows = perturb_by_qid.get(qid, [])
    perturbation_info = {}  # perturbation_name -> {image, rephrasings}
    
    for prow in perturb_rows:
        pname = prow["perturbation_name"]
        ptype = prow["perturbation_type"]
        pintensity = prow["perturbation_intensity"]
        rephrase_idx = prow["rephrase_idx"]
        
        if pname not in perturbation_info:
            # Copy perturbed image
            perturb_img_rel = prow["image_path"]
            perturb_img_src = os.path.join(BASE_DIR, perturb_img_rel)
            perturb_img_name = os.path.basename(perturb_img_rel)
            
            if os.path.exists(perturb_img_src):
                shutil.copy2(perturb_img_src, os.path.join(images_dir, perturb_img_name))
            
            perturbation_info[pname] = {
                "perturbation_type": ptype,
                "perturbation_intensity": pintensity,
                "perturbed_image": perturb_img_name,
                "rephrasings": []
            }
        
        perturbation_info[pname]["rephrasings"].append({
            "rephrase_idx": rephrase_idx,
            "rephrased_question": prow["question"],
            "id": prow["id"]
        })
    
    # Write info JSON
    info = {
        "question_id": qid,
        "question_type": orig["question_type"],
        "original_question": orig["question"],
        "multiple_choice_answer": orig["multiple_choice_answer"],
        "answers_joined": orig["answers_joined"],
        "original_image": orig_img_name,
        "num_perturbations": len(perturbation_info),
        "total_rephrasings": len(perturb_rows),
        "perturbations": perturbation_info
    }
    
    with open(os.path.join(sample_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    # Write a human-readable README
    with open(os.path.join(sample_dir, "README.md"), "w") as f:
        f.write(f"# Sample {idx}: Question ID {qid}\n\n")
        f.write(f"**Question Type:** {orig['question_type']}\n\n")
        f.write(f"**Original Question:** {orig['question']}\n\n")
        f.write(f"**Answer:** {orig['multiple_choice_answer']}\n\n")
        f.write(f"**All Answers:** {orig['answers_joined']}\n\n")
        f.write(f"**Original Image:** `images/{orig_img_name}`\n\n")
        f.write(f"---\n\n")
        f.write(f"## Perturbations ({len(perturbation_info)} types)\n\n")
        
        for pname, pinfo in sorted(perturbation_info.items()):
            f.write(f"### {pname} (type: {pinfo['perturbation_type']}, intensity: {pinfo['perturbation_intensity']})\n\n")
            f.write(f"**Perturbed Image:** `images/{pinfo['perturbed_image']}`\n\n")
            f.write(f"**Rephrasings:**\n\n")
            for r in sorted(pinfo["rephrasings"], key=lambda x: int(x["rephrase_idx"])):
                f.write(f"{r['rephrase_idx']}. {r['rephrased_question']}\n")
            f.write(f"\n")
    
    print(f"  Created {folder_name}: {len(perturbation_info)} perturbations, {len(perturb_rows)} total rephrasings")

# Write a top-level README
with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write("# OKVQA Representative Set\n\n")
    f.write("This folder contains 5 representative original datapoints from the OKVQA dataset,\n")
    f.write("along with all their image perturbations and question rephrasings.\n\n")
    f.write("## Structure\n\n")
    f.write("Each sample folder contains:\n")
    f.write("- `README.md` — Human-readable summary of the datapoint\n")
    f.write("- `info.json` — Machine-readable metadata\n")
    f.write("- `images/` — Original image + all perturbed variants\n\n")
    f.write("## Samples\n\n")
    f.write("| # | Question ID | Question Type | Original Question | Answer |\n")
    f.write("|---|-------------|---------------|-------------------|--------|\n")
    for idx, orig in enumerate(selected, 1):
        f.write(f"| {idx} | {orig['question_id']} | {orig['question_type']} | {orig['question']} | {orig['multiple_choice_answer']} |\n")
    f.write("\n## Perturbation Types\n\n")
    f.write("Each original image has 7 perturbation types applied:\n")
    f.write("- **contrast** — Contrast adjustment\n")
    f.write("- **blur** — Gaussian blur\n")
    f.write("- **rotate** — Image rotation\n")
    f.write("- **shift** — Spatial shift\n")
    f.write("- **noise** — Added noise\n")
    f.write("- **masking** — Region masking\n")
    f.write("- **bw** — Black & white conversion\n\n")
    f.write("Each perturbation has 5 rephrased versions of the original question.\n")

print(f"\nDone! Representative set created at: {OUTPUT_DIR}")
