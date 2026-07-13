import argparse
import transformers
import torch
import pandas as pd
from pathlib import Path


def build_pipeline(model_id: str):
    return transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )


def extract_questions(text: str):
    return [q.strip() for q in text.split('\n') if q.strip()]


def generate_variants(pipeline, question: str, dataset_type: str = "general"):
    if dataset_type == "math":
        prompt = (
            "Rephrase the following math question in 5 distinct ways while ensuring the mathematical meaning, numbers, and core problem remain EXACTLY the same. "
            "Do not solve the problem. Just rephrase the question text.\n"
            f"Question: {question}\n"
            "Your response should contain the 5 rephrased questions in 5 different lines ONLY."
        )
        system_msg = "You are an assistant that rephrases math questions while keeping their mathematical meaning and values intact."
    else:
        # Standard VQA / General
        prompt = (
            "Generate 5 rephrased versions of the following question while ensuring that the meaning is the EXACT same: "
            f"{question}\nYour response should contain the 5 questions in 5 different lines ONLY."
        )
        system_msg = "You are an assistant that rephrases questions while keeping their meaning intact."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    outputs = pipeline(messages, max_new_tokens=512)
    generated_text = outputs[0]["generated_text"][-1]["content"]
    variants = extract_questions(generated_text)
    # In case the model returns more than 5 lines, keep first 5
    return variants[:5]


def process_csv(input_csv: str, output_csv: str, model_id: str, 
                question_col: str, id_col: str, image_path_col: str, dataset_type: str):
    df = pd.read_csv(input_csv)

    pipe = build_pipeline(model_id)

    output_rows = []
    for _, row in df.iterrows():
        question = str(row.get(question_col, ""))
        if not question or pd.isna(question):
            continue
            
        base_id = row.get(id_col)
        image_path = row.get(image_path_col, "")
        
        # Pass through other useful columns if present
        extra_cols = {k: v for k, v in row.items() if k not in [question_col, id_col, image_path_col]}

        variants = generate_variants(pipe, question, dataset_type)
        for i, variant in enumerate(variants, 1):
            new_id = f"{base_id}_rephrased{i}"
            row_data = {
                "id": new_id,
                "original_id": base_id,
                "original_question": question,
                "image_path": image_path,
                "question": variant,
                "rephrase_idx": i,
            }
            # Add extra columns back (e.g. answer, question_type)
            row_data.update(extra_cols)
            output_rows.append(row_data)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv, index=False)
    print(f"Processed CSV saved as: {output_csv} with {len(output_rows)} rows")


def main():
    parser = argparse.ArgumentParser(description="Rephrase questions for metadata CSV and produce final output.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to base metadata CSV")
    parser.add_argument("--out_csv", type=str, default="rephrased_questions.csv", help="Path to write final CSV")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="HF model id")
    
    # Column mapping args
    parser.add_argument("--question_col", type=str, default="question", help="Column name for question text")
    parser.add_argument("--id_col", type=str, default="id", help="Column name for ID")
    parser.add_argument("--image_path_col", type=str, default="image_path", help="Column name for image path")
    parser.add_argument("--dataset_type", type=str, default="general", choices=["general", "math"], help="Dataset type for prompt selection")

    args = parser.parse_args()

    process_csv(args.input_csv, args.out_csv, args.model_id, 
                args.question_col, args.id_col, args.image_path_col, args.dataset_type)


if __name__ == "__main__":
    main()
