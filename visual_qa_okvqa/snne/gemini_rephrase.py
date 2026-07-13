import argparse
import pandas as pd
from pathlib import Path
import google.generativeai as genai
import os
import time


def build_model(model_name: str, api_key: str = None):
    """Initialize Gemini model with API key."""
    # Configure API key (preferred: env var `GOOGLE_API_KEY` or explicit api_key)
    if api_key:
        genai.configure(api_key=api_key)
    elif "GOOGLE_API_KEY" in os.environ:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    else:
        raise ValueError("API key must be provided via --api_key or GOOGLE_API_KEY environment variable")

    # Return a GenerativeModel instance. We'll pass generation config per-call.
    return genai.GenerativeModel(model_name)


def extract_questions(text: str):
    """Extract individual questions from generated text."""
    return [q.strip() for q in text.split('\n') if q.strip()]


def generate_variants(model, question: str, language: str = "French", max_retries: int = 3):
    """Generate 5 rephrased versions of the question in the specified language."""
    prompt = (
        f"Generate 5 rephrased versions of the following {language} statement while ensuring that the meaning is the EXACT same: "
        f"{question}\nYour response should contain the 5 statements in 5 different lines ONLY."
    )
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 1.0
                }
            )

            # Parse response similar to gemini metric parsing
            if getattr(response, 'candidates', None) and response.candidates[0].content.parts:
                # Extract text from the first candidate
                generated_text = response.candidates[0].content.parts[0].text
                if generated_text:
                    variants = extract_questions(generated_text)
                    return variants[:5]
                else:
                    print("  Warning: Empty text in candidate, retrying...")
            else:
                # Response blocked or empty
                finish_reason = None
                try:
                    finish_reason = response.candidates[0].finish_reason
                except Exception:
                    finish_reason = 'unknown'
                print(f"  Warning: Response blocked or incomplete (finish_reason: {finish_reason})")

            if attempt < max_retries - 1:
                print(f"  Retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                print(f"  Using original question as fallback")
                return [question] * 5

        except Exception as e:
            print(f"  Error generating variants: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"  Using original question as fallback")
                return [question] * 5

    return [question] * 5


def process_csv(input_csv: str, output_csv: str, model_name: str, language: str, api_key: str = None, checkpoint_interval: int = 10):
    """Process input CSV and generate rephrased questions."""
    df = pd.read_csv(input_csv)
    
    model = build_model(model_name, api_key)
    
    output_rows = []
    failed_ids = []
    
    for idx, row in df.iterrows():
        question = row.get("question", "")
        base_id = row.get("id")
        
        print(f"Processing question {idx + 1}/{len(df)}: {base_id}")
        
        try:
            variants = generate_variants(model, question, language)
            for i, variant in enumerate(variants, 1):
                new_id = f"{base_id}_rephrased{i}"
                output_rows.append({
                    "id": new_id,
                    "original_id": base_id,
                    "original_question": question,
                    "question": variant,
                    "rephrase_idx": i,
                })
        except Exception as e:
            print(f"  Failed to process question {base_id}: {e}")
            failed_ids.append(base_id)
            continue
        
        # Save checkpoint periodically
        if (idx + 1) % checkpoint_interval == 0:
            checkpoint_df = pd.DataFrame(output_rows)
            checkpoint_file = output_csv.replace('.csv', f'_checkpoint_{idx+1}.csv')
            checkpoint_df.to_csv(checkpoint_file, index=False)
            print(f"  Checkpoint saved: {checkpoint_file}")
    
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv, index=False)
    print(f"\nProcessed CSV saved as: {output_csv} with {len(output_rows)} rows")
    
    if failed_ids:
        print(f"Failed to process {len(failed_ids)} questions: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")


def main():
    parser = argparse.ArgumentParser(description="Rephrase French/German questions using Gemini and produce final output.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to base metadata CSV")
    parser.add_argument("--out_csv", type=str, default="rephrased_questions.csv", help="Path to write final CSV")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Gemini model name (e.g., gemini-1.5-flash, gemini-1.5-pro)")
    parser.add_argument("--language", type=str, default="French", choices=["French", "German"], help="Language of the questions")
    parser.add_argument("--api_key", type=str, default=None, help="Google API key (or set GOOGLE_API_KEY env variable)")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Save checkpoint every N questions")
    args = parser.parse_args()
    
    process_csv(args.input_csv, args.out_csv, args.model_name, args.language, args.api_key, args.checkpoint_interval)


if __name__ == "__main__":
    main()
