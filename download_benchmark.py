from datasets import load_dataset
import json
import os
import re

# --- Configuration ---
CONFIG_NAME = "main"
SPLIT_NAME = "train"

# Number of examples to save in local subset (for development/optimization)
NUM_EXAMPLES_TO_SAVE = 100

# Local directory to save the benchmark data
SAVE_DIR = "./benchmark_data"
os.makedirs(SAVE_DIR, exist_ok=True)
FILE_PATH = os.path.join(SAVE_DIR, f"gsm8k_{CONFIG_NAME}_{SPLIT_NAME}_subset.jsonl")

# --- Utility Function ---
def extract_gsm8k_answer(answer_str):
    """
    Extracts the final numerical answer from the GSM8K answer string.
    The format is typically ".... #### <number>".
    """
    # The pattern looks for "#### " followed by a number, which can be an integer or have decimals, and might be negative.
    # It captures the number part.
    match = re.search(r"####\s*([-\d\.,]+)", answer_str)
    if match:
        # Normalize by removing commas from numbers like "1,000"
        return match.group(1).replace(',', '')
    return None # Or raise an error, or return a special value

# --- Main Script ---
def download_and_prepare_gsm8k(config_name, split_name, num_examples, save_path):
    """
    Downloads the GSM8K dataset, prints info, and saves a subset locally.
    """
    print(f"Attempting to download GSM8K (config: {config_name}, split: {split_name})...")

    try:
        dataset = load_dataset("gsm8k", config_name, split=split_name, trust_remote_code=True)
        print(f"\nSuccessfully loaded dataset: GSM8K (config: {config_name}, split: {split_name})")

        print("\n--- Dataset Info ---")
        print(f"Number of examples in this split: {len(dataset)}")
        print(f"Features: {dataset.features}")

        print("\n--- First Example ---")
        if len(dataset) > 0:
            example = dataset[0]
            print(f"Question: {example['question']}")
            print(f"Answer (contains chain-of-thought and final answer): {example['answer']}")
            extracted_final_answer = extract_gsm8k_answer(example['answer'])
            print(f"Extracted Final Answer (from reference): {extracted_final_answer}")

        # Prepare and save a subset of examples
        print(f"\nSaving {min(num_examples, len(dataset))} examples to {save_path}...")
        saved_count = 0
        with open(save_path, 'w') as f:
            for i in range(min(num_examples, len(dataset))):
                record = dataset[i]
                # Extract the final answer from the reference solution for easier evaluation later
                final_answer_from_reference = extract_gsm8k_answer(record['answer'])
                
                record_to_save = {
                    "id": f"gsm8k_{config_name}_{split_name}_ex_{i}",
                    "question": record["question"],
                    "reference_answer_details": record["answer"], # Keep the full CoT
                    "reference_final_answer": final_answer_from_reference
                }
                f.write(json.dumps(record_to_save) + "\n")
                saved_count += 1
        print(f"Successfully saved {saved_count} examples to {save_path}")
        print(f"\nEach line in '{save_path}' is a JSON object containing:")
        print("  'id': A unique identifier for the example.")
        print("  'question': The math word problem.")
        print("  'reference_answer_details': The full reference solution including chain of thought.")
        print("  'reference_final_answer': The extracted final numerical answer from the reference.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_and_prepare_gsm8k(CONFIG_NAME, SPLIT_NAME, NUM_EXAMPLES_TO_SAVE, FILE_PATH)