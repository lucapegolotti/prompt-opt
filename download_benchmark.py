from datasets import load_dataset
import json
import os
import re

# --- Configuration ---
CONFIG_NAME = "main"
SPLIT_NAME = "train"

# Number of examples for the few-shot prompt demonstrations
NUM_FEW_SHOT_EXAMPLES = 10 
# Number of examples for your main development/optimization set
NUM_DEV_EXAMPLES = 100

# Local directory to save the benchmark data
SAVE_DIR = "./benchmark_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# File paths for the saved datasets
FEW_SHOT_FILE_PATH = os.path.join(SAVE_DIR, f"gsm8k_{CONFIG_NAME}_few_shot_examples.jsonl")
DEV_SET_FILE_PATH = os.path.join(SAVE_DIR, f"gsm8k_{CONFIG_NAME}_{SPLIT_NAME}_dev_subset.jsonl")

# --- Utility Function (remains the same) ---
def extract_gsm8k_answer(answer_str):
    """
    Extracts the final numerical answer from the GSM8K answer string.
    The format is typically ".... #### <number>".
    """
    match = re.search(r"####\s*([-\d\.,]+)", answer_str)
    if match:
        return match.group(1).replace(',', '')
    return None

# --- Main Script ---
def download_and_prepare_gsm8k_splits(
    config_name,
    split_name,
    num_few_shot,
    few_shot_save_path,
    num_dev,
    dev_set_save_path
):
    """
    Downloads the GSM8K dataset, saves a distinct set for few-shot examples,
    and another distinct set for development/optimization.
    """
    print(f"Attempting to download GSM8K (config: {config_name}, split: {split_name})...")

    try:
        # Load the specified split from the dataset
        dataset = load_dataset("gsm8k", config_name, split=split_name, trust_remote_code=True)
        print(f"\nSuccessfully loaded dataset: GSM8K (config: {config_name}, split: {split_name})")
        print(f"Total examples in '{split_name}' split: {len(dataset)}")

        if len(dataset) < num_few_shot + num_dev:
            print(f"Warning: Dataset size ({len(dataset)}) is less than the sum of requested few-shot examples ({num_few_shot}) and dev examples ({num_dev}). Adjust numbers if needed.")
            # Potentially cap the numbers here if desired, or let it proceed and fail if out of bounds.
            # For simplicity, we'll assume it's large enough or the user will adjust.

        # --- Save Few-Shot Examples ---
        print(f"\nSaving {num_few_shot} examples for few-shot prompts to {few_shot_save_path}...")
        saved_count_fs = 0
        with open(few_shot_save_path, 'w') as f:
            for i in range(min(num_few_shot, len(dataset))):
                record = dataset[i]
                final_answer_from_reference = extract_gsm8k_answer(record['answer'])
                record_to_save = {
                    "id": f"gsm8k_{config_name}_{split_name}_fs_ex_{i}",
                    "question": record["question"],
                    "answer": record["answer"], # Save the full answer for few-shot CoT
                    "reference_final_answer": final_answer_from_reference # For reference if needed
                }
                f.write(json.dumps(record_to_save) + "\n")
                saved_count_fs += 1
        print(f"Successfully saved {saved_count_fs} few-shot examples.")

        # --- Save Development Set Examples ---
        # These examples will start *after* the few-shot examples
        start_index_dev = num_few_shot
        end_index_dev = start_index_dev + num_dev
        
        print(f"\nSaving {num_dev} examples for development/optimization set to {dev_set_save_path}...")
        print(f"(Taking examples from index {start_index_dev} to {min(end_index_dev, len(dataset)) -1})")
        
        saved_count_dev = 0
        with open(dev_set_save_path, 'w') as f:
            for i in range(start_index_dev, min(end_index_dev, len(dataset))):
                record = dataset[i]
                final_answer_from_reference = extract_gsm8k_answer(record['answer'])
                record_to_save = {
                    "id": f"gsm8k_{config_name}_{split_name}_dev_ex_{i - start_index_dev}", # Adjust ID to be 0-indexed for the subset
                    "question": record["question"],
                    "reference_answer_details": record["answer"],
                    "reference_final_answer": final_answer_from_reference
                }
                f.write(json.dumps(record_to_save) + "\n")
                saved_count_dev += 1
        print(f"Successfully saved {saved_count_dev} development set examples.")

        print("\nData preparation complete.")
        print(f"Few-shot examples are in: '{few_shot_save_path}'")
        print(f"Development set examples are in: '{dev_set_save_path}'")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the 'datasets' library installed ('pip install datasets regex').")

if __name__ == "__main__":
    download_and_prepare_gsm8k_splits(
        CONFIG_NAME,
        SPLIT_NAME,
        NUM_FEW_SHOT_EXAMPLES,
        FEW_SHOT_FILE_PATH,
        NUM_DEV_EXAMPLES,
        DEV_SET_FILE_PATH
    )