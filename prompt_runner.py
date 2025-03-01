import json
import os
import re
import time # Optional: for rate limiting
from dotenv import load_dotenv
from anthropic import Anthropic # Using Anthropic as per your setup

# --- Configuration ---
load_dotenv() 
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# LLM Configuration
# Find model names here: https://docs.anthropic.com/claude/docs/models-overview
LLM_MODEL_NAME = "claude-3-5-haiku-20241022" 
MAX_TOKENS_TO_SAMPLE = 700 # Max tokens for the LLM's response

# File Paths (ensure these match what your download script created)
BENCHMARK_DATA_DIR = "./benchmark_data"
FEW_SHOT_FILE_PATH = os.path.join(BENCHMARK_DATA_DIR, "gsm8k_main_few_shot_examples.jsonl")
DEV_SET_FILE_PATH = os.path.join(BENCHMARK_DATA_DIR, "gsm8k_main_train_dev_subset.jsonl")

# --- Utility Functions ---
def load_jsonl(file_path):
    """Loads a .jsonl file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

from typing import Optional

def extract_gsm8k_answer(text_output: str) -> Optional[str]:
    """x
    Extracts the final numerical answer from a string.
    GSM8K format is typically ".... #### <number>".
    """
    if not text_output:
        return None
    # The pattern looks for "#### " followed by a number, which can be an integer or have decimals, and might be negative.
    match = re.search(r"####\s*([-\d\.,]+)", text_output)
    if match:
        # Normalize by removing commas from numbers like "1,000"
        return match.group(1).replace(',', '').strip()
    
    # Fallback: if "####" is not found, try to extract the last number in the string.
    # This is less reliable and might need refinement based on observed LLM outputs.
    numbers = re.findall(r"[-+]?\s*[\d,]+\.?\d*|[-+]?\s*\.?[\d,]+", text_output) # Improved regex for numbers
    if numbers:
        # Get the last number found
        last_number_str = numbers[-1].replace(',', '').strip()
        # Further clean if it's just a dot or empty after stripping
        if last_number_str and last_number_str != '.':
            return last_number_str
            
    print(f"Warning: Could not extract final answer using '####' or fallback for: '{text_output[:100]}...'")
    return None

# --- Core Logic Functions ---
def create_gsm8k_prompt(question_to_solve: str, few_shot_examples: list) -> str:
    """
    Creates a few-shot chain-of-thought prompt for GSM8K.
    """
    prompt_string = "Solve the following math problems step-by-step. Your final answer should be demarcated with #### followed by the number.\n\n"
    for ex in few_shot_examples:
        prompt_string += f"Question: {ex['question']}\n"
        # The 'answer' field in our few_shot_examples.jsonl contains the CoT and #### marker
        prompt_string += f"Answer: {ex['answer']}\n\n" 
    
    prompt_string += f"Question: {question_to_solve}\n"
    prompt_string += "Answer:" # LLM will continue from here
    return prompt_string

def get_llm_completion(prompt_text: str, client: Anthropic) -> Optional[str]:
    """
    Sends a prompt to the configured LLM and returns the text response.
    """
    try:
        response = client.messages.create(
            model=LLM_MODEL_NAME,
            max_tokens=MAX_TOKENS_TO_SAMPLE,
            messages=[
                {"role": "user", "content": prompt_text}
            ]
        )
        # Assuming the response structure of Anthropic's API
        if response.content and len(response.content) > 0:
            return response.content[0].text
        else:
            print("Warning: LLM response content is empty.")
            return None
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None

def evaluate_gsm8k_response(llm_output_str: str, reference_final_answer_str: str) -> bool:
    """
    Compares the LLM's extracted final answer to the reference final answer.
    """
    if reference_final_answer_str is None:
        print("Warning: Reference final answer is None, cannot evaluate.")
        return False 

    predicted_answer_str = extract_gsm8k_answer(llm_output_str)
    if predicted_answer_str is None:
        print(f"Warning: Could not extract predicted answer from LLM output: {llm_output_str[:100]}...")
        return False

    try:
        # Convert both to floats for robust comparison, handling "100" vs "100.0"
        pred_float = float(predicted_answer_str)
        ref_float = float(reference_final_answer_str)
        # Using a small tolerance for float comparison. For GSM8K, direct equality after float conversion is often okay.
        return abs(pred_float - ref_float) < 1e-5 
    except ValueError:
        # If float conversion fails (e.g., if an answer is unexpectedly non-numeric after extraction)
        print(f"Warning: ValueError during float conversion. Predicted: '{predicted_answer_str}', Reference: '{reference_final_answer_str}'. Comparing as strings.")
        return predicted_answer_str.strip() == reference_final_answer_str.strip()

# --- Main Evaluation Function ---
def run_baseline_evaluation(
    dev_set_path: str, 
    few_shot_examples_list: list, 
    llm_client: Anthropic
) -> tuple[float, list]:
    """
    Runs the baseline prompt over the development set and calculates accuracy.
    Returns:
        - accuracy (float): The percentage of correct predictions.
        - failure_cases (list): A list of dictionaries, each detailing a failure.
    """
    dev_set = load_jsonl(dev_set_path)
    
    correct_predictions = 0
    total_predictions = 0
    failure_cases = []

    print(f"\nStarting baseline evaluation on {len(dev_set)} examples from '{dev_set_path}'...")
    print(f"Using LLM Model: {LLM_MODEL_NAME}")

    for i, record in enumerate(dev_set):
        question_id = record["id"]
        question_to_solve = record["question"]
        reference_final_answer = record["reference_final_answer"]
        reference_full_solution = record["reference_answer_details"]

        print(f"\n[{i+1}/{len(dev_set)}] Processing: {question_id} - Q: {question_to_solve[:60]}...")

        # 1. Create the full prompt
        full_prompt = create_gsm8k_prompt(question_to_solve, few_shot_examples_list)
        
        # 2. Get LLM response
        llm_output = get_llm_completion(full_prompt, llm_client)

        if llm_output:
            is_correct = evaluate_gsm8k_response(llm_output, reference_final_answer)
            if is_correct:
                correct_predictions += 1
            else:
                failure_cases.append({
                    "id": question_id,
                    "question": question_to_solve,
                    "reference_solution": reference_full_solution,
                    "reference_final_answer": reference_final_answer,
                    "generated_prompt": full_prompt, # For debugging the prompt itself
                    "llm_output": llm_output,
                    "extracted_prediction": extract_gsm8k_answer(llm_output)
                })
            print(f"  Predicted Correct: {is_correct} (Ref: {reference_final_answer}, LLM_Extracted: {extract_gsm8k_answer(llm_output)})")
        else:
            print(f"  Failed to get LLM response for {question_id}.")
            failure_cases.append({
                "id": question_id,
                "question": question_to_solve,
                "reference_solution": reference_full_solution,
                "reference_final_answer": reference_final_answer,
                "generated_prompt": full_prompt,
                "llm_output": "ERROR_NO_RESPONSE",
                "extracted_prediction": None
            })
        
        total_predictions += 1
        
        # Optional: Add a small delay to respect API rate limits if any
        # Consider the limits for your chosen model (e.g., Claude API has rate limits)
        # time.sleep(1) # Adjust or remove based on your API plan and model

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    print(f"\n--- Baseline Evaluation Summary ---")
    print(f"Total Examples Processed: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return accuracy, failure_cases

# --- Script Execution ---
if __name__ == "__main__":
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found in environment variables.")
        print("Please create a .env file in the project root with your API key (e.g., ANTHROPIC_API_KEY='your_key').")
    else:
        # Initialize LLM client
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # Load few-shot examples
        print(f"Loading few-shot examples from: {FEW_SHOT_FILE_PATH}")
        few_shot_examples = load_jsonl(FEW_SHOT_FILE_PATH)
        if not few_shot_examples:
            print(f"Error: No few-shot examples loaded from {FEW_SHOT_FILE_PATH}. Please check the file and path.")
        else:
            print(f"Loaded {len(few_shot_examples)} few-shot examples.")
            
            # Run the baseline evaluation
            baseline_accuracy, failures = run_baseline_evaluation(
                DEV_SET_FILE_PATH, 
                few_shot_examples, 
                client
            )
            
            print(f"\nFinal Baseline Accuracy: {baseline_accuracy:.2f}%")

            if failures:
                print(f"\nCollected {len(failures)} failure cases. First few failures:")
                for i, failure in enumerate(failures[:3]): # Print details of first 3 failures
                    print(f"  Failure {i+1}: ID={failure['id']}")
                    print(f"    Q: {failure['question'][:80]}...")
                    print(f"    Ref_Answer: {failure['reference_final_answer']}")
                    print(f"    LLM_Raw_Output (snippet): {failure['llm_output'][:100] if failure['llm_output'] else 'N/A'}...")
                    print(f"    LLM_Extracted: {failure['extracted_prediction']}")