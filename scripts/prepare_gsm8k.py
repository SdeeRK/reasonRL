import re
import json
import os
from datasets import load_dataset


def main():
    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")

    # Read template
    with open("data/r1_zero_prompt.txt", "r") as f:
        template = f.read()

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    def process_split(split_name, output_file):
        print(f"Processing {split_name}...")
        data = dataset[split_name]

        with open(output_file, "w") as f:
            for item in data:
                question = item["question"]
                answer_raw = item["answer"]

                # Extract ground truth (everything after ####)
                match = re.search(r"####\s*(.*)", answer_raw)
                if match:
                    ground_truth = match.group(1).strip()
                    # Remove commas from numbers if present (often formatted as 1,000)
                    ground_truth = ground_truth.replace(",", "")
                else:
                    print(f"Warning: Could not extract answer from: {answer_raw}")
                    continue

                # Apply template
                prompt = template.format(question=question)

                # Create sample object
                sample = {"prompt": prompt, "ground_truth": ground_truth}

                f.write(json.dumps(sample) + "\n")
        print(f"Saved {len(data)} samples to {output_file}")

    process_split("train", "data/gsm8k_train.jsonl")
    process_split("test", "data/gsm8k_test.jsonl")


if __name__ == "__main__":
    main()
