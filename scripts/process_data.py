import re

from datasets import load_dataset
from tyro import cli

from reason_rl.data.parser import DataParser


def process_gsm8k(item):
    question = item["question"]
    answer_raw = item["answer"]

    # Extract ground truth (everything after ####)
    match = re.search(r"####\s*(.*)", answer_raw)
    if match:
        ground_truth = match.group(1).strip()
        # Remove commas from numbers if present
        ground_truth = ground_truth.replace(",", "")
    else:
        return None

    return {"question": question, "ground_truth": ground_truth, "answer": answer_raw}


def process_dapo(item):
    # Extract prompt
    raw_prompt = item.get("prompt")
    question = ""
    if isinstance(raw_prompt, list) and len(raw_prompt) > 0:
        first_msg = raw_prompt[0]
        if isinstance(first_msg, dict):
            question = first_msg.get("content", "")
    elif isinstance(raw_prompt, str):
        question = raw_prompt

    # Extract ground truth
    reward_model = item.get("reward_model", {})
    ground_truth = None
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")

    if not question or not ground_truth:
        return None

    return {"question": question, "ground_truth": ground_truth, "answer": str(ground_truth)}


def main(dataset: str = "gsm8k", template: str = "data/r1_zero_prompt.txt"):
    data_parser = DataParser(template_path=template)

    if dataset == "gsm8k":
        data_parser.process_and_save(
            dataset_name="gsm8k",
            subset="main",
            output_dir="data/gsm8k",
            processor=process_gsm8k,
            split_mapping={"train": "train", "test": "test"},
        )
    elif dataset == "dapo":
        print("Loading DAPO dataset...")
        dataset_obj = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")

        # Deduplication
        print("Deduplicating DAPO dataset...")
        from datasets import Dataset

        all_data = [item for item in dataset_obj["train"]]
        seen = set()
        unique_data = []
        for item in all_data:
            # Make key
            p = str(item["prompt"])
            gt = str(item.get("reward_model", {}).get("ground_truth"))
            key = (p, gt)
            if key not in seen:
                seen.add(key)
                unique_data.append(item)

        print(f"Reduced from {len(all_data)} to {len(unique_data)} samples.")
        dedup_ds = Dataset.from_list(unique_data)

        # Split train into train/test
        split_ds = dedup_ds.train_test_split(test_size=500, seed=42)

        data_parser.process_and_save(
            dataset_name="dapo",
            subset=None,
            output_dir="data/dapo",
            processor=process_dapo,
            split_mapping={"train": "train", "test": "test"},
            dataset=split_ds,
        )


if __name__ == "__main__":
    cli(main)
