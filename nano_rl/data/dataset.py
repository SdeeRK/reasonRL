import json

from torch.utils.data import Dataset

from ..core.types import Sample


class PromptDataset(Dataset):
    """
    JSONL dataset that loads prompts for GRPO training.

    Input: JSONL file with fields 'prompt', optional 'answer', optional 'ground_truth'.
    Output: Sample objects for DataLoader (use collate_fn=list for List[Sample] batches).
    """

    def __init__(self, data_path: str):
        self.samples: list[Sample] = []
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(
                    Sample(
                        prompt=item["prompt"],
                        answer=item.get("answer", None),
                        ground_truth=item.get("ground_truth", None),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Sample:
        return self.samples[idx]
