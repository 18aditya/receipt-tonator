import json
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from src.utils import json2token

class ReceiptDataset(Dataset):
    def __init__(self, dataset_name_or_path, max_length, processor, split="train", ignore_id=-100, task_start_token="<s_receipt>"):
        super().__init__()
        self.processor = processor
        self.ignore_id = ignore_id
        self.max_length = max_length
        self.gt_token_sequences = []
        self.task_start_token = task_start_token
        
        # Load dataset
        self.dataset = load_dataset(dataset_name_or_path, split=split)
        
        # We need to collect special tokens to update the tokenizer later
        self.new_special_tokens = set()
        
        def update_special_tokens_fn(key):
            self.new_special_tokens.add(fr"<s_{key}>")
            self.new_special_tokens.add(fr"</s_{key}>")
        
        # Pre-process ground truth to simple token sequences
        for item in self.dataset:
            json_obj = json.loads(item["text"])
            token_sequence = json2token(json_obj, update_special_tokens_fn) + self.processor.tokenizer.eos_token
            self.gt_token_sequences.append(token_sequence)
            
        self.add_tokens([self.task_start_token])
        self.add_tokens(list(self.new_special_tokens))

    def add_tokens(self, list_of_tokens):
        """
        Add tokens to the tokenizer and resize the model token embeddings
        """
        newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)

            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get image
        image = item["image"].convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Get labels
        target_sequence = self.gt_token_sequences[idx]
        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        # Mask pad tokens in the labels
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id
        
        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": labels,
            "target_sequence": target_sequence
        }
