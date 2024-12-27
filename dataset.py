from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
import torch

class LMTextDataset(Dataset):
    """
    A dataset that:
    - Takes a list of strings (text samples).
    - Tokenizes them using a GPT-2 tokenizer.
    - Concatenates them into one long token sequence.
    - Chunks them into fixed-length inputs.
    """
    def __init__(self, texts, tokenizer: GPT2Tokenizer, block_size=256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        full_text = "\n".join(texts)

        # Tokenize and concatenate all texts into one long tensor
        encodings = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        # Flatten all tokens into a 1D array
        self.input_ids = encodings["input_ids"].view(-1)
        
        total_length = self.input_ids.size(0)
        # Drop the remainder so we have an exact multiple of block_size
        remainder = total_length % block_size
        if remainder != 0:
            self.input_ids = self.input_ids[:-remainder]

        self.num_chunks = self.input_ids.size(0) // block_size

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        # Each item is a chunk of length block_size
        start = idx * self.block_size
        end = start + self.block_size
        x = self.input_ids[start:end]
        # The target is the same sequence shifted by 1
        y = x.roll(-1)
        return x, y

if __name__ == "__main__":
    max_seq_len = 256
    raw_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_texts = raw_dataset["train"]["text"]
    val_texts   = raw_dataset["validation"]["text"]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token 

    train_dataset = LMTextDataset(train_texts, tokenizer, block_size=max_seq_len)
    val_dataset   = LMTextDataset(val_texts,   tokenizer, block_size=max_seq_len)
    torch.save(train_dataset, "data/train_dataset.pt")
    torch.save(val_dataset, "data/val_dataset.pt")
    print(f"Number of samples in train_dataset: {len(train_dataset)}")
