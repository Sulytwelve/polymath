import os
import json
from typing import Optional
import torch
from torch.utils.data import Dataset
import numpy as np

class CharDataset(Dataset):
    """
    A simple character-level dataset for language modeling (in-memory).
    Reads a string of text, creates a character-to-integer mapping (vocabulary),
    and supports retrieving random batches of context-target windows for training.
    """
    def __init__(self, text: str, block_size: int):
        self.text = text
        self.block_size = block_size
        
        # Build vocabulary from unique characters
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Character <=> index mapping
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        # Convert full text to tensor token IDs
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def encode(self, s: str) -> list:
        """Encode a string of characters into a list of token IDs."""
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l: list) -> str:
        """Decode a list of token IDs back into a string of characters."""
        return "".join([self.itos[i] for i in l])

    def __len__(self) -> int:
        return max(1, len(self.data) - self.block_size)

    def __getitem__(self, idx: int):
        if idx >= len(self.data) - self.block_size:
            idx = torch.randint(len(self.data) - self.block_size, (1,)).item()
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

    def get_batch(self, batch_size: int, device: str = "cpu"):
        """
        Retrieves a random batch of inputs X and next-token targets Y.
        """
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
        if device != "cpu":
            x = x.to(device)
            y = y.to(device)
        return x, y


class MmapDataset(Dataset):
    """
    Memory-mapped dataset backed by preprocessed binary files (.bin).
    Allows fast random slicing of GB-scale text corpora with near-zero RAM overhead.
    """
    def __init__(self, bin_path: str, block_size: int, meta_path: Optional[str] = None):
        assert os.path.exists(bin_path), f"Binary dataset file not found: {bin_path}"
        self.bin_path = bin_path
        self.block_size = block_size
        
        # Determine dtype and metadata
        dtype = np.uint16
        self.vocab_size = 100277 # Default tiktoken cl100k_base
        if meta_path is None:
            # Try finding meta.json in the same directory
            maybe_meta = os.path.join(os.path.dirname(bin_path), "meta.json")
            if os.path.exists(maybe_meta):
                meta_path = maybe_meta

        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            dtype_str = meta.get("dtype", "uint16")
            dtype = np.uint32 if dtype_str == "uint32" else np.uint16
            self.vocab_size = meta.get("vocab_size", self.vocab_size)

        self.dtype = dtype
        # Load memory map
        self.data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        assert len(self.data) > self.block_size, f"Dataset length ({len(self.data)}) must be > block_size ({self.block_size})"

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def __len__(self) -> int:
        return max(1, len(self.data) - self.block_size)

    def __getitem__(self, idx: int):
        if idx >= len(self.data) - self.block_size:
            idx = torch.randint(len(self.data) - self.block_size, (1,)).item()
        # Slice from memmap and convert to torch int64
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + self.block_size + 1].astype(np.int64))
        return x, y

    def get_batch(self, batch_size: int, device: str = "cpu"):
        """
        Retrieves a random batch of inputs X and next-token targets Y using memmap slicing.
        """
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + self.block_size + 1].astype(np.int64)) for i in ix])
        if device != "cpu":
            x = x.to(device)
            y = y.to(device)
        return x, y


def get_dataset(dataset_type: str, block_size: int, split: str = "train", data_dir: Optional[str] = None, text: Optional[str] = None) -> Dataset:
    dataset_type = dataset_type.lower()
    if dataset_type == "char_memory":
        assert text is not None, "For char_memory dataset_type, `text` must be provided."
        return CharDataset(text=text, block_size=block_size)
    elif dataset_type == "mmap":
        assert data_dir is not None and os.path.exists(data_dir), f"For mmap dataset_type, `data_dir` directory must exist: {data_dir}"
        bin_path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(bin_path) and split == "val":
            # Fallback to train.bin if val.bin does not exist
            bin_path = os.path.join(data_dir, "train.bin")
        return MmapDataset(bin_path=bin_path, block_size=block_size)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
