import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    def bos_token_id(self) -> Optional[int]:
        return None

    @property
    def eos_token_id(self) -> Optional[int]:
        return None

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.eos_token_id

    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> "BaseTokenizer":
        raise NotImplementedError


class CharTokenizer(BaseTokenizer):
    """
    Simple character-level tokenizer for lightweight debugging and compatibility.
    """
    def __init__(self, chars: Optional[List[str]] = None, text: Optional[str] = None):
        if chars is not None:
            self.chars = sorted(list(set(chars)))
        elif text is not None:
            self.chars = sorted(list(set(text)))
        else:
            # Default minimal ASCII / common char set fallback
            self.chars = sorted(list(" \t\n\r!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"))
        
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.itos.get(i, "") for i in tokens])

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(chars=data["chars"])


class TiktokenTokenizer(BaseTokenizer):
    """
    BPE tokenizer backed by OpenAI's tiktoken library (e.g., cl100k_base or o200k_base).
    Recommended for training >= 100M parameter models.
    """
    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError:
            raise ImportError("tiktoken is not installed. Please run `uv pip install tiktoken` or select `char` tokenizer.")
        
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special="all")

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    @property
    def eos_token_id(self) -> Optional[int]:
        return getattr(self.enc, "eot_token", None)

    @property
    def bos_token_id(self) -> Optional[int]:
        return getattr(self.enc, "eot_token", None)


def get_tokenizer(tokenizer_type: str, encoding_name: str = "cl100k_base", text: Optional[str] = None) -> BaseTokenizer:
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == "char":
        return CharTokenizer(text=text)
    elif tokenizer_type == "tiktoken":
        return TiktokenTokenizer(encoding_name=encoding_name)
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")
