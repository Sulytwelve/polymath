import os
import json
import argparse
import numpy as np
from tokenizer import get_tokenizer

def prepare_data(input_file: str, output_dir: str, tokenizer_type: str = "tiktoken", encoding_name: str = "cl100k_base", val_ratio: float = 0.1):
    assert os.path.exists(input_file), f"Input file not found: {input_file}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading text from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Initializing {tokenizer_type} tokenizer...")
    tokenizer = get_tokenizer(tokenizer_type, encoding_name=encoding_name, text=text if tokenizer_type == "char" else None)
    
    print("Encoding text into tokens...")
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    print(f"Total tokens encoded: {total_tokens:,}")

    # Determine optimal numpy storage dtype
    if tokenizer.vocab_size < 65536:
        dtype = np.uint16
        dtype_name = "uint16"
    else:
        dtype = np.uint32
        dtype_name = "uint32"

    # Split train vs val
    val_tokens_count = int(total_tokens * val_ratio)
    train_tokens_count = total_tokens - val_tokens_count

    train_tokens = tokens[:train_tokens_count]
    val_tokens = tokens[train_tokens_count:]

    # Write train.bin
    train_bin_path = os.path.join(output_dir, "train.bin")
    print(f"Writing {len(train_tokens):,} tokens to {train_bin_path} ({dtype_name})...")
    train_arr = np.memmap(train_bin_path, dtype=dtype, mode="w+", shape=(len(train_tokens),))
    train_arr[:] = train_tokens
    train_arr.flush()

    # Write val.bin
    val_bin_path = os.path.join(output_dir, "val.bin")
    print(f"Writing {len(val_tokens):,} tokens to {val_bin_path} ({dtype_name})...")
    val_arr = np.memmap(val_bin_path, dtype=dtype, mode="w+", shape=(len(val_tokens),))
    val_arr[:] = val_tokens
    val_arr.flush()

    # Save metadata
    meta = {
        "tokenizer_type": tokenizer_type,
        "encoding_name": encoding_name,
        "vocab_size": tokenizer.vocab_size,
        "dtype": dtype_name,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens)
    }
    if tokenizer_type == "char" and hasattr(tokenizer, "chars"):
        meta["chars"] = tokenizer.chars

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Data preparation complete! Metadata saved to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and tokenize text corpus into binary mmap files.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to store train.bin and val.bin.")
    parser.add_argument("--tokenizer", type=str, default="tiktoken", choices=["char", "tiktoken"], help="Tokenizer type.")
    parser.add_argument("--encoding_name", type=str, default="cl100k_base", help="Tiktoken encoding name.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data fraction.")
    args = parser.parse_args()

    prepare_data(
        input_file=args.input_file,
        output_dir=args.output_dir,
        tokenizer_type=args.tokenizer,
        encoding_name=args.encoding_name,
        val_ratio=args.val_ratio
    )

if __name__ == "__main__":
    main()
