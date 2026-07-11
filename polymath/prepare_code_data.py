import os
import json
import argparse
import numpy as np
from datasets import load_dataset, interleave_datasets
from tokenizer import get_tokenizer

def process_code_dataset(output_dir: str, max_samples: int, tokenizer_type: str = "tiktoken", encoding_name: str = "cl100k_base", val_ratio: float = 0.05, hf_token: str = None):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initializing {tokenizer_type} tokenizer...")
    tokenizer = get_tokenizer(tokenizer_type, encoding_name=encoding_name, text=None)
    
    if tokenizer.vocab_size < 65536:
        dtype = np.uint16
        dtype_name = "uint16"
    else:
        dtype = np.uint32
        dtype_name = "uint32"

    print(f"Streaming flytech/python-codes-25k...")
    dataset = load_dataset('flytech/python-codes-25k', split='train', streaming=True)
    
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    
    train_f = open(train_bin_path, "wb")
    val_f = open(val_bin_path, "wb")
    
    total_train = 0
    total_val = 0
    sample_count = 0
    
    print(f"Processing up to {max_samples} samples...")
    for row in dataset:
        if sample_count >= max_samples:
            break
            
        instruction = row.get("instruction", "").strip()
        output = row.get("output", "").strip()
        if not instruction or not output:
            continue
            
        text = f"Instruction: {instruction}\nOutput: {output}<|endoftext|>"
        tokens = tokenizer.encode(text)
        
        # Decide if val or train
        is_val = (sample_count % int(1/val_ratio)) == 0 if val_ratio > 0 else False
        
        arr = np.array(tokens, dtype=dtype)
        if is_val:
            val_f.write(arr.tobytes())
            total_val += len(tokens)
        else:
            train_f.write(arr.tobytes())
            total_train += len(tokens)
            
        sample_count += 1
        if sample_count % 5000 == 0:
            print(f"Processed {sample_count} files... (Train tokens: {total_train:,}, Val tokens: {total_val:,})")
            
    train_f.close()
    val_f.close()
    
    meta = {
        "tokenizer_type": tokenizer_type,
        "encoding_name": encoding_name,
        "vocab_size": tokenizer.vocab_size,
        "dtype": dtype_name,
        "train_tokens": total_train,
        "val_tokens": total_val,
        "samples_processed": sample_count
    }
    
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nData preparation complete!")
    print(f"Train tokens: {total_train:,} | Val tokens: {total_val:,}")
    print(f"Metadata saved to {meta_path}")

def main():
    parser = argparse.ArgumentParser(description="Stream code datasets into binary files.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to store train.bin and val.bin.")
    parser.add_argument("--tokenizer", type=str, default="tiktoken", choices=["tiktoken", "custom_bpe"], help="Tokenizer type.")
    parser.add_argument("--encoding_name", type=str, default="cl100k_base", help="Tiktoken encoding name, or path to custom_bpe json.")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation data fraction.")
    parser.add_argument("--max_samples", type=int, default=50000, help="Max samples to stream.")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Token for gated access.")
    args = parser.parse_args()

    process_code_dataset(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        tokenizer_type=args.tokenizer,
        encoding_name=args.encoding_name,
        val_ratio=args.val_ratio,
        hf_token=args.token
    )

if __name__ == "__main__":
    main()
