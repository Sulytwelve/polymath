import os
import json
import argparse
import numpy as np
from datasets import load_dataset
from tokenizer import get_tokenizer

def process_hf_dataset(output_dir: str, max_samples: int, tokenizer_type: str = "tiktoken", encoding_name: str = "cl100k_base", val_ratio: float = 0.1):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initializing {tokenizer_type} tokenizer...")
    # Initialize tokenizer without full text since we stream
    tokenizer = get_tokenizer(tokenizer_type, encoding_name=encoding_name, text=None)
    
    if tokenizer.vocab_size < 65536:
        dtype = np.uint16
        dtype_name = "uint16"
    else:
        dtype = np.uint32
        dtype_name = "uint32"

    print("Streaming dataset nvidia/OpenMathInstruct-1...")
    dataset = load_dataset('nvidia/OpenMathInstruct-1', split='train', streaming=True)
    
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    
    # Pre-allocate large memmaps, we will resize them at the end.
    # To avoid OOM, we process in chunks and append.
    # numpy.memmap doesn't support easy appending, so we will use a raw file and np.memmap it for writing chunks.
    
    # We will write raw binary data iteratively
    train_f = open(train_bin_path, "wb")
    val_f = open(val_bin_path, "wb")
    
    total_train = 0
    total_val = 0
    sample_count = 0
    
    print(f"Processing up to {max_samples} samples...")
    for row in dataset:
        if sample_count >= max_samples:
            break
            
        problem = row.get("problem", "")
        solution = row.get("generated_solution", "")
        
        text = f"Problem:\n{problem}\nSolution:\n{solution}<|endoftext|>"
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
        if sample_count % 10000 == 0:
            print(f"Processed {sample_count} samples... (Train tokens: {total_train:,}, Val tokens: {total_val:,})")
            
    train_f.close()
    val_f.close()
    
    # Save metadata
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
    parser = argparse.ArgumentParser(description="Stream HuggingFace datasets into binary mmap files.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to store train.bin and val.bin.")
    parser.add_argument("--tokenizer", type=str, default="tiktoken", choices=["tiktoken"], help="Tokenizer type.")
    parser.add_argument("--encoding_name", type=str, default="cl100k_base", help="Tiktoken encoding name.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data fraction.")
    parser.add_argument("--max_samples", type=int, default=100000, help="Max samples to stream.")
    args = parser.parse_args()

    process_hf_dataset(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        tokenizer_type=args.tokenizer,
        encoding_name=args.encoding_name,
        val_ratio=args.val_ratio
    )

if __name__ == "__main__":
    main()
