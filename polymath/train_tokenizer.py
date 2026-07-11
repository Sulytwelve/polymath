# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "tokenizers",
# ]
# ///
import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def get_training_corpus():
    # Load dataset streaming
    dataset = load_dataset('flytech/python-codes-25k', split='train', streaming=True)
    for row in dataset:
        instruction = row.get("instruction", "").strip()
        output = row.get("output", "").strip()
        if not instruction or not output:
            continue
        text = f"Instruction: {instruction}\nOutput: {output}<|endoftext|>"
        yield text

def main():
    print("Initializing BPE Tokenizer...")
    # Initialize a tokenizer with Byte-Pair Encoding
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    
    # Use byte-level pre-tokenization (same as GPT-2/GPT-4)
    tokenizer.pre_tokenizer = ByteLevel()
    
    # Add byte-level decoder so tokens map back to spaces/newlines correctly
    tokenizer.decoder = ByteLevelDecoder()
    
    # Setup the trainer to strictly output 4096 tokens
    trainer = BpeTrainer(
        vocab_size=4096,
        special_tokens=["<|endoftext|>", "<|unk|>"]
    )
    
    print("Training tokenizer on flytech/python-codes-25k... this may take a minute...")
    # Train the tokenizer from the python iterator
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    # Ensure config directory exists
    os.makedirs("configs", exist_ok=True)
    
    # Save the tokenizer
    tokenizer.save("configs/tokenizer.json")
    print("✅ Successfully trained and saved custom tokenizer to configs/tokenizer.json")
    print(f"Final Vocab Size: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    main()
