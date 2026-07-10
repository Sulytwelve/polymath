import os
import argparse
import torch
from model import TransformerLM
from tokenizer import get_tokenizer
from train import DEFAULT_CORPUS

def main():
    parser = argparse.ArgumentParser(description="Generate text using fast KV-Cache from a trained Transformer checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_standard.pt", help="Path to model checkpoint.")
    parser.add_argument("--prompt", type=str, default="First Citizen:\n", help="Starting text prompt.")
    parser.add_argument("--num_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering limit.")
    parser.add_argument("--no_cache", action="store_true", help="Disable KV-Cache decoding (fallback to O(T^2) recomputation).")
    args = parser.parse_args()

    # Determine device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file {args.checkpoint} not found. Please train a model first using train.py.")
        return

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    vocab_size = checkpoint["vocab_size"]

    print(f"Loaded checkpoint step {checkpoint.get('step', 'unknown')}")
    print(f"Model trained in mode '{config.model.mode}' with d_model={config.model.d_model}, n_layers={config.model.n_layers}, n_heads={config.model.n_heads}, n_kv_heads={config.model.n_kv_heads}")

    # Initialize Model
    model = TransformerLM(config.model, vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Initialize Tokenizer
    tokenizer = get_tokenizer(
        config.tokenizer.tokenizer_type,
        config.tokenizer.tiktoken_encoding,
        text=DEFAULT_CORPUS if config.tokenizer.tokenizer_type == "char" else None
    )

    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    if not prompt_tokens:
        prompt_tokens = [0]

    context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    print(f"Prompt: {repr(args.prompt)}")
    print("--- Generated Text ---")

    # Generate using KV-Cache
    use_cache = not args.no_cache
    generated_ids = model.generate(
        context,
        max_new_tokens=args.num_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        use_cache=use_cache
    )[0].tolist()

    print(tokenizer.decode(generated_ids))
    print("----------------------")

if __name__ == "__main__":
    main()
