"""
Evaluate a trained model on HellaSwag.

Usage:
    python -m scripts.eval_hellaswag --checkpoint checkpoints/tinyllm_climbing.pth
    python -m scripts.eval_hellaswag --checkpoint checkpoints/tinyllm_shakespeare.pth --tokenizer shakespeare
"""

import argparse
import numpy as np
import torch
import tiktoken
from tinyllm.models import Config, DecoderTransformer
from tinyllm.hellaswag import evaluate_hellaswag


def infer_config(state_dict, block_size):
    vocab_size, embed_dim = state_dict["token_embedding.weight"].shape
    # block_size cannot be inferred from the checkpoint anymore: RoPE replaced
    # the learned positional embedding table and its cos/sin buffers are not
    # persisted. It is supplied explicitly instead.
    num_layers = sum(1 for k in state_dict if k.startswith("layers.") and k.endswith(".attn.kqv.weight"))
    bias = any("bias" in k for k in state_dict)
    head_size = 64
    num_heads = embed_dim // head_size
    return Config(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        num_layers=num_layers,
    )


class ShakespeareTokenizer:
    """Wraps the char-level encoder dict to expose encode_ordinary(), matching the tiktoken API used in hellaswag.py."""
    def __init__(self):
        self.encoder = np.load("datasets/shakespeare_ch_to_ix.npy", allow_pickle=True).item()

    def encode_ordinary(self, text):
        return [self.encoder[c] for c in text if c in self.encoder]


parser = argparse.ArgumentParser(description="Evaluate a model on HellaSwag")
parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint (.pth)")
parser.add_argument("--tokenizer", type=str, default="shakespeare", choices=["gpt2", "shakespeare"])
parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
parser.add_argument("--max-examples", type=int, default=None, help="limit number of examples (default: full split)")
parser.add_argument("--head-size", type=int, default=None, help="override head size (inferred as 64 by default)")
parser.add_argument("--block-size", type=int, default=512, help="context length the model was trained with")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = ShakespeareTokenizer() if args.tokenizer == "shakespeare" else tiktoken.get_encoding("gpt2")

ckpt = torch.load(args.checkpoint, map_location=device)
# New checkpoints wrap the weights in a dict (model/optimizer/scheduler/iter);
# legacy ones are a bare state_dict.
state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
config = infer_config(state_dict, args.block_size)
if args.head_size is not None:
    config.head_size = args.head_size
    config.num_heads = config.embed_dim // args.head_size
print(f"Inferred config: {config}")

model = DecoderTransformer(config).to(device)
model.load_state_dict(state_dict)
model.eval()

print(f"Loaded checkpoint: {args.checkpoint}")
print(f"Evaluating on HellaSwag {args.split}...")
accuracy = evaluate_hellaswag(model, tokenizer, device, split=args.split, max_examples=args.max_examples)
print(f"HellaSwag accuracy: {100 * accuracy:.2f}%")
