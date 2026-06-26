"""Wrapper to train the climbing model with env-var overrides.

Reads:
  NUM_ITERS  number of training iterations   (default 600000)
  BATCH      batch size                      (default 64)
  DATA_DIR   parquet dataset directory       (default /root/climbmix)
  RESUME     if set (1/true), resume from the existing checkpoint

Usage:
  DATA_DIR=/root/climbmix BATCH=8 NUM_ITERS=500 PYTHONPATH=. uv run python scripts/run_climbing.py
  RESUME=1 DATA_DIR=/root/climbmix BATCH=64 NUM_ITERS=600000 PYTHONPATH=. uv run python scripts/run_climbing.py
"""
import os

import tiktoken
import torch

from tinyllm.models import Config, DecoderTransformer
from tinyllm.load_data import data_loader, list_parquet_files
from tinyllm.train import train, get_warmup_cosine_scheduler, load_checkpoint

num_iters = int(os.environ.get("NUM_ITERS", 600))
batch_size = int(os.environ.get("BATCH", 64))
data_dir = os.environ.get("DATA_DIR", "/root/climbmix")
resume = os.environ.get("RESUME", "0").lower() not in ("0", "", "false", "no")

torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"
check_point_path = "checkpoints/tinyllm_climbing.pth"

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab
encoder = tokenizer.encode_ordinary
decoder = lambda ids: tokenizer.decode(ids.tolist())

print(f"vocab_size={vocab_size} num_iters={num_iters} batch_size={batch_size} data_dir={data_dir}")

config = Config(vocab_size=vocab_size, embed_dim=768, num_heads=12, head_size=64, block_size=512, num_layers=12)
all_files = list_parquet_files(data_dir)
split = int(0.9 * len(all_files))
train_loader = data_loader(all_files[:split], batch_size=batch_size, seq_len=config.block_size, dataset="climbing", tokenizer=tokenizer)
eval_loader = data_loader(all_files[split:], batch_size=batch_size, seq_len=config.block_size, dataset="climbing", tokenizer=tokenizer)

model = DecoderTransformer(config).to(device)
# NOTE: torch.compile is disabled on purpose. With autocast bf16 it destabilises
# training here (logits grow unbounded, loss stops descending and eventually goes
# to nan) -- the eager path trains cleanly. Re-enable only after verifying numerics.
# model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
warmup_iters = min(2000, num_iters // 10)
scheduler = get_warmup_cosine_scheduler(optimizer, warmup_iters=warmup_iters, num_iters=num_iters)

start_iter = 0
if resume:
    if os.path.exists(check_point_path):
        start_iter = load_checkpoint(check_point_path, model, optimizer, scheduler, map_location=device)
        print(f"Resuming from checkpoint {check_point_path} at iter {start_iter}")
    else:
        print(f"RESUME set but no checkpoint at {check_point_path}; starting from scratch")

model.train()

train(model, train_loader, batch_size=batch_size, seq_len=config.block_size, num_iters=num_iters,
      optimizer=optimizer, scheduler=scheduler, vocab_size=vocab_size, eval_loader=eval_loader,
      checkpoint_path=check_point_path, encoder=encoder, decoder=decoder, start_iter=start_iter)

load_checkpoint(check_point_path, model, map_location=device)
model.eval()
prompt = "To be or not to be, that is the question:"
input_ids = torch.tensor([encoder(prompt)], dtype=torch.long).to(device)
output_ids = model.generate(input_ids, max_new_tokens=100)
print(decoder(output_ids))
