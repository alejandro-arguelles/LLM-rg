import os
import math
import tiktoken
import sys
from time import time

import numpy as np
import torch
from tinyllm.models import Config, DecoderTransformer
from tinyllm.load_data import data_loader, list_parquet_files


def get_warmup_cosine_scheduler(optimizer, warmup_iters, num_iters, min_lr_ratio=0.1):
    """Linear warmup followed by cosine decay, as a multiplier on the base lr.

    - For it < warmup_iters the factor ramps linearly 0 -> 1.
    - Afterwards it follows half a cosine from 1 down to min_lr_ratio over the
      remaining iterations.
    """
    def lr_lambda(it):
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = (it - warmup_iters) / max(1, num_iters - warmup_iters)
        progress = min(progress, 1.0)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path, model, optimizer, scheduler, it):
    """Save a full training checkpoint (weights + optimizer + scheduler + iter).

    Written atomically (tmp file + rename) so a crash mid-write can't leave a
    truncated checkpoint behind.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iter": it,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None):
    """Load a checkpoint. Restores optimizer/scheduler too if given.

    Returns the iteration to resume from (0 for legacy weights-only files).
    """
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        return ckpt.get("iter", 0)
    # Legacy format: a bare state_dict saved by model.save().
    model.load_state_dict(ckpt)
    return 0


def train(model, train_loader, batch_size, seq_len, num_iters, optimizer, scheduler, vocab_size, eval_loader, checkpoint_path, encoder=None, decoder=None, start_iter=0, checkpoint_interval=1000):
    t0 = time()
    for it in range(start_iter, num_iters):
        x_batch, y_batch = next(train_loader)
        # bf16 autocast: same exponent range as fp32, so activations don't
        # overflow to inf/nan and no GradScaler is needed.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x_batch)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y_batch.view(-1).long())
        optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Skip the step on a non-finite gradient so one bad batch can't
        # permanently corrupt the weights (GradScaler used to do this for us).
        if torch.isfinite(total_norm):
            optimizer.step()
        scheduler.step()
        if it % 100 == 0:
            with torch.no_grad():
                x_eval, y_eval = next(eval_loader)
                model.eval()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    test_loss = torch.nn.functional.cross_entropy(model(x_eval).view(-1, vocab_size), y_eval.view(-1).long())
                print(f"Iter: {it}, Train loss: {loss.item()}, Eval Loss: {test_loss.item()}, "
                      f"Time: {time() - t0:.2f}, Gradient Norm: {total_norm:.2f}", flush=True)
                prompt = "What is the purpose of life?"
                input_ids = torch.tensor([encoder(prompt)], dtype=torch.long).to("cuda")
                output_ids = model.generate(input_ids, max_new_tokens=100)
                print(decoder(output_ids), flush=True)
                model.train()
            t0 = time()
        if it % checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, it)
    save_checkpoint(checkpoint_path, model, optimizer, scheduler, num_iters)

def train_shakespeare():
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_point_path = 'checkpoints/tinyllm_shakespeare.pth'
    train_data = np.load('datasets/shakespeare_train.npy')
    test_data = np.load('datasets/shakespeare_test.npy')
    ch_to_ix = np.load('datasets/shakespeare_ch_to_ix.npy', allow_pickle=True).item()
    ix_to_ch = np.load('datasets/shakespeare_ix_to_ch.npy', allow_pickle=True).item()
    vocab_size = np.load('datasets/shakespeare_vocab_size.npy')[0]
    encoder = lambda s: [ch_to_ix[c] for c in s]
    decoder = lambda ids: ''.join([ix_to_ch[i] for i in ids.tolist()])
    batch_size = 512
    num_iters = 500
    config = Config(vocab_size=vocab_size, embed_dim=256, num_heads=8, head_size=32, block_size=512, num_layers=6)
    train_loader = data_loader(train_data, batch_size=batch_size, seq_len=config.block_size)
    eval_loader = data_loader(test_data, batch_size=batch_size, seq_len=config.block_size)
    x, y = next(train_loader)
    model = DecoderTransformer(config).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)
    model.train()
    train(model, train_loader, batch_size=batch_size, seq_len=config.block_size, num_iters=num_iters, optimizer=optimizer, scheduler=scheduler, vocab_size=vocab_size, eval_loader=eval_loader, checkpoint_path=check_point_path, encoder=encoder, decoder=decoder)
    load_checkpoint(check_point_path, model, map_location=device)
    model.eval()
    prompt = "What is the most fundamental thing about the universe?"

    input_ids = torch.tensor([encoder(prompt)], dtype=torch.long).to("cuda")
    output_ids = model.generate(input_ids, max_new_tokens=100)
    print(decoder(output_ids))

def train_climbing():
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_point_path = 'checkpoints/tinyllm_climbing.pth'
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    encoder = tokenizer.encode_ordinary
    decoder = lambda ids: tokenizer.decode(ids.tolist())
    print(vocab_size)
    batch_size = 64
    num_iters = 600000
    config = Config(vocab_size=vocab_size, embed_dim=768, num_heads=12, head_size=64, block_size=512, num_layers=12)
    all_files = list_parquet_files('/home/criteo/.cache/nanochat/base_data_climbmix')
    split = int(0.9 * len(all_files))
    train_loader = data_loader(all_files[:split], batch_size=batch_size, seq_len=config.block_size, dataset='climbing', tokenizer=tokenizer)
    eval_loader = data_loader(all_files[split:], batch_size=batch_size, seq_len=config.block_size, dataset='climbing', tokenizer=tokenizer)
    model = DecoderTransformer(config).to(device)
    # NOTE: torch.compile disabled -- with autocast bf16 it destabilises training
    # here (logits grow unbounded -> nan). The eager path trains cleanly.
    # model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    warmup_iters = min(2000, num_iters // 10)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_iters=warmup_iters, num_iters=num_iters)
    model.train()
    train(model, train_loader, batch_size=batch_size, seq_len=config.block_size, num_iters=num_iters, optimizer=optimizer, scheduler=scheduler, vocab_size=vocab_size, eval_loader=eval_loader, checkpoint_path=check_point_path, encoder=encoder, decoder=decoder)
    load_checkpoint(check_point_path, model, map_location=device)
    model.eval()
    prompt = "What is the most fundamental thing about the universe?"
    input_ids = torch.tensor([encoder(prompt)], dtype=torch.long).to(device)
    output_ids = model.generate(input_ids, max_new_tokens=100)
    print(decoder(output_ids))


if __name__ == "__main__":
    train_climbing()