import os
import tiktoken
import sys
from time import time

import numpy as np
import torch
from tinyllm.models import Config, DecoderTransformer
from tinyllm.load_data import data_loader, list_parquet_files


def train(model, train_loader, batch_size, seq_len, num_iters, optimizer, scheduler, vocab_size, eval_loader, checkpoint_path, encoder=None, decoder=None):
    scaler = torch.cuda.amp.GradScaler()
    t0 = time()
    for it in range(num_iters):
        x_batch, y_batch = next(train_loader)
        with torch.autocast(device_type="cuda"):
            logits = model(x_batch)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y_batch.view(-1).long())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if it % 100 == 0:
            with torch.no_grad():
                x_eval, y_eval = next(eval_loader)
                model.eval()
                test_loss = torch.nn.functional.cross_entropy(model(x_eval).view(-1, vocab_size), y_eval.view(-1).long())
                print(f"Iter: {it}, Train loss: {loss.item()}, Eval Loss: {test_loss.item()}, "
                      f"Time: {time() - t0:.2f}, Gradient Norm: {total_norm:.2f}")
                model.save(checkpoint_path)
                prompt = "To be, or not to be, that is the question:"
                input_ids = torch.tensor([encoder(prompt)], dtype=torch.long).to("cuda")
                output_ids = model.generate(input_ids, max_new_tokens=100)
                print(decoder(output_ids))
                model.train()
            t0 = time()
    model.save(checkpoint_path) 

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
    model = model.load(check_point_path)
    model.eval()
    prompt = "To be or not to be, that is the question:"
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
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)
    model.train()
    train(model, train_loader, batch_size=batch_size, seq_len=config.block_size, num_iters=num_iters, optimizer=optimizer, scheduler=scheduler, vocab_size=vocab_size, eval_loader=eval_loader, checkpoint_path=check_point_path, encoder=encoder, decoder=decoder)
    model = model.load(check_point_path)
    model.eval()
    prompt = "To be or not to be, that is the question:"
    input_ids = torch.tensor([encoder(prompt)], dtype=torch.long).to(device)
    output_ids = model.generate(input_ids, max_new_tokens=100)
    print(decoder(output_ids))


if __name__ == "__main__":
    train_climbing()