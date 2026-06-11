import os
import sys
from time import time

import numpy as np
import torch
from tinyllm.models import Config, DecoderTransformer

def data_loader(data, batch_size, seq_len):
    n = len(data)
    # Randomly sample starting indices for each sequence in the batch
    start_indices = np.random.randint(0, n - seq_len - 1, size=batch_size)
    x_batch = np.array([data[i:i+seq_len] for i in start_indices])
    y_batch = np.array([data[i+1:i+seq_len+1] for i in start_indices])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_batch = torch.tensor(x_batch, dtype=torch.long).to(device)
    y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)
    return x_batch, y_batch


def train(model, train_data, batch_size, seq_len, num_iters, optimizer, scheduler, vocab_size, test_data, checkpoint_path=None):
    scaler = torch.cuda.amp.GradScaler()
    t0 = time()
    for it in range(num_iters):
        x_batch, y_batch = data_loader(train_data, batch_size, seq_len)
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
                x_eval, y_eval = data_loader(test_data, batch_size=batch_size, seq_len=seq_len)
                model.eval()
                test_loss = torch.nn.functional.cross_entropy(model(x_eval).view(-1, vocab_size), y_eval.view(-1).long())
                print(f"Iter: {it}, Train loss: {loss.item()}, Eval Loss: {test_loss.item()}, "
                      f"Time: {time() - t0:.2f}, Gradient Norm: {total_norm:.2f}")
                model.train()
            t0 = time()
    model.save(check_point_path) 

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_point_path = 'checkpoints/tinyllm_shakespeare.pth'
    train_data = np.load('datasets/shakespeare_train.npy')
    test_data = np.load('datasets/shakespeare_test.npy')
    encoder = np.load('datasets/shakespeare_ch_to_ix.npy', allow_pickle=True).item()
    decoder = np.load('datasets/shakespeare_ix_to_ch.npy', allow_pickle=True).item()
    vocab_size = np.load('datasets/shakespeare_vocab_size.npy')[0]
    batch_size = 1024
    num_iters = 10000
    config = Config(vocab_size=vocab_size, embed_dim=256, num_heads=8, head_size=32, block_size=512, num_layers=6)
    x, y = data_loader(train_data, batch_size=batch_size, seq_len=config.block_size) 
    model = DecoderTransformer(config).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)
    model.train()
    train(model, train_data, batch_size=batch_size, seq_len=config.block_size, num_iters=num_iters, optimizer=optimizer, scheduler=scheduler, vocab_size=vocab_size, test_data=test_data)
    model = model.load(check_point_path)
    model.eval()
    print(model.generate("To be, or not to be, that is the question:", encoder, decoder, max_new_tokens=100))





