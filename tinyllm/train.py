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


def train(model, train_data, batch_size, seq_len, num_iters, optimizer, vocab_size, test_data):
    t0 = time()
    for it in range(num_iters):
        x_batch, y_batch = data_loader(train_data, batch_size, seq_len)
        with torch.autocast(device_type="cuda"):
            logits = model(x_batch)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y_batch.view(-1).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % 100 == 0:
            with torch.no_grad():
                x_eval, y_eval = data_loader(test_data, batch_size=batch_size, seq_len=seq_len)
                test_loss = torch.nn.functional.cross_entropy(model(x_eval).view(-1, vocab_size), y_eval.view(-1).long())
                print(f"Iter: {it}, Train loss: {loss.item()},Eval Loss: {test_loss.item()}, Time: {time() - t0:.2f}s")
            t0 = time()

    model.save('checkpoints/tinyllm_shakespeare.pth') 

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = np.load('datasets/shakespeare_train.npy')
    test_data = np.load('datasets/shakespeare_test.npy')
    vocab_size = np.load('datasets/shakespeare_vocab_size.npy')[0]
    batch_size = 128
    config = Config(vocab_size=vocab_size, embed_dim=256, num_heads=8, head_size=32, block_size=256, num_layers=6)
    x, y = data_loader(train_data, batch_size=batch_size, seq_len=config.block_size) 
    model = DecoderTransformer(config).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters())
    train(model, train_data, batch_size=batch_size, seq_len=config.block_size, num_iters=5000, optimizer=optimizer, vocab_size=vocab_size, test_data=test_data)




