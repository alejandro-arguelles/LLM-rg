import os
import numpy as np
import torch
import pyarrow.parquet as pq


def data_loader(data, batch_size, seq_len, dataset="shakespeare", tokenizer=None):
    """
    Returns a generator that yields (x, y) batches of shape (batch_size, seq_len) indefinitely.
    For 'shakespeare': data is a pre-loaded np.ndarray.
    For 'climbing': data is a list of parquet file paths; requires tokenizer (tiktoken Encoding).
    """
    match dataset:
        case "shakespeare":
            return shakespeare_data_loader(data, batch_size, seq_len)
        case "climbing":
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided for climbing dataset")
            return climbing_data_loader(data, batch_size, seq_len, tokenizer=tokenizer)
        case _:
            raise ValueError(f"Unsupported dataset: {dataset}")


def shakespeare_data_loader(data, batch_size, seq_len):
    """Generator that yields random batches from a pre-loaded numpy array indefinitely."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = len(data)
    while True:
        start_indices = np.random.choice(n - seq_len - 1, size=batch_size, replace=False)
        x_batch = np.array([data[i:i+seq_len] for i in start_indices])
        y_batch = np.array([data[i+1:i+seq_len+1] for i in start_indices])
        yield (
            torch.tensor(x_batch, dtype=torch.long).to(device),
            torch.tensor(y_batch, dtype=torch.long).to(device),
        )


def list_parquet_files(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".parquet"))


def climbing_data_loader(parquet_files, batch_size, seq_len, tokenizer, tokenizer_batch_size=128):
    """Generator that streams batches from a list of parquet files indefinitely."""
    assert parquet_files, "parquet_files list is empty"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    buffer = []
    needed = batch_size * seq_len + 1

    while True:
        for filepath in parquet_files:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                texts = pf.read_row_group(rg_idx, columns=["text"]).column("text").to_pylist()
                for i in range(0, len(texts), tokenizer_batch_size):
                    for tokens in tokenizer.encode_ordinary_batch(texts[i:i + tokenizer_batch_size]):
                        buffer.extend(tokens)

                while len(buffer) >= needed:
                    chunk = np.array(buffer[:needed], dtype=np.int64)
                    buffer = buffer[needed:]
                    x = chunk[:-1].reshape(batch_size, seq_len)
                    y = chunk[1:].reshape(batch_size, seq_len)
                    yield (
                        torch.tensor(x, dtype=torch.long).to(device),
                        torch.tensor(y, dtype=torch.long).to(device),
                    )
