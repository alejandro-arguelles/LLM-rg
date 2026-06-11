import requests
import numpy as np

def fetch_shakespeare_data(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

    with open(input_file_path, 'r') as f:
        data = f.read()
    return data

def tokenize(data):
    n = len(data)
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    ch_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_ch = { i:ch for i,ch in enumerate(chars) }
    ix_data = [ch_to_ix[c] for c in data]
    train_data = ix_data[:int(0.9 * n)]
    test_data = ix_data[int(0.9 * n):]
    return train_data, test_data, ch_to_ix, ix_to_ch, vocab_size

if __name__ == "__main__":
    input_file_path = 'shakespeare_input.txt'
    data = fetch_shakespeare_data(input_file_path)
    train_data, test_data, ch_to_ix, ix_to_ch, vocab_size = tokenize(data)
    np.save('datasets/shakespeare_train.npy', train_data)
    np.save('datasets/shakespeare_test.npy', test_data)
    np.save('datasets/shakespeare_ch_to_ix.npy', ch_to_ix)
    np.save('datasets/shakespeare_ix_to_ch.npy', ix_to_ch)
    np.save('datasets/shakespeare_vocab_size.npy', np.array([vocab_size]))


