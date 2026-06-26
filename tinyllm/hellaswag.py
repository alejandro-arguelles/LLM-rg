import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset


def evaluate_hellaswag(model, tokenizer, device, split="validation", max_examples=None):
    """
    Evaluate model on HellaSwag using likelihood-based scoring.
    For each example, pick the candidate ending with the lowest mean cross-entropy
    loss conditioned on the context. Returns accuracy.
    """
    dataset = load_dataset("Rowan/hellaswag", split=split)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    block_size = model.config.block_size
    model.eval()
    correct = 0

    with torch.no_grad():
        for example in tqdm(dataset, desc="HellaSwag"):
            ctx_tokens = tokenizer.encode_ordinary(example["ctx"])
            label = int(example["label"])
            losses = []

            for ending in example["endings"]:
                ending_tokens = tokenizer.encode_ordinary(" " + ending)
                tokens = ctx_tokens + ending_tokens

                # Truncate from the left if the sequence exceeds block_size
                if len(tokens) > block_size:
                    tokens = tokens[-block_size:]
                    # Recalculate where the ending starts after truncation
                    ending_start = max(0, len(tokens) - len(ending_tokens))
                else:
                    ending_start = len(ctx_tokens)

                ids = torch.tensor([tokens], dtype=torch.long).to(device)
                logits = model(ids)  # (1, T, vocab_size)

                # Score only the ending tokens: logits[i] predicts token[i+1]
                end_logits = logits[0, ending_start - 1 : len(tokens) - 1]
                end_targets = ids[0, ending_start : len(tokens)]
                loss = F.cross_entropy(end_logits, end_targets)
                losses.append(loss.item())

            if losses.index(min(losses)) == label:
                correct += 1

    return correct / len(dataset)
