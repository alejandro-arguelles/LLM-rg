"""Interpretability analyzer for the climbing model.

Two analyses, rendered as clean 'pastel tech' plots:

  1. Evolution of a sentence through the layers
       - residual-stream L2 norm per (layer, token)
       - logit lens: when the final next-token prediction "crystallises" across depth

  2. Attention-matrix comparison
       - one sentence, a grid of layers x heads
       - one (layer, head), compared across random sentences from the dataset

The model uses F.scaled_dot_product_attention, which does not return attention
weights, so we recompute them here (q.kᵀ/√d + causal mask + softmax), exactly
mirroring CausalSelfAttention.forward.

Usage:
  PYTHONPATH=. uv run python tinyllm/analyzer.py --all
  PYTHONPATH=. uv run python tinyllm/analyzer.py --evolution --prompt "Climbing is"
  PYTHONPATH=. uv run python tinyllm/analyzer.py --attention --layers 0,5,11 --heads 0,3,7,11
"""
import os
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from tinyllm.models import Config, DecoderTransformer, apply_rotary
from tinyllm.load_data import list_parquet_files
from tinyllm.train import load_checkpoint


# --------------------------------------------------------------------------- #
# Aesthetics: minimal "pastel tech" palette (Anthropic-ish warm neutrals).
# --------------------------------------------------------------------------- #
PALETTE = {
    "bg": "#F0EEE6",      # warm cream background
    "panel": "#FAF9F5",   # near-white panel
    "ink": "#1A1A18",     # near-black text
    "muted": "#8A867C",   # muted captions / ticks
    "grid": "#DEDACE",    # faint gridlines
    "accent": "#CC785C",  # clay accent
}
CATEGORICAL = ["#CC785C", "#6A9FB5", "#9B8AC4", "#7FB3A3",
               "#E0A458", "#C77B9E", "#5E9CA8", "#B5915A",
               "#8DA47E", "#A88BB0", "#D69A77", "#7C9DB5"]

# Sequential maps tuned to the palette.
CMAP_CLAY = LinearSegmentedColormap.from_list(
    "clay", ["#F4F2EA", "#EAD3C2", "#DDAE8E", "#CC785C", "#9C4A33"])
CMAP_MIST = LinearSegmentedColormap.from_list(
    "mist", ["#FAF9F5", "#DCE4DF", "#AEC6C4", "#7FA6AE", "#4C7A86"])


def set_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"],
        "savefig.edgecolor": PALETTE["bg"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": PALETTE["ink"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": PALETTE["muted"],
        "axes.labelsize": 9,
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.grid": False,
        "figure.dpi": 140,
    })


def _clean(ax, keep=()):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def _header(fig, title, subtitle, x=0.04, top=0.985, gap=0.04):
    fig.text(x, top, title, ha="left", va="top", fontsize=15,
             fontweight="bold", color=PALETTE["ink"])
    fig.text(x, top - gap, subtitle, ha="left", va="top",
             fontsize=9.5, color=PALETTE["muted"])


def _soft_colorbar(fig, mappable, ax, label=""):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=6.5, colors=PALETTE["muted"])
    if label:
        cb.set_label(label, fontsize=7.5, color=PALETTE["muted"])
    return cb


# --------------------------------------------------------------------------- #
# Model loading + token helpers
# --------------------------------------------------------------------------- #
# Must match scripts/run_climbing.py / train_climbing.
CLIMBING_CONFIG = dict(embed_dim=768, num_heads=12, head_size=64,
                       block_size=512, num_layers=12)

# Two simple English sentences that share almost no content words
# (cat/feline, sat/rested, mat/rug) yet mean nearly the same thing -- used to
# check whether their representations converge across depth.
DEFAULT_SENTENCE_A = "The cat sat on the mat."
DEFAULT_SENTENCE_B = "A feline rested upon the rug."

# Groups of sentences: within a group they mean similar things with different
# words; across groups they are semantically unrelated (orthogonal topics).
SEMANTIC_GROUPS = {
    "weather": [
        "It is raining hard outside.",
        "The storm brought heavy rain.",
        "Water is pouring from the sky.",
    ],
    "money": [
        "She invested in the stock market.",
        "He bought shares to grow savings.",
        "They put their cash into bonds.",
    ],
    "cooking": [
        "He cooked pasta for dinner.",
        "She prepared a delicious meal.",
        "They baked fresh bread today.",
    ],
    "travel": [
        "They flew to Japan on holiday.",
        "She drove across the country.",
        "He sailed to a distant island.",
    ],
}

# Normal English sentences for the attention grids: clear syntax and pronouns
# that refer back to earlier nouns, so head/layer roles are easy to read.
ATTENTION_SENTENCES = [
    "The dog chased the cat because it was scared.",
    "After she finished her homework, she went to bed.",
    "Paris is the capital of France, and it is beautiful.",
]


def load_model(ckpt_path, device):
    tokenizer = tiktoken.get_encoding("gpt2")
    config = Config(vocab_size=tokenizer.n_vocab, **CLIMBING_CONFIG)
    model = DecoderTransformer(config).to(device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Train first (scripts/run_climbing.py).")
    it = load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()
    print(f"Loaded {ckpt_path} (trained to iter {it})")
    return model, tokenizer, config


def token_labels(tokenizer, ids, maxlen=12):
    """Human-readable per-token strings; leading space -> '·', newline -> '⏎'."""
    labels = []
    for i in ids:
        s = tokenizer.decode([int(i)])
        s = s.replace("\n", "⏎").replace("\t", "⇥")
        if s.startswith(" "):
            s = "·" + s[1:]
        if s == "":
            s = "∅"
        labels.append(s[:maxlen])
    return labels


# --------------------------------------------------------------------------- #
# Forward pass with capture (residual stream + recomputed attention)
# --------------------------------------------------------------------------- #
def attention_weights(attn, x):
    """Recompute softmax attention weights for one CausalSelfAttention module.

    Mirrors its forward up to the softmax. Returns (num_heads, T, T).
    """
    B, T, C = x.shape
    nh, hs = attn.config.num_heads, attn.config.head_size
    kqv = attn.kqv(x)
    k, q, _ = kqv.split(hs * nh, dim=-1)
    k = k.view(B, T, nh, hs).transpose(1, 2)
    q = q.view(B, T, nh, hs).transpose(1, 2)
    cos, sin = attn.rope(T)
    q, k = apply_rotary(q, k, cos, sin)
    att = (q @ k.transpose(-2, -1)) / (hs ** 0.5)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    att = att.masked_fill(~mask, float("-inf"))
    att = torch.softmax(att, dim=-1)
    return att[0]  # (nh, T, T)


@torch.no_grad()
def capture(model, ids):
    """Run the model, returning (residual states, attention maps, logits).

    states: list of (T, C) tensors -- embedding output, then output of each block.
    attn:   list of (nh, T, T) tensors, one per layer.
    """
    x = model.token_embedding(ids)
    states = [x[0].clone()]
    attn = []
    for layer in model.layers:
        n1 = layer.ln1(x)
        attn.append(attention_weights(layer.attn, n1))
        x = x + layer.attn(n1)
        x = x + layer.mlp(layer.ln2(x))
        states.append(x[0].clone())
    logits = model.head(model.ln_f(x))
    return states, attn, logits[0]


@torch.no_grad()
def logit_lens(model, states):
    """Project every residual state through the final norm + head (logit lens).

    Returns M (n_states, T): probability assigned, at each depth, to the token
    the *full* model finally predicts at that position -- i.e. when the
    prediction crystallises.  Also returns the final predicted ids (T,).
    """
    layer_logits = [model.head(model.ln_f(s.unsqueeze(0)))[0] for s in states]
    final_pred = layer_logits[-1].argmax(dim=-1)  # (T,)
    rows = []
    for ll in layer_logits:
        p = F.softmax(ll, dim=-1)
        rows.append(p.gather(-1, final_pred.unsqueeze(-1)).squeeze(-1))
    return torch.stack(rows), final_pred


# --------------------------------------------------------------------------- #
# Dataset sampling
# --------------------------------------------------------------------------- #
def sample_sentences(tokenizer, data_dir, n, max_tokens, seed=0):
    files = list_parquet_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    pf = pq.ParquetFile(files[0])
    texts = pf.read_row_group(0, columns=["text"]).column("text").to_pylist()
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    out = []
    for idx in order:
        t = " ".join(texts[idx].split())  # collapse whitespace
        ids = tokenizer.encode_ordinary(t)[:max_tokens]
        if len(ids) >= 8:
            out.append(ids)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# Analysis 1 -- evolution through layers
# --------------------------------------------------------------------------- #
def plot_evolution(model, tokenizer, prompt, device, out_dir):
    ids = torch.tensor([tokenizer.encode_ordinary(prompt)], dtype=torch.long, device=device)
    labels = token_labels(tokenizer, ids[0].tolist())
    states, _, _ = capture(model, ids)
    M_lens, final_pred = logit_lens(model, states)

    norms = torch.stack([s.norm(dim=-1) for s in states]).cpu().numpy()  # (n_states, T)
    M_lens = M_lens.cpu().numpy()
    n_states, T = norms.shape
    ylabels = ["emb"] + [f"L{i+1}" for i in range(n_states - 1)]
    pred_labels = token_labels(tokenizer, final_pred.tolist())

    fig, axes = plt.subplots(2, 1, figsize=(max(7, T * 0.42), 8.4),
                             gridspec_kw={"hspace": 0.42})

    # Panel A: residual norm growth.
    ax = axes[0]
    im = ax.imshow(norms, aspect="auto", cmap=CMAP_CLAY)
    ax.set_title("Residual-stream magnitude  ·  L2 norm of hidden state, per layer & token", loc="left")
    ax.set_yticks(range(n_states)); ax.set_yticklabels(ylabels)
    ax.set_xticks(range(T)); ax.set_xticklabels(labels, rotation=60, ha="right")
    _clean(ax)
    _soft_colorbar(fig, im, ax, "L2 norm")

    # Panel B: logit lens.
    ax = axes[1]
    im = ax.imshow(M_lens, aspect="auto", cmap=CMAP_MIST, vmin=0, vmax=1)
    ax.set_title("Logit lens  ·  probability of the final next-token prediction, by depth", loc="left")
    ax.set_yticks(range(n_states)); ax.set_yticklabels(ylabels)
    # x labels: input token  →  predicted next token
    xl = [f"{a}→{b}" for a, b in zip(labels, pred_labels)]
    ax.set_xticks(range(T)); ax.set_xticklabels(xl, rotation=60, ha="right")
    _clean(ax)
    _soft_colorbar(fig, im, ax, "P(final pred)")

    _header(fig, "Evolution through the layers",
            f"prompt: “{prompt[:80]}”", top=0.985, gap=0.032)
    fig.subplots_adjust(top=0.87, bottom=0.13, left=0.08, right=0.98, hspace=0.5)
    path = os.path.join(out_dir, "evolution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
# Analysis 1b -- PCA trajectory of two sentences across layers
# --------------------------------------------------------------------------- #
def _unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def pca_2d(X):
    """Project rows of X (N, D) onto their first two principal components."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T
    var = (S ** 2) / (S ** 2).sum()
    return proj, var[:2]


@torch.no_grad()
def sentence_vectors(model, tokenizer, sentence, device, pool="mean"):
    """One vector per layer for a sentence (mean- or last-token pooled)."""
    ids = torch.tensor([tokenizer.encode_ordinary(sentence)], dtype=torch.long, device=device)
    states, _, _ = capture(model, ids)  # list of (T, C)
    vecs = [(s[-1] if pool == "last" else s.mean(dim=0)).cpu().numpy() for s in states]
    return np.stack(vecs)  # (n_states, C)


def plot_pca_evolution(model, tokenizer, sent_a, sent_b, device, out_dir, pool="mean"):
    # Unit-normalise so we track direction (semantics), not the magnitude that
    # grows with depth.
    A = _unit(sentence_vectors(model, tokenizer, sent_a, device, pool))
    B = _unit(sentence_vectors(model, tokenizer, sent_b, device, pool))
    n = A.shape[0]
    proj, var = pca_2d(np.concatenate([A, B], axis=0))
    PA, PB = proj[:n], proj[n:]
    labels = ["emb"] + [f"L{i}" for i in range(1, n)]
    cos = (A * B).sum(axis=-1)  # unit vectors -> dot == cosine

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel A: PCA trajectories.
    ax = axes[0]
    for P, color, name in [(PA, CATEGORICAL[0], sent_a), (PB, CATEGORICAL[1], sent_b)]:
        ax.plot(P[:, 0], P[:, 1], "-", color=color, lw=1.5, alpha=0.45, zorder=1, label=name)
        ax.scatter(P[:, 0], P[:, 1], s=20, color=color, zorder=3,
                   edgecolor=PALETTE["bg"], linewidth=0.6)
        for i, (x, y) in enumerate(P):
            ax.annotate(labels[i], (x, y), fontsize=5.5, color=color,
                        xytext=(3, 3), textcoords="offset points", zorder=4)
        ax.scatter(P[0, 0], P[0, 1], s=70, facecolor="none", edgecolor=color,
                   linewidth=1.4, zorder=5)  # start = hollow
        ax.scatter(P[-1, 0], P[-1, 1], marker="*", s=180, color=color,
                   edgecolor=PALETTE["bg"], linewidth=0.5, zorder=6)  # end = star
    ax.set_title(f"PCA of sentence vectors, layer by layer  ·  "
                 f"PC1 {var[0]*100:.0f}% · PC2 {var[1]*100:.0f}%", loc="left")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    _clean(ax, keep=("left", "bottom"))
    leg = ax.legend(frameon=False, fontsize=8, loc="upper center",
                    bbox_to_anchor=(0.5, -0.1), ncol=2)
    for t, c in zip(leg.get_texts(), (CATEGORICAL[0], CATEGORICAL[1])):
        t.set_color(c)

    # Panel B: cosine similarity across depth.
    ax = axes[1]
    ax.plot(range(n), cos, "-o", color=PALETTE["accent"], lw=1.8, ms=4,
            markeredgecolor=PALETTE["bg"], markeredgewidth=0.5)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(min(0, cos.min()) - 0.05, 1.02)
    ax.axhline(cos[0], color=PALETTE["muted"], lw=0.7, ls=":", alpha=0.7)
    ax.annotate(f"{cos[0]:.2f}", (0, cos[0]), fontsize=7, color=PALETTE["muted"],
                xytext=(2, -10), textcoords="offset points")
    ax.annotate(f"{cos[-1]:.2f}", (n - 1, cos[-1]), fontsize=7.5, color=PALETTE["accent"],
                xytext=(-4, 6), textcoords="offset points", ha="right")
    ax.set_title("Are they converging?  ·  cosine similarity by depth", loc="left")
    _clean(ax, keep=("left", "bottom"))

    _header(fig, "Semantic convergence across layers",
            f"“{sent_a}”  vs  “{sent_b}”   ·   {pool}-pooled, unit-normalised",
            top=0.975, gap=0.05)
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.07, right=0.97, wspace=0.22)
    path = os.path.join(out_dir, "pca_evolution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}  (cosine: emb={cos[0]:.2f} -> final={cos[-1]:.2f})")


# --------------------------------------------------------------------------- #
# Analysis 1c -- PCA of many sentences (similar vs orthogonal groups)
# --------------------------------------------------------------------------- #
def _short(text, n=22):
    return text[:n].rstrip(" .") + ("…" if len(text) > n else "")


def plot_pca_clusters(model, tokenizer, groups, device, out_dir, pool="mean", layer=-1):
    # Flatten groups, keeping the group of each sentence.
    sents, glabels = [], []
    for g, lst in groups.items():
        for s in lst:
            sents.append(s); glabels.append(g)
    group_names = list(groups.keys())
    colors = {g: CATEGORICAL[i % len(CATEGORICAL)] for i, g in enumerate(group_names)}

    # One unit-normalised vector per sentence at the chosen layer.
    V = np.stack([_unit(sentence_vectors(model, tokenizer, s, device, pool)[layer])
                  for s in sents])  # (N, C)
    proj, var = pca_2d(V)
    sim = V @ V.T  # cosine similarity matrix (unit vectors)
    N = len(sents)
    layer_name = "emb" if layer == 0 else (f"L{layer}" if layer > 0 else "final")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # Panel A: PCA scatter, coloured by group, with a faint hull per group.
    ax = axes[0]
    from matplotlib.patches import Polygon
    for g in group_names:
        idx = [i for i in range(N) if glabels[i] == g]
        P = proj[idx]
        if len(P) >= 3:
            c = P.mean(0)
            order = np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
            ax.add_patch(Polygon(P[order], closed=True, facecolor=colors[g],
                                 alpha=0.12, edgecolor=colors[g], lw=0.8, zorder=1))
        ax.scatter(P[:, 0], P[:, 1], s=42, color=colors[g], label=g, zorder=3,
                   edgecolor=PALETTE["bg"], linewidth=0.7)
        for i in idx:
            ax.annotate(_short(sents[i], 18), (proj[i, 0], proj[i, 1]), fontsize=5.5,
                        color=colors[g], xytext=(4, 3), textcoords="offset points", zorder=4)
    ax.set_title(f"PCA of sentence vectors @ {layer_name}  ·  "
                 f"PC1 {var[0]*100:.0f}% · PC2 {var[1]*100:.0f}%", loc="left")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    _clean(ax, keep=("left", "bottom"))
    leg = ax.legend(frameon=False, fontsize=8, loc="upper center",
                    bbox_to_anchor=(0.5, -0.1), ncol=len(group_names))
    for t, g in zip(leg.get_texts(), group_names):
        t.set_color(colors[g])

    # Panel B: cosine similarity matrix (block structure = clustering).
    ax = axes[1]
    im = ax.imshow(sim, cmap=CMAP_MIST, vmin=max(-0.2, sim.min()), vmax=1)
    ax.set_xticks(range(N)); ax.set_xticklabels([_short(s, 14) for s in sents],
                                                rotation=90, fontsize=5)
    ax.set_yticks(range(N)); ax.set_yticklabels([_short(s, 16) for s in sents], fontsize=5)
    # Group separators.
    sizes = [len(groups[g]) for g in group_names]
    edges = np.cumsum(sizes)[:-1] - 0.5
    for e in edges:
        ax.axhline(e, color=PALETTE["bg"], lw=2)
        ax.axvline(e, color=PALETTE["bg"], lw=2)
    ax.set_title("Cosine similarity  ·  block diagonal = topics cluster", loc="left")
    _clean(ax)
    _soft_colorbar(fig, im, ax, "cosine")

    _header(fig, "Similar vs orthogonal sentences",
            f"4 topics × 3 paraphrases   ·   {pool}-pooled, unit-normalised @ {layer_name}",
            top=0.975, gap=0.045)
    fig.subplots_adjust(top=0.82, bottom=0.24, left=0.06, right=0.99, wspace=0.28)
    path = os.path.join(out_dir, "pca_clusters.png")
    fig.savefig(path)
    plt.close(fig)
    # Quick numeric summary: mean within-group vs across-group similarity.
    same = [sim[i, j] for i in range(N) for j in range(N)
            if i < j and glabels[i] == glabels[j]]
    diff = [sim[i, j] for i in range(N) for j in range(N)
            if i < j and glabels[i] != glabels[j]]
    print(f"  wrote {path}  (within-group cos={np.mean(same):.2f}, "
          f"across-group cos={np.mean(diff):.2f})")


# --------------------------------------------------------------------------- #
# Analysis 1d -- evolution of the groups across layers
# --------------------------------------------------------------------------- #
def plot_cluster_evolution(model, tokenizer, groups, device, out_dir, pool="mean"):
    group_names = list(groups.keys())
    colors = {g: CATEGORICAL[i % len(CATEGORICAL)] for i, g in enumerate(group_names)}

    # Per-sentence vectors at every layer -> M (n_states, N, C), unit-normalised.
    mats, sent_group = [], []
    for g in group_names:
        for s in groups[g]:
            mats.append(sentence_vectors(model, tokenizer, s, device, pool))
            sent_group.append(g)
    M = np.stack(mats, axis=1)
    n_states, N = M.shape[0], M.shape[1]
    Mu = M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-8)

    # One unit centroid per (group, layer); fit a single global PCA on all of them.
    centroids = []
    for g in group_names:
        idx = [i for i in range(N) if sent_group[i] == g]
        cen = Mu[:, idx, :].mean(axis=1)
        centroids.append(cen / (np.linalg.norm(cen, axis=-1, keepdims=True) + 1e-8))
    proj, var = pca_2d(np.concatenate(centroids, axis=0))
    proj = proj.reshape(len(group_names), n_states, 2)

    # Within- vs across-group cosine at each layer.
    within, across = [], []
    for l in range(n_states):
        sim = Mu[l] @ Mu[l].T
        w = [sim[i, j] for i in range(N) for j in range(N)
             if i < j and sent_group[i] == sent_group[j]]
        a = [sim[i, j] for i in range(N) for j in range(N)
             if i < j and sent_group[i] != sent_group[j]]
        within.append(np.mean(w)); across.append(np.mean(a))
    within, across = np.array(within), np.array(across)
    labels = ["emb"] + [f"L{i}" for i in range(1, n_states)]
    sizes = 10 + 36 * (np.arange(n_states) / (n_states - 1))  # grow with depth

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1.3, 1]})

    # Panel A: centroid trajectories (marker grows emb -> final).
    ax = axes[0]
    for gi, g in enumerate(group_names):
        P = proj[gi]
        ax.plot(P[:, 0], P[:, 1], "-", color=colors[g], lw=1.3, alpha=0.4, zorder=1)
        ax.scatter(P[:, 0], P[:, 1], s=sizes, color=colors[g], zorder=3,
                   edgecolor=PALETTE["bg"], linewidth=0.5, label=g)
        ax.scatter(P[0, 0], P[0, 1], s=70, facecolor="none", edgecolor=colors[g],
                   linewidth=1.4, zorder=4)  # emb = hollow
        ax.scatter(P[-1, 0], P[-1, 1], marker="*", s=200, color=colors[g],
                   edgecolor=PALETTE["bg"], linewidth=0.5, zorder=5)  # final = star
    ax.set_title(f"Group centroid trajectories  ·  PC1 {var[0]*100:.0f}% · PC2 {var[1]*100:.0f}%"
                 "   (small dot = emb · large star = final)", loc="left", fontsize=9)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    _clean(ax, keep=("left", "bottom"))
    leg = ax.legend(frameon=False, fontsize=8, loc="upper center",
                    bbox_to_anchor=(0.5, -0.1), ncol=len(group_names))
    for t, g in zip(leg.get_texts(), group_names):
        t.set_color(colors[g])

    # Panel B: separation metric across depth.
    ax = axes[1]
    x = range(n_states)
    ax.fill_between(x, across, within, color=PALETTE["accent"], alpha=0.12, zorder=1)
    ax.plot(x, within, "-o", color=CATEGORICAL[3], lw=1.8, ms=3.5,
            markeredgecolor=PALETTE["bg"], markeredgewidth=0.4, label="within group")
    ax.plot(x, across, "-o", color=CATEGORICAL[1], lw=1.8, ms=3.5,
            markeredgecolor=PALETTE["bg"], markeredgewidth=0.4, label="across groups")
    gap = within - across
    best = int(np.argmax(gap))
    ax.axvline(best, color=PALETTE["muted"], lw=0.7, ls=":", alpha=0.8)
    ax.annotate(f"max separation\n{labels[best]}  (Δ={gap[best]:.2f})", (best, within[best]),
                fontsize=7, color=PALETTE["ink"], xytext=(6, 6),
                textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Topic separation by depth  ·  gap = within − across", loc="left")
    _clean(ax, keep=("left", "bottom"))
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for t, c in zip(leg.get_texts(), (CATEGORICAL[3], CATEGORICAL[1])):
        t.set_color(c)

    _header(fig, "How the groups evolve across layers",
            f"{len(group_names)} topics × {len(groups[group_names[0]])} paraphrases"
            f"   ·   {pool}-pooled, unit-normalised",
            top=0.975, gap=0.045)
    fig.subplots_adjust(top=0.82, bottom=0.2, left=0.06, right=0.97, wspace=0.24)
    path = os.path.join(out_dir, "cluster_evolution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}  (max separation at {labels[best]}, "
          f"within={within[best]:.2f} vs across={across[best]:.2f})")


# --------------------------------------------------------------------------- #
# Analysis 1e -- export sentence-final-period embeddings for the
# TensorFlow Embedding Projector  (https://projector.tensorflow.org)
# --------------------------------------------------------------------------- #
def sample_texts(tokenizer, data_dir, n, max_tokens, seed=0):
    """Sample n dataset sentences, each forced to end in a period token.

    Returns a list of (ids, text):
      - ids:  token ids truncated to (max_tokens-1), then a '.' token appended,
              so the FINAL position is always the sentence-ending period.
      - text: tokenizer.decode(ids) -- exactly what produced the vector.
    """
    files = list_parquet_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    period = tokenizer.encode_ordinary(".")  # gpt2 -> [13]
    rng = random.Random(seed)
    out = []
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for rg in range(pf.num_row_groups):
            texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
            order = list(range(len(texts)))
            rng.shuffle(order)
            for idx in order:
                base = " ".join(texts[idx].split())  # collapse whitespace
                body = tokenizer.encode_ordinary(base)[:max_tokens - 1]
                if len(body) < 7:                    # real sentence (+ period -> >=8 tokens)
                    continue
                ids = body + period
                out.append((ids, tokenizer.decode(ids)))
                if len(out) >= n:
                    return out
    return out


@torch.no_grad()
def export_projector(model, tokenizer, data_dir, device, out_dir,
                     n=1000, max_tokens=64, layer=-1, seed=0):
    """Write vectors.tsv + metadata.tsv for the TF Embedding Projector.

    For each sentence we take the residual state of the FINAL token (the
    sentence-ending period) at `layer` (-1 = final block) as its vector. In a
    causal model that position has attended to the whole sentence, so it acts
    as a sentence embedding. Load both files at https://projector.tensorflow.org
    (everything runs client-side) and reduce to 3D with UMAP / t-SNE / PCA.
    """
    samples = sample_texts(tokenizer, data_dir, n, max_tokens, seed)
    if not samples:
        raise RuntimeError(f"No usable sentences sampled from {data_dir}")

    vecs, labels = [], []
    for i, (ids, text) in enumerate(samples):
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        states, _, _ = capture(model, ids_t)            # list of (T, C)
        vecs.append(states[layer][-1].cpu().numpy())     # final token = period
        labels.append(text.replace("\t", " ").replace("\n", " ").strip())
        if (i + 1) % 100 == 0:
            print(f"  embedded {i + 1}/{len(samples)} sentences")

    V = np.stack(vecs)  # (N, C)
    layer_name = "emb" if layer == 0 else (f"L{layer}" if layer > 0 else "final")

    vec_path = os.path.join(out_dir, "vectors.tsv")
    with open(vec_path, "w") as f:
        for row in V:
            f.write("\t".join(f"{x:.6g}" for x in row) + "\n")

    # Single column => the Projector uses each line directly as the point label
    # (NO header line in this case), so the sentence shows up automatically.
    meta_path = os.path.join(out_dir, "metadata.tsv")
    with open(meta_path, "w") as f:
        for label in labels:
            f.write(label + "\n")

    print(f"  wrote {vec_path}  ({V.shape[0]} x {V.shape[1]}, layer={layer_name})")
    print(f"  wrote {meta_path}")
    print("  -> open https://projector.tensorflow.org , click 'Load', pick both "
          "TSVs, then choose UMAP or t-SNE and set the dimension to 3D.")


# --------------------------------------------------------------------------- #
# Analysis 2 -- attention matrices
# --------------------------------------------------------------------------- #
def _attn_cell(ax, A, cmap):
    """Draw one causal attention matrix (upper triangle hidden)."""
    A = np.asarray(A)
    masked = np.where(np.tril(np.ones_like(A)) > 0, A, np.nan)
    cmap = cmap.copy(); cmap.set_bad(PALETTE["bg"])
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=np.nanmax(masked))
    _clean(ax)
    return im


def plot_attention_grid(model, tokenizer, sentence_ids, layers, heads, device, out_dir,
                        name="attention_grid", sentence_text=None):
    ids = torch.tensor([sentence_ids], dtype=torch.long, device=device)
    labels = token_labels(tokenizer, sentence_ids, maxlen=8)
    _, attn, _ = capture(model, ids)
    T = len(sentence_ids)

    nr, nc = len(layers), len(heads)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 1.9 + 1.2, nr * 1.9 + 1.4),
                             squeeze=False)
    last_im = None
    for r, L in enumerate(layers):
        for c, H in enumerate(heads):
            ax = axes[r][c]
            last_im = _attn_cell(ax, attn[L][H].cpu().numpy(), CMAP_MIST)
            if r == 0:
                ax.set_title(f"head {H}", fontsize=8.5, color=PALETTE["ink"], pad=4)
            if c == 0:
                ax.set_ylabel(f"layer {L}", fontsize=8.5, color=PALETTE["ink"],
                              rotation=90, labelpad=6)
            # token labels only on the outer edges, and only if short enough
            if r == nr - 1 and T <= 22:
                ax.set_xticks(range(T)); ax.set_xticklabels(labels, rotation=90, fontsize=4.5)
            else:
                ax.set_xticks([])
            if c == 0 and T <= 22:
                ax.set_yticks(range(T)); ax.set_yticklabels(labels, fontsize=4.5)
            else:
                ax.set_yticks([])

    subtitle = "rows = query position (top→down) · cols = key position"
    if sentence_text:
        subtitle += f"   ·   “{sentence_text}”"
    _header(fig, "Attention patterns  ·  layers × heads", subtitle,
            top=0.985, gap=0.035)
    fig.subplots_adjust(top=0.88, bottom=0.1, left=0.1, right=0.97, wspace=0.12, hspace=0.12)
    if last_im is not None:
        cax = fig.add_axes([0.1, 0.045, 0.3, 0.012])
        cb = fig.colorbar(last_im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=6, colors=PALETTE["muted"])
        cb.set_label("attention weight", fontsize=7, color=PALETTE["muted"])
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_attention_compare(model, tokenizer, sentences, layer, head, device, out_dir):
    n = len(sentences)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.5 + 0.8, 4.0), squeeze=False)
    last_im = None
    for c, sent in enumerate(sentences):
        ids = torch.tensor([sent], dtype=torch.long, device=device)
        labels = token_labels(tokenizer, sent, maxlen=8)
        _, attn, _ = capture(model, ids)
        T = len(sent)
        ax = axes[0][c]
        last_im = _attn_cell(ax, attn[layer][head].cpu().numpy(), CMAP_CLAY)
        ax.set_title(f"sentence {c + 1}", fontsize=8.5, color=PALETTE["ink"], pad=4)
        if T <= 22:
            ax.set_xticks(range(T)); ax.set_xticklabels(labels, rotation=90, fontsize=4.5)
            ax.set_yticks(range(T)); ax.set_yticklabels(labels, fontsize=4.5)
        else:
            ax.set_xticks([]); ax.set_yticks([])

    _header(fig, f"Same head across sentences  ·  layer {layer}, head {head}",
            "is the pattern consistent? (e.g. previous-token / first-token heads)",
            top=0.97, gap=0.06)
    fig.subplots_adjust(top=0.74, bottom=0.16, left=0.06, right=0.97, wspace=0.18)
    path = os.path.join(out_dir, "attention_compare.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Interpretability analyzer for the climbing model")
    p.add_argument("--ckpt", default="checkpoints/tinyllm_climbing.pth")
    p.add_argument("--data-dir", default=os.path.join(os.environ.get("HOME", ""), "climbmix"))
    p.add_argument("--device", default="cpu", help="cpu or cuda (cpu avoids interfering with training)")
    p.add_argument("--out", default="analysis_out")
    p.add_argument("--prompt", default="The most important thing about climbing is")
    p.add_argument("--sentence-a", default=DEFAULT_SENTENCE_A)
    p.add_argument("--sentence-b", default=DEFAULT_SENTENCE_B)
    p.add_argument("--pool", default="mean", choices=["mean", "last"],
                   help="how to pool tokens into one sentence vector for PCA")
    p.add_argument("--cluster-layer", type=int, default=-1,
                   help="layer index for the cluster PCA (0=emb, -1=final)")
    p.add_argument("--projector-count", type=int, default=1000,
                   help="how many sentences to embed for the projector export")
    p.add_argument("--projector-max-tokens", type=int, default=64,
                   help="max tokens per sentence before the final period")
    p.add_argument("--projector-layer", type=int, default=-1,
                   help="layer index for the projector embedding (0=emb, -1=final)")
    p.add_argument("--layers", default="0,5,11", help="comma list for the attention grid")
    p.add_argument("--heads", default="0,3,7,11", help="comma list for the attention grid")
    p.add_argument("--compare-layer", type=int, default=5)
    p.add_argument("--compare-head", type=int, default=4)
    p.add_argument("--num-sentences", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evolution", action="store_true")
    p.add_argument("--pca", action="store_true")
    p.add_argument("--clusters", action="store_true")
    p.add_argument("--cluster-evolution", action="store_true")
    p.add_argument("--attention", action="store_true")
    p.add_argument("--projector", action="store_true",
                   help="export vectors.tsv + metadata.tsv for projector.tensorflow.org")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    # --projector is a standalone export (interactive, heavier); it is never
    # part of --all and does not trigger the default-everything fallback.
    if not (args.evolution or args.pca or args.clusters
            or args.cluster_evolution or args.attention or args.projector):
        args.all = True

    set_style()
    # Each run writes to a timestamped subfolder so previous runs are kept.
    run_dir = os.path.join(args.out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output dir: {run_dir}")
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model, tokenizer, _ = load_model(args.ckpt, device)

    if args.all or args.evolution:
        print("Evolution through layers:")
        plot_evolution(model, tokenizer, args.prompt, device, run_dir)

    if args.all or args.pca:
        print("PCA semantic convergence:")
        plot_pca_evolution(model, tokenizer, args.sentence_a, args.sentence_b,
                           device, run_dir, pool=args.pool)

    if args.all or args.clusters:
        print("PCA similar vs orthogonal:")
        plot_pca_clusters(model, tokenizer, SEMANTIC_GROUPS, device, run_dir,
                          pool=args.pool, layer=args.cluster_layer)

    if args.all or args.cluster_evolution:
        print("Cluster evolution across layers:")
        plot_cluster_evolution(model, tokenizer, SEMANTIC_GROUPS, device, run_dir,
                               pool=args.pool)

    if args.projector:
        print("Embedding projector export (sentence-final-period vectors):")
        export_projector(model, tokenizer, args.data_dir, device, run_dir,
                         n=args.projector_count, max_tokens=args.projector_max_tokens,
                         layer=args.projector_layer, seed=args.seed)

    if args.all or args.attention:
        print("Attention analysis:")
        layers = [int(x) for x in args.layers.split(",")]
        heads = [int(x) for x in args.heads.split(",")]
        sent_ids = [tokenizer.encode_ordinary(s) for s in ATTENTION_SENTENCES]
        for letter, text, ids in zip("abc", ATTENTION_SENTENCES, sent_ids):
            plot_attention_grid(model, tokenizer, ids, layers, heads, device, run_dir,
                                name=f"attention_grid_phrase_{letter}", sentence_text=text)
        plot_attention_compare(model, tokenizer, sent_ids, args.compare_layer,
                               args.compare_head, device, run_dir)

    print(f"Done. Plots in {run_dir}/")


if __name__ == "__main__":
    main()
