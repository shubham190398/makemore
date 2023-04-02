"""
Train an MLP for generating new text
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Building the training dataset
def create_data(words, s_to_i, block_size):
    X, Y = [], []

    for w in words:
        context = [0] * block_size

        for ch in w + '.':
            idx = s_to_i[ch]
            X.append(context)
            Y.append(idx)
            context = context[1:] + [idx]

    return torch.tensor(X), torch.tensor(Y)


# MLP framework
def mlp():

    # Read the words from the file
    words = open("names.txt", "r").read().splitlines()

    # Build the mapping of the characters to and from integers
    chars = sorted(list(set(''.join(words))))
    s_to_i = {s: i+1 for i, s in enumerate(chars)}
    s_to_i['.'] = 0
    i_to_s = {i: s for s, i in s_to_i.items()}

    # Creating the training dataset from the list of words
    block_size = 3
    X, Y = create_data(words, s_to_i, block_size)

    # Defining a generator for reproducibility
    g = torch.Generator().manual_seed(190398)

    # Creating the embeddings. Due to the power of tensor indexing, we can directly use the array X
    C = torch.randn((27, 2), generator=g)
    emb = C[X]

    # Constructing hidden layer. Number of inputs will be the number of columns in C * block_size
    W1 = torch.randn((block_size*2, 100), generator=g)
    B1 = torch.randn(100, generator=g)
    H = torch.tanh(emb.view(-1, block_size*2) @ W1 + B1)

    # Constructing Output Layer
    W2 = torch.randn((100, 27), generator=g)
    B2 = torch.randn(27, generator=g)

    # Softmax
    """
    logits = H @ W2 + B2
    counts = logits.exp()
    probs = counts/counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(32), Y].log().mean()
    The above can be done in one step with cross entropy
    """
    logits = H @ W2 + B2
    F.cross_entropy(logits, Y)

