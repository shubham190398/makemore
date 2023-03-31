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

    # Creating the embeddings. Due to the power of tensor indexing, we can directly use the array X
    C = torch.randn((27, 2))
    emb = C[X]
