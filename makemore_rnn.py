"""
Using an RNN for predicting names
"""

# Importing Dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# Helper function to create data
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


# RNN Body
def rnn():
    # Load the words and create a torch tensor
    words = open("names.txt", "r").read().splitlines()

    # Create string to integer and integer to string dictionaries, indexing the . element at position 0
    chars = sorted(list(set(''.join(words))))
    s_to_i = {s: i + 1 for i, s in enumerate(chars)}
    s_to_i['.'] = 0
    i_to_s = {i: s for s, i in s_to_i.items()}
    vocab_size = len(i_to_s)

    """
        Creating the training dataset from the list of words. 80% of the data will be used for training, while 10%
        will be used for validation and 10% will be used for testing
    """
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    block_size = 3
    Xtrain, Ytrain = create_data(words[:n1], s_to_i, block_size)
    Xval, Yval = create_data(words[n1:n2], s_to_i, block_size)
    Xtest, Ytest = create_data(words[n2:], s_to_i, block_size)

    # Defining the embedding and hidden layers
    n_embd = 10
    n_hidden = 200

    """
    Defining the same parameters used in MLP like:
    Generator = for reproducibility
    C = Embeddings
    W1, B1 = The Hidden Layer
    W2, B2 = The Output Layer
    """
    g = torch.Generator().manual_seed(1123)
    C = torch.randn((vocab_size, n_embd), generator=g, requires_grad=True)
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g, requires_grad=True)
    B1 = torch.randn(n_hidden, generator=g, requires_grad=True)
    W2 = torch.randn((n_hidden, vocab_size), generator=g, requires_grad=True)
    B2 = torch.randn(vocab_size, generator=g, requires_grad=True)

    parameters = [C, W1, B1, W2, B2]
    print("Total number of parameters in this neural network is: ",
          sum(p.nelement() for p in parameters))

    # Training parameters
    max_steps = 200000
    batch_size = 32
    loss_i = []

    # Training Loop
    for i in range(max_steps):

        # Constructing minibatch
        idx = torch.randint(0, Xtrain.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtrain[idx], Ytrain[idx]

        # Forward Pass
        """
        We embed the characters into vectors and then concatenate the vectors.
        The hidden layers are pre-activated, while the loss function is defined as a cross-entropy loss
        """
        emb = C[Xb]
        emb_cat = emb.view(emb.shape[0], -1)
        h_preact = emb_cat @ W1 + B1
        h = torch.tanh(h_preact)
        logits = h @ W2 + B2
        loss = F.cross_entropy(logits, Yb)

        # Backward Pass
        for p in parameters:
            p.grad = None

        loss.backward()

        # Update

