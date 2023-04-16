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


# Helper function for calculating the loss in a split
@torch.no_grad()
def split_loss(X, Y, parameters):
    C, W1, B1, W2, B2 = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
    emb = C[X]
    emb_cat = emb.view(emb.shape[0], -1)
    h = torch.tanh(emb_cat @ W1 + B1)
    logits = h @ W2 + B2
    loss = F.cross_entropy(logits, Y)
    print(loss.item())


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
    W2 is multiplied by 0.01 to ensure its a small number while B2 is initialized as 0 so that the initial loss
    is not very big
    """
    g = torch.Generator().manual_seed(42)
    C = torch.randn((vocab_size, n_embd), generator=g)
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
    B1 = torch.randn(n_hidden, generator=g)
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
    B2 = torch.randn(vocab_size, generator=g) * 0

    parameters = [C, W1, B1, W2, B2]
    print("Total number of parameters in this neural network is: ",
          sum(p.nelement() for p in parameters))
    for p in parameters:
        p.requires_grad = True

    # Training parameters
    max_steps = 200000
    batch_size = 32
    print_every = 10000
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
        LEARNING_RATE = 0.1 if i < (max_steps / 2) else 0.01

        for p in parameters:
            p.data += -LEARNING_RATE * p.grad

        # Track stats
        if not i % print_every:
            print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")

        loss_i.append(loss.log10().item())

    # Printing split losses
    print("Training loss")
    split_loss(Xtrain, Ytrain, parameters)

    print("Validation loss")
    split_loss(Xval, Yval, parameters)

    # Sampling from the model
    g = torch.Generator().manual_seed(5813)

    for _ in range(20):
        context = [0]*block_size
        output = []

        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + B1)
            logits = h @ W2 + B2
            probs = F.softmax(logits, dim=1)

            idx = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [idx]
            output.append(idx)

            if not idx:
                break

        print(''.join(i_to_s[i] for i in output[:len(output)-1]))


rnn()
