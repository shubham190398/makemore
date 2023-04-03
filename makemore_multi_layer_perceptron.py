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
    C = torch.randn((27, 2), generator=g, requires_grad=True)

    # Constructing hidden layer. Number of inputs will be the number of columns in C * block_size
    W1 = torch.randn((block_size*2, 100), generator=g, requires_grad=True)
    B1 = torch.randn(100, generator=g, requires_grad=True)

    # Constructing Output Layer
    W2 = torch.randn((100, 27), generator=g, requires_grad=True)
    B2 = torch.randn(27, generator=g, requires_grad=True)

    # List of parameters
    parameters = [C, W1, B1, W2, B2]

    # Training
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.1

    """
    Finding optimum learning rate

    learning_rate_exponent = torch.linspace(-3, 0, 1000)
    learning_rates = 10**learning_rate_exponent
    learning_rates_i = []
    loss_i = []
    
    After plotting the graph, it was determined that the optimum learning rate is around 0.1
    """

    for i in range(NUM_EPOCHS):

        # Constructing minibatch to reduce amount of processing per iteration
        idx = torch.randint(0, X.shape[0], (32,))

        # Forward Pass
        """
        logits = H @ W2 + B2
        counts = logits.exp()
        probs = counts/counts.sum(1, keepdims=True)
        loss = -probs[torch.arange(32), Y].log().mean()
        The above can be done in one step with cross entropy
        """
        emb = C[X[idx]]
        H = torch.tanh(emb.view(-1, block_size*2) @ W1 + B1)
        logits = H @ W2 + B2
        loss = F.cross_entropy(logits, Y[idx])

        print(f"For epoch {i}, loss = {loss.item()}")

        # Backward pass
        for p in parameters:
            p.grad = None

        loss.backward()

        for p in parameters:
            p.data += -LEARNING_RATE * p.grad

        """
        # Tracking stats for learning rates
        learning_rates_i.append(learning_rate_exponent[i])
        loss_i.append(loss.item())
        """

    # plt.plot(learning_rates_i, loss_i)
    # plt.show()


mlp()
