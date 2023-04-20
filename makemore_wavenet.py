"""
Wavenet framework for generating more text out of a file
"""

# Importing Dependencies
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


# Defining the linear layer
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


# Defining the batch normalization layer
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            x_mean = x.mean(dim, keepdim=True)
            x_var = x.var(dim, keepdim=True)
        else:
            x_mean = self.running_mean
            x_var = self.running_var

        x_hat = (x - x_mean)/torch.sqrt(x_var + self.eps)
        self.out = self.gamma * x_hat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# Definition for non-linearity
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


# Definition for embedding
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, idx):
        self.out = self.weight[idx]
        return self.out

    def parameters(self):
        return [self.weight]


# Definition for Flatten:
class Flatten:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)

        if x.shape[1] == 1:
            x = x.squeeze(1)

        return x

    def parameters(self):
        return []


# Definition for Sequential Layer:
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


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
def split_loss(X, Y, model):
    logits = model(X)
    loss = F.cross_entropy(logits, Y)

    return loss.item()


# Body of Wavenet
def wavenet():

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
    random.seed(32)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    block_size = 8
    Xtrain, Ytrain = create_data(words[:n1], s_to_i, block_size)
    Xval, Yval = create_data(words[n1:n2], s_to_i, block_size)
    Xtest, Ytest = create_data(words[n2:], s_to_i, block_size)

    # Setting torch seed
    torch.manual_seed(1223)

    # Defining the embedding and hidden layers
    n_embd = 10
    n_hidden = 200

    model = Sequential([
        Embedding(vocab_size, n_embd),
        Flatten(2),
        Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Flatten(2),
        Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Flatten(2),
        Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])

    # Initializing the parameters
    with torch.no_grad():
        model.layers[-1].weight *= 0.1

    parameters = model.parameters()
    print("Total parameters: ", sum(p.nelement() for p in parameters))

    # Switch on gradient for the parameters
    for p in parameters:
        p.requires_grad = True

    # Training
    max_steps = 200000
    batch_size = 32
    loss_i = []
    print_every = 10000

    for i in range(max_steps):

        # Constructing minibatch
        idx = torch.randint(0, Xtrain.shape[0], (batch_size,))
        Xb, Yb = Xtrain[idx], Ytrain[idx]

        # Forward Pass
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # Update
        LEARNING_RATE = 0.1 if i < 150000 else 0.01
        for p in parameters:
            p.data += -LEARNING_RATE * p.grad

        # Tracking stats
        if not i % print_every:
            print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
        loss_i.append(loss.log10().item())

    """
    Plotting losses while taking an average of 1000 losses at a time. Achieved
    by taking a view of the loss matrix
    """
    plt.plot(torch.tensor(loss_i).view(-1, 1000).mean(1))
    plt.show()

    # Evaluating the model
    for layer in model.layers:
        layer.training = False

    print('Training loss: ', split_loss(Xtrain, Ytrain, model))
    print('Validation loss: ', split_loss(Xval, Yval, model))

    # Sampling from the model
    for _ in range(20):
        context = [0]*block_size
        output = []

        while True:
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)

            idx = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [idx]
            output.append(idx)

            if not idx:
                break

        print(''.join(i_to_s[i] for i in output[:len(output)-1]))


wavenet()
