"""
A much more indepth neural network implemented with PyTorch
"""

# Importing dependencies
import torch
import random
import torch.nn.functional as F


# Global variable
g = torch.Generator().manual_seed(1234)


# Defining the linear layer
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
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
            x_mean = x.mean(0, keepdim=True)
            x_var = x.var(0, keepdim=True)
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

    @staticmethod
    def parameters():
        return []


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
        Creating the training dataset from the list of words. 85% of the data will be used for training, while 10%
        will be used for validation and 5% will be used for testing
    """
    random.seed(60)
    random.shuffle(words)
    n1 = int(0.85 * len(words))
    n2 = int(0.95 * len(words))

    block_size = 3
    Xtrain, Ytrain = create_data(words[:n1], s_to_i, block_size)
    Xval, Yval = create_data(words[n1:n2], s_to_i, block_size)
    Xtest, Ytest = create_data(words[n2:], s_to_i, block_size)

    # Defining the layers
    n_emb = 10
    n_hidden = 100

    C = torch.randn((vocab_size, n_emb), generator=g)
    layers = [
        Linear(n_emb * block_size, n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ]

    # Initializing the layers
    with torch.no_grad():
        layers[-1].weight *= 0.1

        for layer in layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= 5/3

    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    print("Total parameters: ", sum(p.nelement() for p in parameters))

    # Adding gradient computation
    for p in parameters:
        p.requires_grad = True
