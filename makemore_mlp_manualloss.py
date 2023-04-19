"""
Writing loss.backward() for the MLP ourselves
"""

# Importing Dependencies
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


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
def split_loss(X, Y, parameters, BN_MEAN, BN_STD):
    C, W1, B1, W2, B2, BN_GAIN, BN_BIAS = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], \
        parameters[5], parameters[6]
    emb = C[X]
    emb_cat = emb.view(emb.shape[0], -1)
    h_preact = emb_cat @ W1 + B1
    h_preact = BN_GAIN * ((h_preact - BN_MEAN) / BN_STD) + BN_BIAS
    h = torch.tanh(h_preact)
    logits = h @ W2 + B2
    loss = F.cross_entropy(logits, Y)
    print(loss.item())


# Helper Function for comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    max_diff = (dt - t.grad).abs().max().item()
    print(f"{s:15s} | exact:{str(ex):5s} | approximate: {str(app):5s} | max_diff: {max_diff}")


# Body of Improved MLP
def mlp_manual_loss():
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
    
    The weights and biases have been initialized in a non-standard manner to note errors
    during backpropagation
    """
    g = torch.Generator().manual_seed(42)
    C = torch.randn((vocab_size, n_embd), generator=g)
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / ((n_embd * block_size) ** 0.5)
    B1 = torch.randn(n_hidden, generator=g) * 0.1
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
    B2 = torch.randn(vocab_size, generator=g) * 0.1

    BN_GAIN = torch.ones((1, n_hidden)) * 0.1 + 1.0
    BN_BIAS = torch.zeros((1, n_hidden)) * 0.1
    BN_MEAN = torch.zeros((1, n_hidden))
    BN_STD = torch.ones((1, n_hidden))

    parameters = [C, W1, B1, W2, B2, BN_GAIN, BN_BIAS]
    print("Total number of parameters in this neural network is: ",
          sum(p.nelement() for p in parameters))
    for p in parameters:
        p.requires_grad = True

    # Training parameters
    N = 32
    batch_size = 32
    print_every = 10000
    loss_i = []

    # Training Loop

    # Creating Minibatch
    idx = torch.randint(0, Xtrain.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtrain[idx], Ytrain[idx]

    # Forward Pass
    emb = C[Xb]
    emb_cat = emb.view(emb.shape[0], -1)

    # Linear Layer
    h_pre_bn = emb_cat @ W1 + B1

    # Batch Norm Layer
    bn_mean_i = 1/N * h_pre_bn.sum(0, keepdim=True)
    bn_diff = h_pre_bn - bn_mean_i
    bn_diff2 = bn_diff ** 2
    bn_var = 1/(N-1) * bn_diff2.sum(0, keepdim=True)
    bn_var_inv = (bn_var + 1e-5)**-0.5
    bn_raw = bn_diff + bn_var_inv

    h_pre_act = BN_GAIN * bn_raw + BN_BIAS

    # Adding Non-Linearity
    h = torch.tanh(h_pre_act)

    # Linear layer 2
    logits = h @ W2 + B2

    # Cross Entropy Loss
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdim=True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    log_probs = probs.log()
    loss = -log_probs[range(N), Yb].mean()

    # Pytorch backward pass
    for p in parameters:
        p.grad = None
    for t in [log_probs, probs, counts, counts_sum,
              counts_sum_inv, norm_logits, logit_maxes, logits, h,
              h_pre_act, bn_raw, bn_var_inv, bn_diff2, bn_diff, h_pre_bn,
              bn_mean_i, emb_cat, emb]:
        t.retain_grad()
    loss.backward()
    print(loss)



mlp_manual_loss()