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
    n1 = int(0.85 * len(words))
    n2 = int(0.95 * len(words))

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
    g = torch.Generator().manual_seed(532)
    C = torch.randn((vocab_size, n_embd), generator=g)
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / ((n_embd * block_size) ** 0.5)
    B1 = torch.randn(n_hidden, generator=g) * 0.1
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
    B2 = torch.randn(vocab_size, generator=g) * 0.01

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
    bn_mean_i = (1/N) * h_pre_bn.sum(0, keepdim=True)
    bn_diff = h_pre_bn - bn_mean_i
    bn_diff2 = bn_diff ** 2
    bn_var = 1/(N-1) * bn_diff2.sum(0, keepdim=True)
    bn_var_inv = (bn_var + 1e-5)**-0.5
    bn_raw = bn_diff * bn_var_inv

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
              h_pre_act, bn_raw, bn_var, bn_var_inv, bn_diff2, bn_diff, h_pre_bn,
              bn_mean_i, emb_cat, emb]:
        t.retain_grad()
    loss.backward()
    print(loss)

    # Manual Backpropagation
    """
    d_log_probs/dx = -1/N for each element indexed by Yb and 0 for all other elements
    d_probs/dx = 1/probs
    d_counts_sum_inv requires broadcasting the sum
    d_counts has 2 branches added together
    d_count_sum/dx = -counts_sum ** -2
    d_norm_logits/dx = counts
    d_logits_maxes requires broadcasting
    d_logits has 2 branches
    d_h, d_W2 and d_B2 are obtained via matrix multiplication with d_logits
    d_h_pre_act/dx = (1 - h**2)
    d_BN_GAIN, d_bn_raw and d_BN_BIAS are obtained from their elementwise multiplication
    d_bn_var_inv/dx = bn_diff
    d_bn_var/dx = -0.5*(bn_var + 1e-5)**-1.5
    d_bn_diff has 2 branches
    d_bn_diff2/dx = 2 * bn_diff along with broadcasting
    d_bn_mean_i also has broadcasting
    d_h_pre_bn requires broadcasting
    d_emb_cat, d_W1 and d_B1 are obtained via matrix multiplication with d_h_pre_bn
    d_emb is just a representing the data of d_emb_cat in its original view
    d_C is undoing the indexing used to create emb
    """
    d_log_probs = torch.zeros_like(log_probs)
    d_log_probs[range(N), Yb] = -1.0/N
    cmp('log_probs', d_log_probs, log_probs)

    d_probs = (1.0/probs) * d_log_probs
    cmp('probs', d_probs, probs)

    d_counts_sum_inv = (counts * d_probs).sum(1, keepdim=True)
    cmp('counts_sum_inv', d_counts_sum_inv, counts_sum_inv)

    d_counts = counts_sum_inv * d_probs

    d_counts_sum = (-counts_sum ** -2) * d_counts_sum_inv
    cmp('counts_sum', d_counts_sum, counts_sum)

    d_counts += torch.ones_like(counts) * d_counts_sum
    cmp('counts', d_counts, counts)

    d_norm_logits = counts * d_counts
    cmp('norm logits', d_norm_logits, norm_logits)

    d_logits = d_norm_logits.clone()

    d_logit_maxes = (-d_norm_logits).sum(1, keepdim=True)
    cmp('logit_maxes', d_logit_maxes, logit_maxes)

    d_logits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * d_logit_maxes
    cmp('logits', d_logits, logits)

    d_h = d_logits @ W2.T
    cmp('h', d_h, h)

    d_W2 = h.T @ d_logits
    cmp('W2', d_W2, W2)

    d_B2 = d_logits.sum(0)
    cmp('B2', d_B2, B2)

    d_h_pre_act = (1 - h**2) * d_h
    cmp('h_pre_act', d_h_pre_act, h_pre_act)

    d_BN_GAIN = (bn_raw * d_h_pre_act).sum(0)
    cmp('BN_GAIN', d_BN_GAIN, BN_GAIN)

    d_bn_raw = BN_GAIN * d_h_pre_act
    cmp('bn_raw', d_bn_raw, bn_raw)

    d_BN_BIAS = d_h_pre_act.sum(0, keepdim=True)
    cmp('BN_BIAS', d_BN_BIAS, BN_BIAS)

    d_bn_diff = bn_var_inv * d_bn_raw

    d_bn_var_inv = (bn_diff * d_bn_raw).sum(0, keepdim=True)
    cmp('bn_var_inv', d_bn_var_inv, bn_var_inv)

    d_bn_var = (-0.5 * (bn_var + 1e-5)**-1.5) * d_bn_var_inv
    cmp('bn_var', d_bn_var, bn_var)

    d_bn_diff2 = (1.0/(N-1)) * torch.ones_like(bn_diff2) * d_bn_var
    cmp('bn_diff2', d_bn_diff2, bn_diff2)

    d_bn_diff += 2 * bn_diff * d_bn_diff2
    cmp('bn_diff', d_bn_diff, bn_diff)

    d_hp_pre_bn = d_bn_diff.clone()

    d_bn_mean_i = -d_bn_diff.sum(0)
    cmp('bn_mean_i', d_bn_mean_i, bn_mean_i)

    d_hp_pre_bn += (1.0/N) * torch.ones_like(h_pre_bn) * d_bn_mean_i
    cmp('hp_pre_bn', d_hp_pre_bn, h_pre_bn)

    d_emb_cat = d_hp_pre_bn @ W1.T
    cmp('emb_cat', d_emb_cat, emb_cat)

    d_W1 = emb_cat.T @ d_hp_pre_bn
    cmp('W1', d_W1, W1)

    d_B1 = d_hp_pre_bn.sum(0, keepdim=True)
    cmp('B1', d_B1, B1)

    d_emb = d_emb_cat.view(emb.shape)
    cmp('emb', d_emb, emb)

    d_C = torch.zeros_like(C)
    for j in range(Xb.shape[0]):
        for k in range(Xb.shape[1]):
            d_C[Xb[j, k]] += d_emb[j, k]
    cmp('C', d_C, C)


mlp_manual_loss()
