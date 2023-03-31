"""
Making a neural network to generate names. First we generate the training dataset, and then make a table of log counts.
We can interpret the result of the neural network as log counts by exponentiation. The logits count matrix generated is
equivalent to the N matrix generated before.
"""

import torch
import torch.nn.functional as F


# Generating training data for the neural net
def create_data(words, s_to_i):
    x, y = [], []

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            idx1 = s_to_i[ch1]
            idx2 = s_to_i[ch2]
            x.append(idx1)
            y.append(idx2)

    return torch.tensor(x), torch.tensor(y)


# Main neural net code
def neural_net():
    # Load the words and create a torch tensor
    words = open("names.txt", "r").read().splitlines()

    # Create string to integer and integer to string dictionaries, indexing the . element at position 0
    chars = sorted(list(set(''.join(words))))
    s_to_i = {s: i + 1 for i, s in enumerate(chars)}
    s_to_i['.'] = 0

    # Create one hot encodings for the entire dataset. Cast it to float explicitly as otherwise the datatype is int
    x, y = create_data(words, s_to_i)
    num_elements = x.nelement()
    xenc, yenc = F.one_hot(x, num_classes=27).float(), F.one_hot(y, num_classes=27).float()

    # Initialize weights and add gradient to them for loss back propagation
    W = torch.randn((27, 27), requires_grad=True)
    learning_rate = 50
    num_epochs = 150

    # Training Loop
    for i in range(num_epochs):

        # Forward Pass
        # To get the weight of one neuron for one character, essentially multiply the weight matrix.
        # So (xenc @ W)[3, 13] = xenc[3] * W[:, 13].sum()
        logits = xenc @ W

        # Implementing softmax
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        loss = -probs[torch.arange(num_elements), y].log().mean()
        print(f"Loss for epoch {i + 1} = {loss.item()}")

        # Backward Pass
        W.grad = None
        loss.backward()

        # Update by multiplying learning rate with the gradients and then nudging the data in the opposite direction
        W.data += -learning_rate * W.grad


neural_net()
