"""
Counting the bigrams in the text. Add special characters at the start and the end for ease of identification.
Have one token instead of two because obviously a word cannot start with the end token and the start token cannot follow
a word token. Have a measure of likelihood of a character to make a loss function. We take negative of log likelihood
to have a monotonically increasing function which needs to be minimized.
"""

import torch
import matplotlib.pyplot as plt


def bigram():

    # Load the words and create a torch tensor
    words = open("names.txt", "r").read().splitlines()
    N = torch.zeros((27, 27), dtype=torch.int32)

    # Create string to integer and integer to string dictionaries, indexing the . element at position 0
    chars = sorted(list(set(''.join(words))))
    s_to_i = {s: i+1 for i, s in enumerate(chars)}
    s_to_i['.'] = 0
    i_to_s = {i: s for s, i in s_to_i.items()}

    # Update the frequency of the particular character in the tensor
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            idx1 = s_to_i[ch1]
            idx2 = s_to_i[ch2]
            N[idx1, idx2] += 1

    # Create the lookup table, uncomment the grayed code to visualize the table

    # plt.figure(figsize=(16, 16))
    # plt.imshow(N, cmap='Blues', aspect='auto')

    for i in range(27):
        for j in range(27):
            ch_str = i_to_s[i] + i_to_s[j]
            plt.text(j, i, ch_str, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")

    # plt.axis('off')
    # plt.show()

    # Normalizing N and smoothening the model to prevent infinity in the log likelihood
    P = (N+1).float()
    P /= P.sum(1, keepdim=True)

    # Sampling from N
    g = torch.Generator().manual_seed(1123581321)

    for i in range(20):
        output_names = []
        idx = 0
        while True:
            p = P[idx]
            idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            output_names.append(i_to_s[idx])
            if not idx:
                break
        print(''.join(output_names))

    # Getting Average negative log likelihood

    negative_log_likelihood = 0.0
    number_of_words = 0

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            idx1 = s_to_i[ch1]
            idx2 = s_to_i[ch2]
            prob = P[idx1, idx2]
            negative_log_likelihood -= torch.log(prob)
            number_of_words += 1

    print(f'{negative_log_likelihood/number_of_words:.4f}')


bigram()
