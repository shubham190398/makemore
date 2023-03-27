"""
Counting the bigrams in the text. Add special characters at the start and the end for ease of identification.
"""

import torch
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()
N = torch.zeros((28, 28), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
s_to_i = {s: i for i, s in enumerate(chars)}
s_to_i['<S>'] = 26
s_to_i['<E>'] = 27

for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        idx1 = s_to_i[ch1]
        idx2 = s_to_i[ch2]
        N[idx1, idx2] += 1

i_to_s = {i: s for s, i in s_to_i.items()}

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(28):
    for j in range(28):
        ch_str = i_to_s[i] + i_to_s[j]
        plt.text(j, i, ch_str, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")

plt.axis('off')
plt.show()
