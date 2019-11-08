import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_loader import fetch_data

# data pre-processing 
train_data, _ = fetch_data()
vocab = set()
for document, _ in train_data:
    sub_vocab = set(document)
    vocab = vocab.union(sub_vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}

print(len(vocab))
for document, _ in train_data:
    input_vector = [word_to_idx[word] for word in document]
    print(input_vector)

# # vectorization 
# embeds = nn.Embedding(len(vocab), 10)
# print(embeds)
# word_embed = {}
# for i in range(len(word_to_idx)):
#     word_embed[i] = embeds(torch.tensor([i], dtype=torch.long))
# print(word_embed[0])

# class RNN(nn.Module):
#     def __init__(self, vocab_size, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.embeds = nn.Embedding(vocab_size, input_size)
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.h2o = nn.Linear(hidden_size, output_size)

#     def forward(self, input, last_hidden):
#         input = torch.cat((input, last_hidden), 1)
#         hidden = self.i2h(input)
#         output = self.h2o(hidden)
#         return hidden, output


# # rnn = RNN(, , )
# # label = rnn(hello_embed[0],)