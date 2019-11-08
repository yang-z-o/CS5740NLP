import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os

import time
from tqdm import tqdm
from data_loader import fetch_data

unk = '<UNK>'


class RNN(nn.Module):
	def __init__(self, vocab_size, input_size, hidden_size, output_size): # Add relevant parameters
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.embeds = nn.Embedding(vocab_size, input_size)
		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.h2o = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax()
		self.loss = nn.NLLLoss()

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)	

	def step(self, input, last_hidden):
		embed = self.embeds(torch.tensor([input], dtype=torch.long))
		input = torch.cat((embed, last_hidden), 1)
		hidden = self.i2h(input)
		output = self.h2o(hidden)
		return hidden, output

	def forward(self, inputs): 
		last_hidden = torch.tensor(np.zeros(self.hidden_size).reshape(1,-1), dtype=torch.float)
		for input in inputs:
			last_hidden, output = self.step(input, last_hidden)
		predicted_vector = self.softmax(output)
		return predicted_vector


def main(input_size, hidden_size):
	train_data, valid_data = fetch_data() 

	# data pre-processing 
	train_data, _ = fetch_data()
	vocab = set()
	for document, _ in train_data:
		sub_vocab = set(document)
		vocab = vocab.union(sub_vocab)
	vocab.add('unk')
	word_to_idx = {word: i for i, word in enumerate(vocab)}

	model = RNN(len(vocab), input_size, hidden_size, 5)
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) 
	model.train()

	correct, total = 0, 0
	for document, gold_label in train_data:
		optimizer.zero_grad()
		input_vector = [word_to_idx[word] for word in document]
		predicted_vector = model(input_vector)
		predicted_label = torch.argmax(predicted_vector)
		correct += int(predicted_label == gold_label)
		total += 1
		#print(predicted_vector, predicted_label, gold_label)
		loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
		loss.backward()
		optimizer.step()
	
	print("Training accuracy: {}".format(correct / total))

	# Set the model to evaluation mode
	model.train(False) 
	correct, total = 0, 0
	for document, gold_label in valid_data:
		input_vector = [word_to_idx.get(word,word_to_idx['unk']) for word in document]
		predicted_vector = model(input_vector)
		predicted_label = torch.argmax(predicted_vector)
		correct += int(predicted_label == gold_label)
		total += 1
	print("Validation accuracy: {}".format(correct / total))