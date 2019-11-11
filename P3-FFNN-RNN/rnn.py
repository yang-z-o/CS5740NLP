import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import pickle

import time
from tqdm import tqdm
from data_loader import fetch_data

unk = '<UNK>'

class RNN(nn.Module):
	def __init__(self, vocab_size, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.embeds = nn.Embedding(vocab_size, input_size)

		self.i2h = nn.Linear(input_size, hidden_size)
		self.rnn = nn.GRU(hidden_size, hidden_size)
		self.h2o = nn.Linear(hidden_size, output_size)

		self.softmax = nn.LogSoftmax(dim=1)
		self.loss = nn.NLLLoss()

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)	

	def step(self, input, last_hidden):
		input = self.embeds(torch.tensor([input], dtype=torch.long))
		hidden = self.i2h(input.view(1, -1)).unsqueeze(1)
		output, hidden = self.rnn(hidden, last_hidden)
		output = self.h2o(output.squeeze(1))
		# output = self.h2o(hidden)
		return hidden, output

	def forward(self, inputs): 
		last_hidden = None
		for input in inputs:
			last_hidden, output = self.step(input, last_hidden)
		predicted_vector = self.softmax(output)
		return predicted_vector


def main(input_size, hidden_size, number_of_epochs):
	train_data, valid_data = fetch_data() 

	# data pre-processing 
	train_data, _ = fetch_data()
	vocab = set()
	for document, _ in train_data:
		sub_vocab = set(document)
		vocab = vocab.union(sub_vocab)
	vocab.add('unk')
	
	word_to_idx = {word: i for i, word in enumerate(vocab)}
	model = torch.load('rnn_model.pkl')
	#model = RNN(len(vocab), input_size, hidden_size, 5)
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) 
	
	for epoch in range(number_of_epochs):
		start_time = time.time()
		print("Training for {} epochs".format(epoch + 1))
		model.train()
		optimizer.zero_grad()
		loss = None
		correct, total = 0, 0
		random.shuffle(train_data)
		minibatch_size = 16
		N = len(train_data) 
		for minibatch_index in tqdm(range(N // minibatch_size)):
			optimizer.zero_grad()
			loss = None
			for example_index in range(minibatch_size):
				document, gold_label = train_data[minibatch_index * minibatch_size + example_index]
				input_vector = [word_to_idx[word] for word in document]
				predicted_vector = model(input_vector)
				predicted_label = torch.argmax(predicted_vector)
				correct += int(predicted_label == gold_label)
				total += 1
				example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
				if loss is None:
					loss = example_loss
				else:
					loss += example_loss
			loss = loss / minibatch_size
			loss.backward()
			optimizer.step()
			#torch.save(model, 'rnn_model.pkl')
		
		print("Training accuracy: {}".format(correct / total))
		print("Training time for this epoch: {}".format(time.time() - start_time))

		# Set the model to evaluation mode
		model.train(False) 
		correct, total = 0, 0
		start_time = time.time()
		print("Validation started for epoch {}".format(epoch + 1))
		for document, gold_label in valid_data:
			input_vector = [word_to_idx.get(word,word_to_idx['unk']) for word in document]
			predicted_vector = model(input_vector)
			predicted_label = torch.argmax(predicted_vector)
			correct += int(predicted_label == gold_label)
			total += 1
		print("Validation accuracy: {}".format(correct / total))
		print("Validation time for this epoch: {}".format(time.time() - start_time))
		