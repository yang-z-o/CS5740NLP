import sys
import os
import numpy as np

def main():
	training_file_path = 'DATASET/train/' + sys.argv[1] + '.txt'
	validation_file_path = 'DATASET/validation/' + sys.argv[2] + '.txt'
	with open(training_file_path) as training_file:
		train_count, unicounts, bicounts = word_counter(training_file)
	with open(validation_file_path) as validation_file:
		# smooth_training(validation_file, train_count, unicounts, bicounts)
		# interpolation_smoothing(validation_file, train_count, unicounts, bicounts)
		kneser_ney_smoothing(validation_file, train_count, unicounts, bicounts)

def word_counter(file):
	unicounts, bicounts = {}, {}
	train_count = 0
	seen_words = set()
	for line in file:
		words = line.strip().split(' ')
		words.insert(0, 'unk')
		words.append('unk')
		train_count += 1
		unigram_unk(words[0].lower(), seen_words, unicounts)
		for i in range(1, len(words)):
			train_count += 1
			unigram_unk(words[i].lower(), seen_words, unicounts)
			bigram_unk(words[i - 1].lower(), words[i].lower(), unicounts, bicounts)
	return train_count, unicounts, bicounts

def unigram_unk(word, seen_words, unicounts):
	if word in seen_words:
		unicounts[word] = unicounts.get(word, 0) + 1
	else:
		unicounts['unk'] = unicounts.get('unk', 0) + 1
		seen_words.add(word)

def bigram_unk(word1, word2, unicounts, bicounts):
	s1 = unicounts.get(word1, 0)
	s2 = unicounts.get(word2, 0)
	if s1 > 0:
		if s2 > 0:
			bicountspp(word1, word2, bicounts)
		else:
			bicountspp(word1, 'unk', bicounts)
	else:
		if s2 > 0:
			bicountspp('unk', word2, bicounts)
		else:
			bicountspp('unk', 'unk', bicounts)

def bicountspp(word1, word2, bicounts):
	bicounts[word1 + '+' + word2] = bicounts.get(word1 + '+' + word2, 0) + 1

def add_k_probability(train_count, unicounts, bicounts, k):
	p1, p2 = {}, {}
	for wi, ci in unicounts.items():
		p1[wi] = float(ci + k) / (train_count + k * len(unicounts))
		for wj, cj in unicounts.items():
			p2[wj + '|' + wi] = float(bicounts.get(wi + '+' + wj, 0) + k) / (ci + k * len(unicounts))
	return p1, p2

def interpolation_probability(train_count, unicounts, bicounts, lambd):
	p1, p2 = {}, {}
	for wi, ci in unicounts.items():
		p1[wi] = float(ci) / train_count
		for wj, cj in unicounts.items():
			p2[wj + '|' + wi] = lambd * (float(bicounts.get(wi + '+' + wj, 0)) / ci) + (1 - lambd) * p1[wi]
	return p1, p2

def kneser_ney_probability(train_count, unicounts, bicounts):
	p1, p2, application_time, continuation_time = {}, {}, {}, {}
	for wi, ci in unicounts.items():
		p1[wi] = float(ci) / train_count
		for wj, cj in bicounts.items():
			if wi + '+' in wj:
				application_time[wi] = application_time.get(wi, 0) + cj
			if '+' + wi in wj:
				continuation_time[wi] = continuation_time.get(wi, 0) + cj
	for wi, ci in unicounts.items():
		for wj, cj in unicounts.items():
			bicount = bicounts.get(wi + '+' + wj, 0)
			if bicount > 1:
				d = 0.75
			else:
				d = 0.5
			unicount = ci
			discounted_bigram = max((bicount - d), 0) / unicount
			normalized_discount = d / unicount
			lambd = normalized_discount * application_time[wi]
			interpolation_unigram = lambd * continuation_time[wj] / len(bicounts)
			p2[wj + '|' + wi] = discounted_bigram + interpolation_unigram
	return p1, p2

def validation(validation_file, p1, p2, unicounts):
	pp1,pp2 = 0,0
	validation_count = 0
	for line in validation_file:
		words = line.strip().split(' ')
		validation_count += 1
		pp1 += unigram_test(words[0].lower(), unicounts, p1)
		for i in range(1, len(words)):
			validation_count += 1
			pp1 += unigram_test(words[i].lower(), unicounts, p1)
			pp2 += bigram_test(words[i - 1].lower(), words[i].lower(), unicounts, p2)
	return np.exp(- pp1 / validation_count), np.exp(- pp2 / validation_count)

def unigram_test(word, unicounts, p1):
	if unicounts.get(word,0) != 0:
		return np.log(p1[word])
	else:
		return np.log(p1['unk'])

def bigram_test(word1, word2, unicounts, p2):
	s1 = unicounts.get(word1, 0)
	s2 = unicounts.get(word2, 0)
	if s1 > 0:
		if s2 > 0:
			return np.log(p2[word2 + '|' + word1])
		else:
			return np.log(p2['unk' + '|' + word1])
	else:
		if s2 > 0:
			return np.log(p2[word2 + '|' + 'unk'])
		else:
			return np.log(p2['unk' + '|' + 'unk'])

def smooth_training(validation_file, train_count, unicounts, bicounts):
	# pick the k/lambda value that minimizes perplexity
	unip, bip = {}, {}
	for i in np.linspace(0.01, 0.1, 10):
		# p1,p2 = add_k_probability(train_count, unicounts, bicounts, i)
		p1,p2 = interpolation_probability(train_count, unicounts, bicounts, i)
		pp1, pp2 = validation(validation_file, p1, p2, unicounts)
		unip[i] = pp1
		bip[i] = pp2
		print('lambda = {}, pp1 = {}, pp2 = {}.'.format(i, pp1, pp2))
	i1 = min(unip.keys(), key=(lambda k: unip[k]))
	i2 = min(bip.keys(), key=(lambda k: bip[k]))
	print('Using {} language model on {} validation set:'.format(sys.argv[1], sys.argv[2]))
	print('The minimun unigram perplexity is {}, lambda = {}.'.format(unip[i1], i1))
	print('The minimun bigram perplexity is {}, lambda = {}.\n'.format(bip[i2], i2))

def interpolation_smoothing(validation_file, train_count, unicounts, bicounts):
	# after training, the lambda value that minnimize perplexity of bigram models is 0.06
	p1,p2 = interpolation_probability(train_count, unicounts, bicounts, 0.06)
	pp1, pp2 = validation(validation_file, p1, p2, unicounts)
	print('pp1 = {}, pp2 = {}.'.format(pp1, pp2))

def kneser_ney_smoothing(validation_file, train_count, unicounts, bicounts):
	p1, p2 = kneser_ney_probability(train_count, unicounts, bicounts)
	pp1, pp2 = validation(validation_file, p1, p2, unicounts)
	print('Using Kneser_Ney Smoothing, pp1 = {}, pp2 = {}.'.format(pp1, pp2))

if __name__ == '__main__':
   main()
