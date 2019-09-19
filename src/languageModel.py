import sys
import os
import numpy as np

def main():
	if len(sys.argv) == 2:
		# train mode
		print('todo')
	elif len(sys.argv) == 3:
		# validation mode - training smoothing method
		training_file_path = 'DATASET/train/' + sys.argv[1] + '.txt'
		validation_file_path = 'DATASET/validation/' + sys.argv[2] + '.txt'
		print('Using {} model on {} validation set'.format(sys.argv[1], sys.argv[2]))
		with open(training_file_path) as training_file:
			train_count, unicounts, bicounts = word_counter(training_file)
		with open(validation_file_path) as validation_file:
			# smoothing_parameter_training(validation_file, train_count, unicounts, bicounts)
			# interpolation_smoothing(validation_file, train_count, unicounts, bicounts)
			kneser_ney_smoothing(validation_file, train_count, unicounts, bicounts)
	elif len(sys.argv) == 4:
		# validation mode - training accuracy
		class1, class2, y = map(int, sys.argv[1:4])
		filename = ['deceptive', 'truthful']
		training_file_path1 = 'DATASET/train/' + filename[class1] + '.txt'
		training_file_path2 = 'DATASET/train/' + filename[class2] + '.txt'
		# test_file_path = 'DATASET/test/' + sys.argv[3] + '.txt'
		# test_file_path = sys.argv[3]
		test_file_path = 'DATASET/validation/' + filename[y] + '.txt'
		test(training_file_path1, training_file_path2, test_file_path, class1, class2, y)
	else:
		print('Please input correct files.')

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
	validation_count, line_count = 0, 0
	general_pp1, general_pp2 = 0, 0
	for line in validation_file:
		line_count += 1
		pp1,pp2 = 0,0
		words = line.strip().split(' ')
		validation_count += 1
		pp1 += unigram_test(words[0].lower(), unicounts, p1)
		for i in range(1, len(words)):
			validation_count += 1
			pp1 += unigram_test(words[i].lower(), unicounts, p1)
			pp2 += bigram_test(words[i - 1].lower(), words[i].lower(), unicounts, p2)
		general_pp1 += pp1 
		general_pp2 += pp2
		pp1 = np.exp(- pp1 / validation_count)
		pp2 = np.exp(- pp2 / validation_count)
		print('For line {}, pp1 = {}, pp2 = {}.'.format(line_count, pp1, pp2))
	return np.exp(- general_pp1 / validation_count), np.exp(- general_pp2 / validation_count)

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

def smoothing_parameter_training(validation_file, train_count, unicounts, bicounts):
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
	print('In general, pp1 = {}, pp2 = {}.'.format(pp1, pp2))

def test(training_file_path1, training_file_path2, test_file_path, class1, class2, y):
	with open(training_file_path1) as training_file1:
			train_count1, unicounts1, bicounts1 = word_counter(training_file1)
	with open(training_file_path2) as training_file2:
			train_count2, unicounts2, bicounts2 = word_counter(training_file2)
	with open(test_file_path) as test_file:
		p11, p12 = kneser_ney_probability(train_count1, unicounts1, bicounts1)
		p21, p22 = kneser_ney_probability(train_count2, unicounts2, bicounts2)
		prediction(test_file, class1, class2, y, p11, p12, p21, p22, unicounts1, unicounts2)

def prediction(test_file, class1, class2, y, p11, p12, p21, p22, unicounts1, unicounts2):
	line_count, test_word_count = 0, 0
	accuracy_count = {}
	# print('Id, Prediction')
	for line in test_file:
		pp11, pp12, pp21, pp22 = 0, 0, 0, 0
		words = line.strip().split(' ')
		test_word_count += 1
		pp11 += unigram_test(words[0].lower(), unicounts1, p11)
		pp21 += unigram_test(words[0].lower(), unicounts2, p21)
		for i in range(1, len(words)):
			test_word_count += 1
			pp11 += unigram_test(words[i].lower(), unicounts1, p11)
			pp21 += unigram_test(words[i].lower(), unicounts2, p21)
			pp12 += bigram_test(words[i - 1].lower(), words[i].lower(), unicounts1, p12)
			pp22 += bigram_test(words[i - 1].lower(), words[i].lower(), unicounts2, p22)
		pp11 = np.exp(- pp11 / test_word_count)
		pp21 = np.exp(- pp21 / test_word_count)
		for k in np.linspace(0, 2, 21):
			if k * pp11 + (1 - k) * pp12 <= k * pp21 + (1 - k) * pp22:
				pred = class1
			else: 
				pred = class2
			# print('For line {}: \npp11 = {:.2f}, pp21 = {:.2f}\npp12 = {:.2f}, pp22 = {:.2f}'.format(line_count, pp11, pp21, pp12, pp22))
			# print('{}, {}'.format(line_count, pred))
			if pred == y:
				accuracy_count[k] = accuracy_count.get(k, 0) + 1
		line_count += 1
	print('class1:{}, class2:{}, y:{}\nk, accuracy'.format(class1, class2, y))
	for k in np.linspace(0, 2, 21):
		print('{:.1f}, {}'.format(k, float(accuracy_count[k]) / line_count))

if __name__ == '__main__':
   main()
