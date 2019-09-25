import sys
import os
import numpy as np
import csv

def main():
	if len(sys.argv) == 5:
		class1, class2, y1, y2 = map(int, sys.argv[1:5])
		filename = ['deceptive', 'truthful']
		training_file_path1 = 'DATASET/train/' + filename[class1] + '.txt'
		training_file_path2 = 'DATASET/train/' + filename[class2] + '.txt'
		validation_file_path1 = 'DATASET/validation/' + filename[y1] + '.txt'
		validation_file_path2 = 'DATASET/validation/' + filename[y2] + '.txt'
		k_lambda_training(training_file_path1, training_file_path2, validation_file_path1, validation_file_path2, class1, class2, y1, y2)
		# kn_training(training_file_path1, training_file_path2, validation_file_path1, validation_file_path2, class1, class2, y1, y2)
	else:
		training_file_path1 = 'DATASET/train/truthful.txt'
		training_file_path2 = 'DATASET/train/deceptive.txt'
		test_file_path = 'DATASET/test/test.txt'
		test(training_file_path1, training_file_path2, test_file_path)


def k_lambda_training(training_file_path1, training_file_path2, validation_file_path1, validation_file_path2, class1, class2, y1, y2):
	unk_range = np.linspace(0.001, 0.004, 4)
	acc_unk = {}
	for unk_rate in unk_range:
		k_range = np.linspace(0.1, 1, 10)
		#lambd_range = np.linspace(0, 1, 11)
		r_range = np.linspace(0.87, 0.87, 1)
		acc_k, acc_lambd, acc_r = {}, {}, {}	
		for k in k_range:
		#for lambd in lambd_range:
			# train language models on each training file
			with open(training_file_path1) as training_file1:
				train_count1, unicounts1, bicounts1 = word_counter(training_file1, unk_rate)
				p11, p12 = add_k_probability(train_count1, unicounts1, bicounts1, k)
				#p11, p12 = interpolation_probability(train_count1, unicounts1, bicounts1, lambd)
			with open(training_file_path2) as training_file2:
				train_count2, unicounts2, bicounts2 = word_counter(training_file2, unk_rate)
				p21, p22 = add_k_probability(train_count2, unicounts2, bicounts2, k)
				#p21, p22 = interpolation_probability(train_count2, unicounts2, bicounts2, lambd)
			# use two bigrams on each validation file
			with open(validation_file_path1) as validation_file1:
				acc1 = accuracy(validation_file1, class1, class2, y1, p11, p12, p21, p22, unicounts1, unicounts2, r_range)
			with open(validation_file_path2) as validation_file2:
				acc2 = accuracy(validation_file2, class1, class2, y2, p11, p12, p21, p22, unicounts1, unicounts2, r_range)
			# for each rate r, compute its accuracy
			for r in r_range:
				acc_r[r] = acc1.get(r, 0) * 0.5 + acc2.get(r, 0) * 0.5
			r_max = max(acc_r.keys(), key=(lambda k: acc_r[k]))
			print('k = {:.2f}, the rate that maximizes accuracy is {:.2f}: {:.2f}%'.format(k, r_max, acc_r[r_max] * 100))
			# print('lambda = {:.2f}, the rate that maximizes accuracy is {:.2f}: {:.2f}%'.format(lambd, r_max, acc_r[r_max] * 100))
			acc_k[k] = acc_r[r_max] * 100
			#acc_lambd[lambd] = acc_r[r_max] * 100
		k_max = max(acc_k.keys(), key=(lambda k: acc_k[k]))
		print('The k that maximizes accuracy is {:.2f}: {:.2f}%'.format(k_max, acc_k[k_max]))
		#lambd_max = max(acc_lambd.keys(), key=(lambda k: acc_lambd[k]))
		#print('The lambda that maximizes accuracy is {:.2f}: {:.2f}%'.format(lambd_max, acc_lambd[lambd_max]))
		acc_unk[unk_rate] = acc_k[k_max]
	unk_max = max(acc_unk.keys(), key=(lambda k: acc_unk[k]))
	print('The rate of unkown words that maximizes accuracy is {:2f}: {:2f}%'.format(unk_max, acc_unk[unk_max]))

def kn_training(training_file_path1, training_file_path2, validation_file_path1, validation_file_path2, class1, class2, y1, y2):
	r_range = np.linspace(0.8, 1.0, 21)
	acc_r = {}
	with open(training_file_path1) as training_file1:
		train_count1, unicounts1, bicounts1 = word_counter(training_file1)
		p11, p12 = kneser_ney_probability(train_count1, unicounts1, bicounts1)
	with open(training_file_path2) as training_file2:
		train_count2, unicounts2, bicounts2 = word_counter(training_file2)
		p21, p22 = kneser_ney_probability(train_count2, unicounts2, bicounts2)
	# use two bigrams on each validation file
	with open(validation_file_path1) as validation_file1:
		acc1 = accuracy(validation_file1, class1, class2, y1, p11, p12, p21, p22, unicounts1, unicounts2, r_range)
	with open(validation_file_path2) as validation_file2:
		acc2 = accuracy(validation_file2, class1, class2, y2, p11, p12, p21, p22, unicounts1, unicounts2, r_range)
	# for each rate r, compute its accuracy
	for r in r_range:
		acc_r[r] = acc1.get(r, 0) * 0.5 + acc2.get(r, 0) * 0.5
		print('r = {:.2f}, acc_r = {:.2f}%'.format(r, acc_r[r] * 100))
	r_max = max(acc_r.keys(), key=(lambda k: acc_r[k]))
	print('The r that maximizes accuracy is {:.2f}: {:.2f}%'.format(r_max, acc_r[r_max] * 100))

def test(training_file_path1, training_file_path2, test_file_path):
	unk_rate = 0.0001
	k = 0.8
	r = 0.85
	# train language models on each training file
	with open(training_file_path1) as training_file1:
		train_count1, unicounts1, bicounts1 = word_counter(training_file1, unk_rate)
		p11, p12 = add_k_probability(train_count1, unicounts1, bicounts1, k)
	with open(training_file_path2) as training_file2:
		train_count2, unicounts2, bicounts2 = word_counter(training_file2, unk_rate)
		p21, p22 = add_k_probability(train_count2, unicounts2, bicounts2, k)
	# use two bigrams on test file
	with open(test_file_path) as test_file:
		prediction(test_file, p11, p12, p21, p22, unicounts1, unicounts2, r)
		
def prediction(test_file, p11, p12, p21, p22, unicounts1, unicounts2, r):
	line_count = 0
	accuracy_count, acc = {}, {}
	with open('prediction.csv', 'w') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',')
		filewriter.writerow(['Id','Prediction'])
		for line in test_file:
			pp11, pp21, pp12, pp22 = calculate_perplexity(line, p11, p12, p21, p22, unicounts1, unicounts2)
			# only use perplexity of bigram models after test
			if r * pp12 <= pp22:
				pred = 0
			else: 
				pred = 1
			filewriter.writerow([line_count, pred])
			line_count += 1

def word_counter(file, unk_rate):
	train_count = 0
	bag_of_words, unicounts, bicounts = {}, {}, {}
	unk = set()
	# first loop, pick unk
	for line in file:
		for word in line.strip().split(' '):
			train_count += 1
			bag_of_words[word.lower()] = bag_of_words.get(word.lower(), 0) + 1
	words = [(word,cnt) for word,cnt in bag_of_words.items()]
	sorted_words = sorted(words, key=lambda x: x[1])
	for i in range(int(round(unk_rate * len(bag_of_words)))):
		unk.add(sorted_words[i][0])
	file.seek(0)
	# second loop, count include unk
	for line in file:
		nwords = line.strip().split(' ')
		unigram_unk(nwords[0].lower(), unk, unicounts)
		for i in range(1, len(nwords)):
			unigram_unk(nwords[i].lower(), unk, unicounts)
			bigram_unk(nwords[i - 1].lower(), nwords[i].lower(), unk, unicounts, bicounts)
	return train_count, unicounts, bicounts

def unigram_unk(word, unk, unicounts):
	if word in unk:
		unicounts['unk'] = unicounts.get('unk', 0) + 1
	else:
		unicounts[word] = unicounts.get(word, 0) + 1
		
def bigram_unk(word1, word2, unk, unicounts, bicounts):
	if word1 in unk:
		if word2 in unk:
			bicountspp('unk', 'unk', bicounts)
		else:
			bicountspp('unk', word2, bicounts)
	else:
		if word2 in unk:
			bicountspp(word1, 'unk', bicounts)
		else:
			bicountspp(word1, word2, bicounts)
			
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

def accuracy(test_file, class1, class2, y, p11, p12, p21, p22, unicounts1, unicounts2, r_range):
	line_count = 0
	accuracy_count, acc = {}, {}
	for line in test_file:
		pp11, pp21, pp12, pp22 = calculate_perplexity(line, p11, p12, p21, p22, unicounts1, unicounts2)
		for r in r_range:
			# only use perplexity of bigram models after test
			if r * pp12 <= pp22:
				pred = class1
			else: 
				pred = class2
			if pred == y:
				accuracy_count[r] = accuracy_count.get(r, 0) + 1
		line_count += 1
	for r in r_range:
		acc[r] = float(accuracy_count.get(r, 0)) / line_count
	return acc

def calculate_perplexity(line, p11, p12, p21, p22, unicounts1, unicounts2):
	pp11, pp12, pp21, pp22 = 0, 0, 0, 0
	test_word_count = 0
	words = line.strip().split(' ')
	test_word_count += 1
	pp11 += unigram_test(words[0].lower(), unicounts1, p11)
	pp21 += unigram_test(words[0].lower(), unicounts2, p21)
	for i in range(1, len(words)):
		if words[i] != '':
			test_word_count += 1
			pp11 += unigram_test(words[i].lower(), unicounts1, p11)
			pp21 += unigram_test(words[i].lower(), unicounts2, p21)
			pp12 += bigram_test(words[i - 1].lower(), words[i].lower(), unicounts1, p12)
			pp22 += bigram_test(words[i - 1].lower(), words[i].lower(), unicounts2, p22)
		else:
			i += 1
	pp11 = np.exp(- pp11 / test_word_count)
	pp21 = np.exp(- pp21 / test_word_count)
	pp12 = np.exp(- pp12 / test_word_count)
	pp22 = np.exp(- pp22 / test_word_count)
	return pp11, pp21, pp12, pp22

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

if __name__ == '__main__':
	main()
