import json
import random

def fetch_data():
	with open('training1600.json') as training_f:
		training = json.load(training_f)
	with open('validation160.json') as valid_f:
		validation = json.load(valid_f)
	# If needed you can shrink the training and validation data to speed up somethings but this isn't always safe to do by setting k < 16000
	# k = 160
	# random.shuffle(training)
	# random.shuffle(validation)
	# training, validation = training[:k], validation[:k]

	# # with open('training1600.json', 'w') as outfile:
	# # 	json.dump(training, outfile)
	# with open('validation160.json', 'w') as outfile2:
	# 	json.dump(validation, outfile2)
	
	tra = []
	val = []
	for elt in training:
		tra.append((elt["text"].split(),int(elt["stars"]-1)))
	for elt in validation:
		val.append((elt["text"].split(),int(elt["stars"]-1)))
	return tra, val