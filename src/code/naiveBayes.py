from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
import numpy as np
import csv

def main():
	file_path = 'test_validation.txt'
	for n in range(2,3):
		vectorizer = CountVectorizer(ngram_range=(1, n), token_pattern=r'\b\w+\b', min_df=1)
		with open(file_path) as file:
			X = vectorizer.fit_transform(file).toarray()
			y = np.zeros(1280)
			y[512:1024] = 1
			y[1152:1280] = 1
			print('X shape is', X.shape, 'y shape is', y.shape)
			gnb = MultinomialNB() 
			y_pred = gnb.fit(X[:1280], y[:1280]).predict(X)
			err = (y != y_pred[:1280]).sum()
			print("n = %d: err rate on validation is %d / 256, %.2f"
					% (n, err, float(err)/256))
			
			with open('prediction_nb.csv', 'w') as csvfile:
				filewriter = csv.writer(csvfile, delimiter=',')
				filewriter.writerow(['Id','Prediction'])
				for i in range(1280, 1600):
					filewriter.writerow([i-1280, int(y_pred[i])])
			

if __name__ == '__main__':
	main()
