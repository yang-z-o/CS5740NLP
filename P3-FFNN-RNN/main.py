from rnn import main as rnn_main
from ffnn1fixed import main as ffnn_main


FLAG = 'RNN'


def main():
	if FLAG == 'RNN':
		input_size = 50
		hidden_size = 32
		number_of_epochs = 5
		minibatch_size = 16
		rnn_main(input_size, hidden_size, number_of_epochs, minibatch_size)
	elif FLAG == 'FFNN':
		hidden_dim = 32
		number_of_epochs = 5
		minibatch_size = 16
		ffnn_main(hidden_dim, number_of_epochs, minibatch_size)

if __name__ == '__main__':
	main()

