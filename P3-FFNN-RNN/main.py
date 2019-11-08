from rnn import main as rnn_main
from ffnn1fixed import main as ffnn_main


FLAG = 'RNN'


def main():
	if FLAG == 'RNN':
		input_size = 300
		hidden_size = 32
		rnn_main(input_size, hidden_size)
	elif FLAG == 'FFNN':
		hidden_dim = 32
		number_of_epochs = 10
		ffnn_main(hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)



if __name__ == '__main__':
	main()

