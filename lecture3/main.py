from bin.run import run
from data.data import chinese, english
from config.config import *


if __name__ == '__main__':
    run(chinese, english, unites, batch_size, lr, vocabulary_dimension)
