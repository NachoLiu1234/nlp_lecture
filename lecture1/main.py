from bin.make_words import make_words


data_path = './data/AutoMaster_TrainSet.csv'
stop_words = './data/百度停用词表.txt'
save_path = './save/words_index.txt'


if __name__ == '__main__':
    make_words(data_path, stop_words, save_path)
