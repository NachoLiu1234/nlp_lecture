from bin.run import train, predict
from config.config import *


if __name__ == '__main__':
    valid_input_tensor, valid_target_tensor, unites, inp_lang, targ_lang, encoder, decoder = train(vocabulary_dimension, unites, batch_size)

    predict(valid_input_tensor, valid_target_tensor, unites, inp_lang, targ_lang, encoder, decoder)