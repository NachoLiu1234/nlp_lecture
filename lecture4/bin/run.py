import tensorflow as tf
import numpy as np
from pyltp import SentenceSplitter,NamedEntityRecognizer,Postagger,Parser,Segmentor
import pandas as pd
from gensim.models.word2vec import LineSentence, Word2Vec
import time, shutil, os
print(tf.__version__)




def train(vocabulary_dimension, unites, batch_size):

    df = pd.read_csv('./AutoMaster_TrainSet.csv')

    cws_model = "./ltp_data_v3.4.0/cws.model"

    def get_word_list(sentence=None,sentences=None,model=cws_model):
        #得到分词
        segmentor = Segmentor()
        segmentor.load(model)
        if sentences is not None:
            for i, s in enumerate(sentences):
                sentences[i] = list(segmentor.segment(s))
            return sentences
        else:
            word_list = list(segmentor.segment(sentence))
        segmentor.release()
        return word_list

    inp = list(df.Question)
    outp = list(df.Report)
    assert len(inp) == len(outp)

    # 去除随时联系, 查阅资料等无意义回答
    drop_index_set = set()
    for index, (i, o) in enumerate(zip(inp, outp)):
        if not type(i) == type(o) == str or len(i) < 8 or len(o) < 8:
            drop_index_set.add(index)
    inp = [el for i, el in enumerate(inp) if i not in drop_index_set]
    outp = [el for i, el in enumerate(outp) if i not in drop_index_set]

    inp = get_word_list(sentences=inp)
    outp = get_word_list(sentences=outp)

    def get_length(tensor):
        length = sorted([len(t) for t in tensor])
        return length[int(len(length) * 0.95)]

    inp_length = get_length(inp)
    outp_length = get_length(outp)

    with open('../gensim_input.txt', 'w') as f:
        for s in inp:
            #         s = s[:inp_length]
            #         s = s + ['<pad>'] * (inp_length - len(s))
            s = ['<start>'] + s + ['<end>']
            f.write(' '.join(s) + '\n')
        for s in outp:
            #         s = s[:outp_length]
            #         s = s + ['<pad>'] * (outp_length - len(s))
            s = ['<start>'] + s + ['<end>']
            f.write(' '.join(s) + '\n')

    model = Word2Vec(corpus_file='../gensim_input.txt', size=vocabulary_dimension, min_count=5, workers=8, sg=0, negative=5, iter=5)

    class Encoder(tf.keras.Model):
        def __init__(self, vocabulary_number, vocabulary_dimension, vocabulary_matrix, gru_unites, batch_size):
            super(Encoder, self).__init__()
            self.batch_size = batch_size
            self.gru_unites = gru_unites
            self.embedding = tf.keras.layers.Embedding(vocabulary_number, vocabulary_dimension,
                                                       weights=[vocabulary_matrix], trainable=False
                                                       )
            self.gru = tf.keras.layers.GRU(gru_unites, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        def call(self, inp, hidden):
            x = self.embedding(inp)
            x, hidden = self.gru(x, initial_state=hidden)
            return x, hidden

        def initialize_hidden_state(self):
            return tf.zeros((self.batch_size, self.gru_unites))

    class Decoder(tf.keras.Model):
        def __init__(self, vocabulary_number, vocabulary_dimension, vocabulary_matrix, unites, batch_size):
            super(Decoder, self).__init__()
            self.embedding = tf.keras.layers.Embedding(vocabulary_number, vocabulary_dimension,
                                                       weights=[vocabulary_matrix], trainable=False
                                                       )
            self.gru = tf.keras.layers.GRU(unites, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
            self.fc = tf.keras.layers.Dense(vocabulary_number)

            # BahdanauAttention
            self.w1 = tf.keras.layers.Dense(unites)
            self.w2 = tf.keras.layers.Dense(unites)
            self.v = tf.keras.layers.Dense(1)

        def call(self, x, dec_hidden, enc_output):
            # dec_hidden shape == (batch_size, hidden size)
            # enc_output shape == (batch_size, max_length, hidden_size)

            hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)
            score = tf.nn.tanh(self.w1(enc_output) + self.w2(hidden_with_time_axis))
            attention_weights = tf.nn.softmax(self.v(score), axis=1)
            context_vector = attention_weights * enc_output
            context_vector = tf.reduce_sum(context_vector, axis=1)

            #         print(x.shape)
            #         print(x)
            #         e = tf.keras.layers.Embedding(len(targ_lang.word2idx), vocabulary_dimension)
            #         print(e(x))
            #         self.embedding(1)
            x = self.embedding(x)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

            output, state = self.gru(x)

            # output shape == (batch_size * max_length, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))

            # output shape == (batch_size * max_length, vocab)
            output = self.fc(output)

            return output, state, attention_weights

        def initialize_hidden_state(self):
            return tf.zeros((self.batch_sz, self.dec_units))


    class LanguageIndex():
        def __init__(self, lang, model):
            self.lang = lang
            self.word2idx = {}
            self.idx2word = {}
            self.vocab = set(model.wv.vocab.keys())

            self.create_index()

        def create_index(self):
            self.word2idx['<pad>'] = 0
            self.word2idx['<start>'] = 1
            self.word2idx['<end>'] = 2
            self.word2idx['<unknown>'] = 3
            for index, word in enumerate(self.vocab):
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)

            for word, index in self.word2idx.items():
                self.idx2word[index] = word

    inp_lang = LanguageIndex(inp, model)
    targ_lang = LanguageIndex(outp, model)

    inp = [[inp_lang.word2idx[w] if w in inp_lang.word2idx else inp_lang.word2idx['<unknown>'] for w in s] for s in inp]
    outp = [[targ_lang.word2idx[w] if w in targ_lang.word2idx else targ_lang.word2idx['<unknown>'] for w in s] + [targ_lang.word2idx['<end>']] for s in outp]

    pad_index = targ_lang.word2idx['<pad>']
    def loss_function(real, pred):  #(128,) (128, 13706)
        mask = 1 - np.equal(real, pad_index)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    max_length_inp, max_length_tar = get_length(inp), get_length(outp)

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(inp,
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(outp,
                                                                  maxlen=max_length_tar,
                                                                  padding='post')

    input_tensor_backup = input_tensor
    target_tensor_backup = target_tensor

    input_tensor = input_tensor_backup
    target_tensor = target_tensor_backup

    valid_input_tensor = input_tensor[-1000:]
    valid_target_tensor = target_tensor[-1000:]
    input_tensor = input_tensor[:-1000]
    target_tensor = target_tensor[:-1000]

    BUFFER_SIZE = len(input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    BUFFER_SIZE = len(valid_input_tensor)
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_input_tensor, valid_target_tensor)).shuffle(BUFFER_SIZE)
    # valid_dataset = dataset.batch(1, drop_remainder=True)


    w = '<pad>'
    model.wv[w] = np.zeros((100,))
    w = '<unknown>'
    model.wv[w] = np.zeros((100,))
    inp_vocabulary_matrix = np.random.rand(len(inp_lang.word2idx), vocabulary_dimension)
    targ_vocabulary_matrix = np.random.rand(len(targ_lang.word2idx), vocabulary_dimension)
    for i, w in inp_lang.idx2word.items():
        #     if w not in model.wv:
        #         continue
        inp_vocabulary_matrix[i] = model.wv[w]
    for i, w in targ_lang.idx2word.items():
        #     if w not in model.wv:
        #         continue
        targ_vocabulary_matrix[i] = model.wv[w]

    w = '<unknown>'
    model.wv[w] = np.average(inp_vocabulary_matrix, axis=0)
    model.wv[w]
    inp_vocabulary_matrix[inp_lang.word2idx[w]] = model.wv[w]
    targ_vocabulary_matrix[targ_lang.word2idx[w]] = model.wv[w]
    targ_vocabulary_matrix[targ_lang.word2idx['<unknown>']]

    encoder = Encoder(len(inp_lang.word2idx), vocabulary_dimension, inp_vocabulary_matrix, unites, batch_size)
    decoder = Decoder(len(targ_lang.word2idx), vocabulary_dimension, targ_vocabulary_matrix, unites, batch_size)

    optimizer = tf.keras.optimizers.Adam(0.01)


    checkpoint_dir = '/content/drive/My Drive/Colab Notebooks/checkpoint'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    EPOCHS = 10
    loss_plot = []
    t1 = time.time()
    for epoch in range(EPOCHS):
        t2 = time.time()
        total_loss = 0
        hidden = encoder.initialize_hidden_state()

        for (batch, (inp, targ)) in enumerate(dataset):

            loss = 0
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)

                dec_hidden = enc_hidden

                dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * batch_size, 1)

                for t in range(1, targ.shape[1]):
                    predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)
                    del attention_weights

                    loss += loss_function(targ[:, t], predictions)

                    dec_input = tf.expand_dims(targ[:, t], 1)

            total_loss += (loss / targ.shape[1])
            variables = encoder.variables + decoder.variables

            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

        checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'loss: {total_loss/(batch + 1)}, time cost: {time.time() - t2}')
        loss_plot.append(total_loss/(batch + 1))
    print(f'finished training, time cost: {time.time() - t1}')

    return valid_input_tensor, valid_target_tensor, unites, inp_lang, targ_lang, encoder, decoder


def predict(valid_input_tensor, valid_target_tensor, unites, inp_lang, targ_lang, encoder, decoder):

    def id2word(sentence, lang):
        if type(sentence) != list:
            sentence = list(sentence.numpy())

        sentence = [lang.idx2word[el] for el in sentence if el != lang.word2idx['<pad>']]
        return ''.join(sentence)
    BUFFER_SIZE = len(valid_input_tensor)
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_input_tensor, valid_target_tensor))

    hidden = tf.zeros((1, unites))

    for (batch, (inp, targ)) in enumerate(valid_dataset):
        print('input:', id2word(inp, inp_lang))
        print('targ:', id2word(targ, targ_lang))
        inp = tf.expand_dims(inp, 0)
        enc_output, enc_hidden = encoder(inp, hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 1)


        output = []
        for _ in range(100):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)
            predictions = (tf.keras.layers.Softmax()(tf.reshape(predictions, (-1,))))
            predictions = tf.argmax(predictions)


            predictions = int(predictions.numpy())
            output.append(predictions)
            if predictions == targ_lang.word2idx['<end>']:
                break


            dec_input = tf.expand_dims([predictions], 1)
        print('predict:', id2word(output, targ_lang))

        print('#' * 100)













