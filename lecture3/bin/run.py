import tensorflow as tf
import numpy as np
import jieba
import time

print(tf.__version__)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocabulary_number, vocabulary_dimension, vocabulary_matrix, gru_unites, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.gru_unites = gru_unites
        self.embedding = tf.keras.layers.Embedding(vocabulary_number, vocabulary_dimension,
                                                   #                                                weights=vocabulary_matrix, trainable=False
                                                   )
        self.gru = tf.keras.layers.GRU(gru_unites, return_sequences=True, return_state=True)

    def call(self, inp, hidden):
        x = self.embedding(inp)
        x, hidden = self.gru(x, initial_state=hidden)
        return x, hidden

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.gru_unites))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        self.w1 = tf.keras.layers.Dense(units)
        self.w2 = tf.keras.layers.Dense(units)
        self.v = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output):
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)
        score = self.v(tf.nn.tanh(self.w1(enc_output) + self.w2(hidden_with_time_axis)))
        attn_dist = tf.nn.softmax(score, axis=1)
        #         attn_dist = tf.expand_dims(attn_dist, axis=2)
        context_vector = attn_dist * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attn_dist


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocabulary_number, vocabulary_dimension, vocabulary_matrix, unites, batch_size):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocabulary_number, vocabulary_dimension,
                                                   weights=[vocabulary_matrix], trainable=False)
        self.gru = tf.keras.layers.GRU(unites, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocabulary_number, activation=tf.keras.activations.softmax)

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


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def max_length(tensor):
    return max(len(t) for t in tensor)


def run(chinese, english, unites, batch_size, lr, vocabulary_dimension):

    inp_lang = LanguageIndex(chinese)
    targ_lang = LanguageIndex(english)

    chinese = [[inp_lang.word2idx[w] for w in s] for s in chinese]
    english = [[targ_lang.word2idx[w] for w in s] for s in english]

    max_length_inp, max_length_tar = max_length(chinese), max_length(english)

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(chinese,
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(chinese,
                                                                  maxlen=max_length_tar,
                                                                  padding='post')

    inp_vocabulary_matrix = np.random.rand(len(inp_lang.word2idx), vocabulary_dimension)
    targ_vocabulary_matrix = np.random.rand(len(targ_lang.word2idx), vocabulary_dimension)
    encoder = Encoder(len(inp_lang.word2idx), vocabulary_dimension, inp_vocabulary_matrix, unites, batch_size)
    decoder = Decoder(len(targ_lang.word2idx), vocabulary_dimension, targ_vocabulary_matrix, unites, batch_size)
    optimizer = tf.keras.optimizers.Adam(lr)

    BUFFER_SIZE = len(input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    EPOCHS = 1
    t1 = time.time()
    for epoch in range(EPOCHS):
        hidden = encoder.initialize_hidden_state()
        total_loss = 0

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

        total_loss += loss / targ.shape[1]
        variables = encoder.variables + decoder.variables

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    print(f'finished training, time cost: {time.time() - t1}')
