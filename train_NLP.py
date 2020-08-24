# import required packages
import pandas as pd
import numpy as np
import pickle
import os
import string
import glob
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Embedding, Dropout, Flatten, LSTM, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')




if __name__ == "__main__":

	# Loading and making the training data
    train_dir = "data/aclImdb/train"
    pos_dir = train_dir + "pos/*.txt"
    neg_dir = train_dir + "neg/*.txt"
    data = []
    labels = []
    sentiments = ['pos', 'neg']
    for sent, name in enumerate(sentiments):
        for fname in glob.glob(os.path.join(train_dir, name, '*.*')):
            data.append((open(fname, 'r').read()).lower())
            labels.append(sent)

    # Randomly shuffling the data
    np.random.seed(42)
    data = np.array(data)
    labels = np.array(labels)
    idx = np.random.permutation(len(data))
    data, labels = data[idx], labels[idx]

    # Preprocessing - removing punctuations and stop words.
    review_tkns = []
    sw = set(stopwords.words('english'))
    for line in data:
        line = str(line).translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(str(line))
        words = [w for w in tokens if not w in sw]
        review_tkns.append(words)

    # Getting the mean length of review
    mean = np.mean([len(review) for review in review_tkns])
    print("Mean Review Length: ", int(mean))
    max_len = int(mean)

    # Truncating the reviews to get all reviews of same length equal to mean length
    truncReviews = [' '.join(seq[:max_len]) for seq in review_tkns]

    # Vectorization
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(truncReviews)
    pickle.dump(tokenizer_obj, open("models/tokenizer.pkl", "wb"))
    sequences = tokenizer_obj.texts_to_sequences(truncReviews)

    vocab_size = len(tokenizer_obj.word_index)

    # Creating weights for Embedding Layer using Word2Vec model
    w2v = Word2Vec(review_tkns, min_count=3, size=32)
    emnSize = w2v.vector_size
    length = vocab_size + 1
    mat = np.random.randn(length, emnSize)
    for i, j in tokenizer_obj.word_index.items():
        try:
            emVec = w2v[i]
        except KeyError:
            emVec = None
        if emVec is not None:
            mat[j] = emVec
    del w2v

    # Padding the sequences to get all the sequences of same length, pads shorter sequences and makes them equal to mean length
    data_pad = pad_sequences(sequences, maxlen=max_len, padding='post')


    # Creating and training the model
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=32, input_length=max_len, weights=[mat]))
    model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Getting the Training Accuracy
    acc = model.fit(data_pad, labels, epochs=5, batch_size=64)
    print("Training Accuracy: {0:.3g}%".format(acc.history.get('accuracy')[-1]*100))

	# Saving the model
    model.save("models/20828608_NLP_model")