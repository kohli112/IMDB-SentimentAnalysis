# import required packages
import numpy as np
import os
import glob
import string
from nltk.tokenize import word_tokenize
import pickle
from keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('punkt')
nltk.download('stopwords')




if __name__ == "__main__":

    # Loading the model and required tokenizer
    model = load_model("models/20828608_NLP_model")
    tokenizer_test = pickle.load(open("models/tokenizer.pkl", 'rb'))

	# Loading test data
    test_dir = "data/aclImdb/test"
    pos_dir = test_dir + "pos/*.txt"
    neg_dir = test_dir + "neg/*.txt"
    data_test = []
    labels_test = []
    sentiments = ['pos', 'neg']
    for sent, name in enumerate(sentiments):
        for fname in glob.glob(os.path.join(test_dir, name, '*.*')):
            data_test.append((open(fname, 'r').read()).lower())
            labels_test.append(sent)

    # Shuffling the data
    np.random.seed(42)
    data_test = np.array(data_test)
    labels_test = np.array(labels_test)
    idx_test = np.random.permutation(len(data_test))
    data_test, labels_test = data_test[idx_test], labels_test[idx_test]

    # Preprocessing as done on training data
    sw_test = set(stopwords.words('english'))
    review_tkns_test = []
    for line in data_test:
        line = str(line).translate(str.maketrans('', '', string.punctuation))
        tokens_test = word_tokenize(str(line))
        words_test = [w for w in tokens_test if not w in sw_test]
        review_tkns_test.append(words_test)

    # Setting the max_len which we got from training data
    max_len = 124

    # Truncating reviews
    truncReviews_test = [' '.join(seq[:max_len]) for seq in review_tkns_test]

    # Vectorization of text using the loaded tokenizer
    sequences_test = tokenizer_test.texts_to_sequences(truncReviews_test)

    # Padding
    test_data_pad = pad_sequences(sequences_test, maxlen=max_len)

    # Evaluating the model and printing Test Accuracy
    eval = model.evaluate(test_data_pad, labels_test)
    print('Test accuracy: {0:.3g}%'.format(eval[1]*100))