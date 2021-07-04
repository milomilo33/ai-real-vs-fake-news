import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.tokenize import word_tokenize

nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
# from pywsd.utils import lemmatize_sentence
from nltk.stem import PorterStemmer

nltk.download('wordnet')

import contractions
import string
import re
import os

print_iter = 0
def preprocessing(text):
    # convert to lowercase
    # text = str(text).lower()

    # remove links
    text = re.sub(r'http\S+', '', text)

    # expand contractions (don't -> do not, etc.)
    text = contractions.fix(text)

    # tokenize text/sentence
    word_tokens = word_tokenize(text)

    # convert to lowercase if word is not all uppercase
    # [word.lower() for word in word_tokens if not word.isupper()]

    # lemmatize, lowercase and tokenize text/sentence
    # word_tokens = lemmatize_sentence(text)

    # remove stop words and punctuation
    global print_iter
    if print_iter < 2:
        print("Before removal and stem")
        print(word_tokens)
    stop_words = stopwords.words('english')
    stop_words.extend(['re', 'edu'])
    word_tokens = [word for word in word_tokens if word not in stop_words and word not in string.punctuation]

    # lemmatize words
    # lemmatizer = WordNetLemmatizer()
    # [lemmatizer.lemmatize(word) for word in word_tokens]

    # stem words and remove stop words once more
    porter = PorterStemmer()
    word_tokens = [porter.stem(word) for word in word_tokens]
    word_tokens = [word for word in word_tokens if word not in stop_words and word not in string.punctuation]

    if print_iter < 2:
        print("After removal and stem")
        print(word_tokens)
    print_iter += 1
    print(print_iter)

    return ' '.join(word_tokens)


fake_news = pd.read_csv('Fake.csv')
# fake_news['class'] = 'fake'
fake_news['class'] = 0

real_news = pd.read_csv('True.csv')
# real_news['class'] = 'real'
real_news['class'] = 1

# print(real_news.shape)
# print(real_news.info())

combined_news = pd.concat([fake_news, real_news])

combined_news['text'] = combined_news['title'] + ' ' + combined_news['text']
# combined_news['title'] = combined_news['title'].apply(preprocessing)
combined_news['text_no_preprocessing'] = combined_news['text']
combined_news['text'] = combined_news['text'].apply(preprocessing)

# serialize preprocessed dataframe
combined_news.to_pickle('dataframe.pkl')

# # deserialize dataframe
# combined_news = pd.read_pickle('dataframe.pkl')

# # word cloud
# stop_words = stopwords.words('english')
# stop_words.extend(['re', 'edu'])
# wordcloud = WordCloud(max_words=1000,
#                       background_color='white',
#                       width=1280,
#                       height=720,
#                       stopwords=stop_words).generate(' '.join(combined_news[combined_news['class'] == 'fake'].text))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title("Fake news")
# plt.show()
#
# wordcloud = WordCloud(max_words=1000,
#                       background_color='white',
#                       width=1280,
#                       height=720,
#                       stopwords=stop_words).generate(' '.join(combined_news[combined_news['class'] == 'real'].text))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title("Real news")
# plt.show()

# shuffle
combined_news = combined_news.sample(frac=1).reset_index(drop=True)
print(combined_news.head()['text'])

# split dataset
X_train, X_test, y_train, y_test = train_test_split(combined_news['text'], combined_news['class'], test_size=0.15,
                                                    random_state=69)
print(y_train)
print(y_test)

# tfidf = TfidfVectorizer(ngram_range=(1, 3))
# tfidf_train = tfidf.fit_transform(X_train.values)
# tfidf_test = tfidf.transform(X_test.values)
#
# # -- Multinomial Naive Bayes --
# mn_naive_bayes = MultinomialNB()
# mn_naive_bayes.fit(tfidf_train, y_train)
# mn_naive_bayes_prediction = mn_naive_bayes.predict(tfidf_test)
#
# print(" -- Multinomial Naive Bayes classification report -- \n")
# print(classification_report(y_test, mn_naive_bayes_prediction))
#
# print("Accuracy score: " + str(accuracy_score(y_test, mn_naive_bayes_prediction)))
#
# # -- K Nearest Neighbors --
# X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(combined_news['text_no_preprocessing'],
#                                                                     combined_news['class'], test_size=0.1,
#                                                                     random_state=69)
# tfidf_train_knn = tfidf.fit_transform(X_train_knn.values)
# tfidf_test_knn = tfidf.transform(X_test_knn.values)
# k_nearest_neighbors = KNeighborsClassifier(n_neighbors=1)
# k_nearest_neighbors.fit(tfidf_train_knn, y_train_knn)
# k_nearest_neighbors_prediction = k_nearest_neighbors.predict(tfidf_test_knn)
#
# print(" -- K Nearest Neighbors classification report -- \n")
# print(classification_report(y_test_knn, k_nearest_neighbors_prediction))
#
# print("Accuracy score: " + str(accuracy_score(y_test_knn, k_nearest_neighbors_prediction)))

# -- LSTM --
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model

MAX_FEATURES = 1000
MAX_SEQUENCE_LENGTH = 300

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

word_index = tokenizer.word_index

X_train_lstm = sequence.pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test_lstm = sequence.pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

# prepping embedding layer
print("Prepping embedding layer...")

GLOVE_DIR = os.path.dirname(os.path.realpath(__file__))
EMBEDDING_DIM = 200

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.%dd.txt' % EMBEDDING_DIM), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

BATCH_SIZE = 128
EPOCHS = 6

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.2, min_lr=0.00001)
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(units=128, return_sequences=True, recurrent_dropout=0.25, dropout=0.25))
model.add(LSTM(units=64, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train_lstm, y_train,
                    batch_size=BATCH_SIZE,
                    validation_split=0.2,
                    epochs=EPOCHS,
                    callbacks=[reduce_lr])
# save model
model.save('lstm_model')

# # load the model back
# model = load_model('lstm_model')

lstm_prediction = model.predict(X_test_lstm)

print(" -- LSTM classification report -- \n")
print(classification_report(y_test, lstm_prediction))

print("Accuracy score: " + str(accuracy_score(y_test, lstm_prediction)))
