import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.tokenize import word_tokenize

nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from pywsd.utils import lemmatize_sentence
from nltk.stem import PorterStemmer

nltk.download('wordnet')

import contractions
import string
import re

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


# fake_news = pd.read_csv('Fake.csv')
# fake_news['class'] = 'fake'
#
# real_news = pd.read_csv('True.csv')
# real_news['class'] = 'real'
#
# # print(real_news.shape)
# # print(real_news.info())
#
# combined_news = pd.concat([fake_news, real_news])
#
# combined_news['text'] = combined_news['title'] + ' ' + combined_news['text']
# # combined_news['title'] = combined_news['title'].apply(preprocessing)
# combined_news['text_no_preprocessing'] = combined_news['text']
# combined_news['text'] = combined_news['text'].apply(preprocessing)
#
# # serialize preprocessed dataframe
# combined_news.to_pickle('dataframe.pkl')

# deserialize dataframe
combined_news = pd.read_pickle('dataframe.pkl')

# shuffle
combined_news = combined_news.sample(frac=1).reset_index(drop=True)
print(combined_news.head()['text'])

# split dataset
X_train, X_test, y_train, y_test = train_test_split(combined_news['text'], combined_news['class'], test_size=0.15,
                                                    random_state=69)

tfidf = TfidfVectorizer(ngram_range=(1, 3))
tfidf_train = tfidf.fit_transform(X_train.values)
tfidf_test = tfidf.transform(X_test.values)

# Multinomial Naive Bayes
mn_naive_bayes = MultinomialNB()
mn_naive_bayes.fit(tfidf_train, y_train)
mn_naive_bayes_prediction = mn_naive_bayes.predict(tfidf_test)

print(" -- Multinomial Naive Bayes classification report -- \n")
print(classification_report(y_test, mn_naive_bayes_prediction))

print("Accuracy score: " + str(accuracy_score(y_test, mn_naive_bayes_prediction)))

# K Nearest Neighbors
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(combined_news['text_no_preprocessing'],
                                                                    combined_news['class'], test_size=0.1,
                                                                    random_state=69)
tfidf_train_knn = tfidf.fit_transform(X_train_knn.values)
tfidf_test_knn = tfidf.transform(X_test_knn.values)
k_nearest_neighbors = KNeighborsClassifier(n_neighbors=1)
k_nearest_neighbors.fit(tfidf_train_knn, y_train_knn)
k_nearest_neighbors_prediction = k_nearest_neighbors.predict(tfidf_test_knn)

print(" -- K Nearest Neighbors classification report -- \n")
print(classification_report(y_test_knn, k_nearest_neighbors_prediction))

print("Accuracy score: " + str(accuracy_score(y_test_knn, k_nearest_neighbors_prediction)))
