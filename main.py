import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

import contractions
import string
import re


def preprocessing(text):
    # convert to lowercase
    text = str(text).lower()

    # expand contractions (don't -> do not, etc.)
    text = contractions.fix(text)

    # remove links
    text = re.sub(r'http\S+', '', text)

    # tokenize text/sentence
    word_tokens = word_tokenize(text)

    # remove stop words and punctuation
    stop_words = stopwords.words('english')
    word_tokens = [word for word in word_tokens if word not in stop_words and word not in string.punctuation]

    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    [lemmatizer.lemmatize(word) for word in word_tokens]

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
# combined_news['text'] = combined_news['text'].apply(preprocessing)
#
# # serialize preprocessed dataframe
# combined_news.to_pickle('dataframe.pkl')

# deserialize dataframe
combined_news = pd.read_pickle('dataframe.pkl')

# split dataset
X_train, X_test, y_train, y_test = train_test_split(combined_news['text'], combined_news['class'], test_size=0.15, random_state=69)

tfidf = TfidfVectorizer(ngram_range=(1, 2))
tfidf_train = tfidf.fit_transform(X_train.values)
tfidf_test = tfidf.transform(X_test.values)
