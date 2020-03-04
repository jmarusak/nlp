import time
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from reader import CorpusReader
from loader import CorpusLoader
from normalizer import TextNormalizer 

def identity(words):
    return words

# get data
reader = CorpusReader()
loader = CorpusLoader(reader, folds=10, shuffle=True)

# build model
model = Pipeline([
		('normalizer', TextNormalizer()),
		('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False))
	])

# evaluate model
scores = {
    'model': str(model),
    'name': str(model),
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'time': [],
}

fold = 1
for X_train, X_test, y_train, y_test in loader:
    normalizer = TextNormalizer()
    X_train = normalizer.fit_transform(X_train)

    vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)
    X_train = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    X_test = normalizer.transform(X_test)
    X_test = vectorizer.transform(X_test)

    y_pred = model.predict(X_test)
    print('Accuracy: {:0.3f}'.format(accuracy_score(y_test, y_pred)))