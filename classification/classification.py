import time
import json
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from reader import PickledCorpusReader
from loader import CorpusLoader
from normalizer import TextNormalizer 

def identity(words):
    return words

labels = ["books", "cinema"]

# get data
reader = PickledCorpusReader('../../corpora/Pickled_Corpus_Sample')
loader = CorpusLoader(reader, 5, shuffle=True, categories=labels)

# build model
model = Pipeline([
		('normalizer', TextNormalizer()),
		('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
		('form', LogisticRegression())
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

for X_train, X_test, y_train, y_test in loader:
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    scores['time'].append(time.time() - start)
    scores['accuracy'].append(accuracy_score(y_test, y_pred))
    scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
    scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
    scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))

    print('Time: {:3.3f} Accuracy: {:0.3f}'.format(time.time() - start, accuracy_score(y_test, y_pred)))
    print(list(X_train))
    print(y_pred)

print('Final Accuracy: {:0.3f}'.format(np.mean(scores['accuracy'])))