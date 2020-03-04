import numpy as np
import tabulate

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from reader import CorpusReader
from loader import CorpusLoader
from normalizer import TextNormalizer 

def identity(words):
    return words

def create_pipeline(estimator):
    steps = [
        ('normalize', TextNormalizer()),
        ('vectorize', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
        ('classifier', estimator)
    ]   
    return Pipeline(steps)

models = []
for estimator in (LogisticRegression, MultinomialNB):
    models.append(create_pipeline(estimator()))


# get data
reader = CorpusReader()
loader = CorpusLoader(reader, folds=10, shuffle=True)

scores_table = []
scores_table_fields = ['model', 'precision', 'recall', 'accuracy', 'f1']

for model in models:
    scores = defaultdict(list) 

    fold = 1
    model_name = model.named_steps['classifier'].__class__.__name__     

    print('Model: {}'.format(model_name))
    for X_train, X_test, y_train, y_test in loader:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(' Fold: {}  Accuracy: {:0.3f}'.format(fold, accuracy_score(y_test, y_pred)))

        scores['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        scores['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        fold += 1

   
    row = [model_name]
    for scores_table_field in scores_table_fields[1:]:
        row.append(np.mean(scores[scores_table_field]))

    scores_table.append(row)

scores_table.sort(key=lambda row: row[-1], reverse=True)
print(tabulate.tabulate(scores_table, headers=scores_table_fields))