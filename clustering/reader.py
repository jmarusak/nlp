import re
import numpy as np
import pandas as pd

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

FILE_PATH_MOVIES_PLOT='plot_summaries_sample.txt'

class CorpusReader():
	def __init__(self):		
		columns = ['wikipediaid', 'plot']
		self.movies = pd.read_csv(FILE_PATH_MOVIES_PLOT, sep='\t', index_col=0, names=columns)
	
	def ids(self):
		return self.movies.index.tolist()

	def docs(self, ids=None):
		if ids is None:
			ids = self.ids()
		plots = self.movies.loc[ids, 'plot'].tolist()
		return [self.tokenize(plot) for plot in plots]

	def tokenize(self, doc):
		return [pos_tag(wordpunct_tokenize(sent)) for sent in sent_tokenize(doc)]

if __name__ == '__main__':
    reader = CorpusReader()

    ids = reader.ids()
    for i in range(10):
        print(ids[i])
    
    docs = reader.docs([23890098, 31186339])
    for doc in docs:
        print()
        print(doc)