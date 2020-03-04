import re
import numpy as np
import pandas as pd

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

#FILE_PATH_MOVIES_PLOT='../../corpora/MovieSummaries/plot_summaries.txt'
#FILE_PATH_MOVIES_GENRE='../../corpora/MovieSummaries/movie.metadata.tsv'

# head -2000 plot_summaries.txt > plot_summaries_sample.txt
# head -2000 movie.metadata.tsv > movie.metadata_sample.tsv
FILE_PATH_MOVIES_PLOT='../../corpora/MovieSummaries/plot_summaries_sample.txt'
FILE_PATH_MOVIES_GENRE='../../corpora/MovieSummaries/movie.metadata_sample.tsv'

categories = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Action', 'Horror', 'Documentary']

class CorpusReader():
	def __init__(self):		
		columns = ['wikipediaid', 'plot']
		movies_plot = pd.read_csv(FILE_PATH_MOVIES_PLOT, sep='\t', index_col=0, names=columns)

		columns = ['wikipediaid', 'freebaseid', 'name', 'date', 'revenue', 'runtime', 'languages', 'countries', 'genres']
		columns_use = ['wikipediaid', 'genres']
		movies_genre = pd.read_csv(FILE_PATH_MOVIES_GENRE, sep='\t', index_col=0, names=columns, usecols=columns_use)

		def map_category(genres):
			most_frequent_counter = 0
			most_frequent_category = np.nan

			for category in categories:
				category_count = genres.lower().count(category.lower())
				if category_count > most_frequent_counter:
					most_frequent_category = category
					most_frequent_counter = category_count

			return most_frequent_category

		movies_genre['category'] = movies_genre['genres'].apply(map_category)
		movies_genre.dropna(subset=['category'], inplace=True)

		self.movies = pd.merge(movies_genre, movies_plot, left_index=True, right_index=True, how='inner')

	def ids(self):
		return self.movies.index.tolist()

	def docs(self, ids=None):
		plots = self.movies.loc[ids, 'plot'].tolist()
		return [self.tokenize(plot) for plot in plots]

	def docs_plain(self, ids=None):
		plots = self.movies.loc[ids, 'plot'].tolist()
		return plots

	def categories(self, ids=None):
		return self.movies.loc[ids, 'category'].tolist()

	def tokenize(self, doc):
		return [pos_tag(wordpunct_tokenize(sent)) for sent in sent_tokenize(doc)]


if __name__ == '__main__':
	reader = CorpusReader()

	print(reader.ids())

	docs = reader.docs([32069698, 1132138])

	print(reader.tokenize('Napoleon Dynamite for the full-length film'))

	for doc in docs:
		print("\n\n")
		print(doc)

