import re
import numpy as np
import pandas as pd

class CorpusReader():
	def __init__(self):		
		#FILE_PATH_MOVIES_PLOT='../../corpora/MovieSummaries/plot_summaries.txt'
		#FILE_PATH_MOVIES_GENRE='../../corpora/MovieSummaries/movie.metadata.tsv'

		# head -2000 plot_summaries.txt > plot_summaries_sample.txt
		# head -2000 movie.metadata.tsv > movie.metadata_sample.tsv
		FILE_PATH_MOVIES_PLOT='../../corpora/MovieSummaries/plot_summaries_sample.txt'
		FILE_PATH_MOVIES_GENRE='../../corpora/MovieSummaries/movie.metadata_sample.tsv'


		columns = ['wikipediaid', 'plot']
		movies_plot = pd.read_csv(FILE_PATH_MOVIES_PLOT, sep='\t', index_col=0, names=columns)

		columns = ['wikipediaid', 'freebaseid', 'name', 'date', 'revenue', 'runtime', 'languages', 'countries', 'genres']
		columns_use = ['wikipediaid', 'genres']
		movies_genre = pd.read_csv(FILE_PATH_MOVIES_GENRE, sep='\t', index_col=0, names=columns, usecols=columns_use)

		def extract_genre(genres):
			try:
				genre = re.findall('"([\w\s]+)"', genres)[0]
			except IndexError:
				genre = np.nan
			return genre

		movies_genre['genres'] = movies_genre['genres'].apply(extract_genre)
		movies_genre.dropna(subset=['genres'], inplace=True)

		self.movies = pd.merge(movies_genre, movies_plot, left_index=True, right_index=True, how='inner')

	def ids(self):
		return self.movies.index.tolist()

	def docs(self, ids=None):
		return self.movies.loc[ids, 'plot'].tolist()

	def categories(self, ids=None):
		return self.movies.loc[ids, 'genres'].tolist()

if __name__ == '__main__':
	reader = CorpusReader()

	print(reader.ids())

	categories = reader.categories([34961787,2780906,5283066])
	print(categories)

	docs = reader.docs([34961787,2780906,5283066])

	for doc in docs:
		print("\n\n")
		print(doc)

