import re
import numpy as np
import pandas as pd

FILE_PATH_MOVIES_PLOT='../../corpora/MovieSummaries/plot_summaries.txt'
FILE_PATH_MOVIES_GENRE='../../corpora/MovieSummaries/movie.metadata.tsv'

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

movies = pd.merge(movies_genre, movies_plot, left_index=True, right_index=True, how='inner')

print(movies.head(20))