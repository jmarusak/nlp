import numpy as np

from sklearn.model_selection import KFold

from reader import CorpusReader

class CorpusLoader():
	def __init__(self, reader, folds=12, shuffle=True):
		self.reader = reader
		self.folds = KFold(n_splits=folds, shuffle=shuffle)
		self.all_ids = np.asarray(reader.ids())

	def ids(self, idx=None):
		if idx is None:
			return self.all_ids
		return self.all_ids[idx]

	def __iter__(self):
		for train_idx, test_idx in self.folds.split(self.all_ids):
			X_train = self.reader.docs(self.ids(train_idx))
			y_train = self.reader.categories(self.ids(train_idx))

			X_test = self.reader.docs(self.ids(test_idx))
			y_test = self.reader.categories(self.ids(test_idx))

			yield X_train, X_test, y_train, y_test

if __name__ == '__main__':
	reader = CorpusReader()
	loader = CorpusLoader(reader, folds=10)

	fold = 1
	for X_train, X_test, y_train, y_test in loader:
		print('Fold: {} y_test: {} y_train: {}'.format(fold, y_test, y_train))
		fold += 1
		