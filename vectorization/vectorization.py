#!/usr/bin/env python3

import os
import nltk
import pickle
import gensim
import itertools
import unicodedata

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

DOC_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.json'
PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'

class PickledCorpusReader(CategorizedCorpusReader, CorpusReader):

    def __init__(self, root, fileids=PKL_PATTERN, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining arguments
        are passed to the ``CorpusReader`` constructor.
        """
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def _resolve(self, fileids, categories):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. This primarily bubbles up to
        the high level ``docs`` method, but is implemented here similar to
        the nltk ``CategorizedPlaintextCorpusReader``.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the document loaded from a pickled object for every file in
        the corpus. Similar to the BaleenCorpusReader, this uses a generator
        to acheive memory safe iteration.
        """
        # Resolve the fileids and the categories
        fileids = self._resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def paras(self, fileids=None, categories=None):
        """
        Returns a generator of paragraphs where each paragraph is a list of
        sentences, which is in turn a list of (token, tag) tuples.
        """
        for doc in self.docs(fileids, categories):
            for paragraph in doc:
                yield paragraph

    def sents(self, fileids=None, categories=None):
        """
        Returns a generator of sentences where each sentence is a list of
        (token, tag) tuples.
        """
        for paragraph in self.paras(fileids, categories):
            for sentence in paragraph:
                yield sentence

    def words(self, fileids=None, categories=None):
        """
        Returns a generator of (token, tag) tuples.
        """
        for sentence in self.sents(fileids, categories):
            for token in sentence:
                yield token


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english', normalizer_type='lemma'):
        self.normalizer_type = normalizer_type
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmatizer = nltk.stem.SnowballStemmer(language)

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        if self.normalizer_type == 'lemma':
            return [
                self.lemmatize(token, tag).lower()
                for paragraph in document
                for sentence in paragraph
                for (token, tag) in sentence
                if not self.is_punct(token) and not self.is_stopword(token)
            ]
        else: 
             return [
                self.stemmatizer.stem(token.lower())
                for paragraph in document
                for sentence in paragraph
                for (token, tag) in sentence
                if not self.is_punct(token) and not self.is_stopword(token)
            ]

    def lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.normalize(document)
            

def gensim_tfidf_vectorize(docs_normalized):
    lexicon = gensim.corpora.Dictionary(docs_normalized)
    
    tfidf   = gensim.models.TfidfModel(dictionary=lexicon)
    vectors = [tfidf[lexicon.doc2bow(doc)] for doc in docs_normalized]
    return vectors
 
def gensim_doc2vec_vectorize(docs_normalized):
    from gensim.models.doc2vec import TaggedDocument, Doc2Vec

    docs   = [
        TaggedDocument(words, ['d{}'.format(idx)])
        for idx, words in enumerate(docs_normalized)
    ]
    
    model = Doc2Vec(docs, vector_size=5, min_count=0)
    return model.docvecs


if __name__ == '__main__':

    NUMBER_OF_DOCUMENTS = 2
    
    corpus = PickledCorpusReader('../Pickled_Corpus_Sample')

   # copy generators
    docs_pickled1, docs_pickled2, docs_pickled3  = itertools.tee(corpus.docs(), 3)
  
    print("\nPLAIN (PICKLED):")
    for i in range(NUMBER_OF_DOCUMENTS):
        print('DOC {}:'.format(i+1))
        print(next(docs_pickled1))

    
    normalizer = TextNormalizer(normalizer_type='stemma')
    normalizer.fit(docs_pickled3)
    docs_normalized = normalizer.transform(docs_pickled3)
    
    print("\nNORMALIZED (STEMMA):")
    for i in range(NUMBER_OF_DOCUMENTS):
        print('DOC {}:'.format(i+1))
        print(next(docs_normalized))


    normalizer = TextNormalizer(normalizer_type='lemma')
    normalizer.fit(docs_pickled3)
    docs_normalized, docs_normalized2, docs_normalized3 = itertools.tee(normalizer.transform(docs_pickled2), 3)

    #docs_normalized1 = normalizer.transform(docs_pickled2)
    
    print("\nNORMALIZED (LEMMA):")
    for i in range(NUMBER_OF_DOCUMENTS):
        print('DOC {}:'.format(i+1))
        print(next(docs_normalized))
              
    docs_normalized = list(docs_normalized2)    
    vectors_tfidf = gensim_tfidf_vectorize(docs_normalized)
    print("\nVECTORIZED (TF-IDF):")
    for i in range(NUMBER_OF_DOCUMENTS):
        print('DOC {}:'.format(i+1))
        print(vectors_tfidf[i])
        
    vectors_doc2vec = gensim_doc2vec_vectorize(docs_normalized)
    print("\nVECTORIZED (DOC2VEC):")
    for i in range(NUMBER_OF_DOCUMENTS):
        print('DOC {}:'.format(i+1))
        print(vectors_doc2vec[i])


    with open("TF-IDF.pickle", 'wb') as f:
        pickle.dump(vectors_tfidf, f, pickle.HIGHEST_PROTOCOL)
        
    with open("DOC2VEC.pickle", 'wb') as f:
        pickle.dump(vectors_doc2vec, f, pickle.HIGHEST_PROTOCOL) 

    print("\nDOC (FILEIDS)):")
    for i in range(NUMBER_OF_DOCUMENTS):
        print('DOC {}:'.format(i+1))
        print(corpus.fileids()[i])
