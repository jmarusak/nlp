import os
import pickle
import nltk
import gensim
import itertools
import unicodedata

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

corpus = [
[
    "See the bat sight sneeze!. This is paragraph one",
    "The elephant sneezed at the sight of potatoes. Bats can see via echolocation."
],
[
    "Wondering, she opened the door to the studio. Fruit fly likes banana."
]
]

def reader(corpus):
    for document in corpus:
        yield [paras(document)] 

def paras(document):
    for paragraph in document:
        return [pos_tag(wordpunct_tokenize(sent)) for sent in sent_tokenize(paragraph)]


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
  
    # copy generators
    docs_pickled1, docs_pickled2, docs_pickled3  = itertools.tee(reader(corpus), 3)
  
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
