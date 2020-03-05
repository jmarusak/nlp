from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from reader import CorpusReader
from normalizer import TextNormalizer

N_TOPICS = 20
N_TERM_PER_TOPIC = 10

def identity(words):
    return words

if __name__ == '__main__':
    
    reader = CorpusReader()
    normalizer = TextNormalizer()
    docs = normalizer.fit_transform(reader.docs())
    #docs = normalizer.fit_transform(reader.docs([23890098, 31186339]))
           
    vectorizer = CountVectorizer(preprocessor=None, lowercase=False)
    docs = vectorizer.fit_transform(docs)
    
    model =  LatentDirichletAllocation(n_components=N_TOPICS)
    model.fit_transform(docs)

    names = vectorizer.get_feature_names()
    topics = dict()
    
    for idx, topic in enumerate(model.components_):
        features = topic.argsort()[:-(N_TERM_PER_TOPIC - 1): -1]
        tokens = [names[i] for i in features]
        topics[idx] = tokens
        
    print(topics)
    
    for topic, terms in topics.items():
        print("Topic #{}:".format(topic+1))
        print(terms)

    print("Done.")