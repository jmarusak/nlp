
import itertools
import tabulate
from nltk import ne_chunk

#pip install networkx
import heapq
from operator import itemgetter
import networkx as nx

from sklearn.base import BaseEstimator, TransformerMixin

#GOODLABELS = frozenset(['PERSON', 'ORGANIZATION', 'FACILITY', 'GPE', 'GSP'])
GOODLABELS = frozenset(['PERSON'])

class EntityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, labels=GOODLABELS, **kwargs):
        self.labels = labels

    def get_entities(self, document):
        entities = []
        for paragraph in document:
            for sentence in paragraph:
                trees = ne_chunk(sentence)
                for tree in trees:
                    if hasattr(tree, 'label'):
                        if tree.label() in self.labels:
                            entities.append(
                                ' '.join([child[0].lower() for child in tree])
                                )
        return entities

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.get_entities(document)


class EntityPairs(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(EntityPairs, self).__init__()

    def pairs(self, document):
        return list(itertools.permutations(set(document), 2))

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        return [self.pairs(document) for document in documents]


class GraphExtractor(BaseEstimator,TransformerMixin):
   
    def __init__(self):
        self.G = nx.Graph()
      
    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        for document in documents:
            for first, second in document:
                if (first, second) in self.G.edges():
                    self.G.edges[(first, second)]['weight'] += 1
                else:
                    self.G.add_edge(first, second, weight=1)
        return self.G

def nbest_centrality(G, metrics, n=10):
    # Compute the centrality scores for each vertex
    nbest = {}
    for name, metric in metrics.items():
        scores = metric(G)
        # Set the score as a property on each node
        nx.set_node_attributes(G, name=name, values=scores)
        # Find the top n scores and print them along with their index
        topn = heapq.nlargest(n, scores.items(), key=itemgetter(1))
        nbest[name] = topn
    return nbest


if __name__ == '__main__':
    from reader import PickledCorpusReader

    corpus = PickledCorpusReader(root='../../corpora/politics_pickled')
    docs = corpus.docs()

    entity_extractor = EntityExtractor()
    entities = entity_extractor.fit_transform(docs)

    entity_pairing = EntityPairs()
    pairs = entity_pairing.fit_transform(entities)

    graph = GraphExtractor()
    G = graph.fit_transform(pairs)
    print(nx.info(G))

    centralities = {"Degree Centrality" : nx.degree_centrality,
                    "Betweenness Centrality" : nx.betweenness_centrality}

    centrality = nbest_centrality(G, centralities, 10)

    for measure, scores in centrality.items():
        print("Rankings for {}:".format(measure))
        print((tabulate.tabulate(scores, headers=["Top Terms", "Score"])))
        print("")

    H = nx.ego_graph(G, "harold wilson")
    
    person_centralities = {"closeness" : nx.closeness_centrality}

    person_centrality = nbest_centrality(H, person_centralities, 10)
    
    for measure, scores in person_centrality.items():
        print("Rankings for {}:".format(measure))
        print((tabulate.tabulate(scores, headers=["Top Terms", "Score"])))
        print("")        
