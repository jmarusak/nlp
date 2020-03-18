
import os
import pickle

from nltk import pos_tag
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

class Preprocessor(object):
    """
    The preprocessor wraps a corpus object PlaintextCorpusReader and manages the stateful tokenization and part of 
    speech tagging into a directory that is stored in a format that can be read by the PickledCorpusReader.
    """

    def __init__(self, corpus, target=None, **kwargs):
        """
        The corpus is the PlaintextCorpusReader to preprocess and pickle.
        The target is the directory on disk to output the pickled corpus to.
        """
        self.corpus = corpus
        self.target = target

    def fileids(self, fileids=None, categories=None):
        """
        Helper function access the fileids of the corpus
        """
        if fileids:
            return fileids
        return self.corpus.fileids()

    def abspath(self, fileid):
        """
        Returns the absolute path to the target fileid from the corpus fileid.
        """
        # Find the directory, relative from the corpus root.
        parent = os.path.relpath(
            os.path.dirname(self.corpus.abspath(fileid)), self.corpus.root
        )

        # Compute the name parts to reconstruct
        basename  = os.path.basename(fileid)
        name, ext = os.path.splitext(basename)

        # Create the pickle file extension
        basename  = name + '.pickle'

        # Return the path to the file relative to the target.
        return os.path.normpath(os.path.join(self.target, parent, basename))

    def tagging(self, fileid):
        """
        Tags a document in the corpus. Tokenization done by PlaintextCorpusReader
        Returns a generator of paragraphs, which are lists of sentences,
        which in turn are lists of part of speech tagged words.
        """
        for paragraph in self.corpus.paras(fileids=fileid):
            yield [pos_tag(sent) for sent in paragraph]
            
    def process(self, fileid):
        """
        Writes the document as a pickle to the target location.
        This method is called multiple times from the transform runner.
        """
        
        # Compute the outpath to write the file to.
        target = self.abspath(fileid)
        parent = os.path.dirname(target)
        
        # Make sure the directory exists
        if not os.path.exists(parent):
            os.makedirs(parent)
        
        # Make sure that the parent is a directory and not a file
        if not os.path.isdir(parent):
            raise ValueError("Please supply a directory to write preprocessed data to.")
        
        # Create a data structure for the pickle
        document = list(self.tagging(fileid))
        
        # Open and serialize the pickle to disk
        with open(target, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)
        
        # Clean up the document
        del document
        
        # Return the target fileid
        return target

    def transform(self, fileids=None, categories=None):
        """
        Transform the wrapped corpus, writing out the segmented, tokenized,
        and part of speech tagged corpus as a pickle to the target directory.
        """

        # Make the target directory if it doesn't already exist
        if not os.path.exists(self.target):
            os.makedirs(self.target)
        
        # Resolve the fileids to start processing and return the list of 
        # target file ids to pass to downstream transformers. 
        return [
            self.process(fileid)
            for fileid in self.fileids(fileids, categories)
        ]

if __name__ == '__main__':
    text_corpus = PlaintextCorpusReader('../../corpora/politics', '.00*\.txt')  #'.*\.txt')

    preprocessor = Preprocessor(text_corpus, target='../../corpora/politics_pickled')
    preprocessor.transform()
    print(preprocessor.fileids())
