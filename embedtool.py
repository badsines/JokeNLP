#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
import time

from gensim.models.keyedvectors import KeyedVectors
from pandas import read_json
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from pandas import read_json
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
'''
Build a tool that can be used to help efforts with word embeddings
- it can train a model and serialize it to disk
- it can load a serialized model from disk
- it can use the model to predict classifications
- it can score the model (given a classified input)

embedtool.py --train -i classified_self.pointcloud.txt --algorithm DT --model my_decision_tree.model
embedtool.py --predict -i tile_unclass.txt -model my_decision_tree.model 
embedtool.py --score -i classified_input.txt -model my_decision_tree.model
'''

class EmbeddingsTool():
    NDIM = 100
    def __init__(self, *args, **kwargs):

        self.embeddings_file = kwargs['embeddings_file']
        time0 = time.time()
        print('loading embeddings file {}'.format(self.embeddings_file))
        if os.path.exists(self.embeddings_file):
            if 'GoogleNews' in self.embeddings_file: # Google's pre-trained model
               self.embeddings = KeyedVectors.load_word2vec_format(self.embeddings_file, binary=True)
            else:  # Some other model, presumably the one we generated in "make_embeddings"
               self.embeddings = KeyedVectors.load(self.embeddings_file)
        print('  ... took {} sec'.format(time.time()-time0))
           
        self.analyze = CountVectorizer(stop_words='english').build_analyzer()
        return
    def info(self):
       print ('The vocabulary is stored as a standard {} of len {}'.format(type(self.embeddings.vocab), len(self.embeddings.vocab)))
       print ('There is also a {}, also of len {}'.format(type(self.embeddings.index2word), len(self.embeddings.index2word)))
       print ('Here are the first 10 entries:  {}'.format(self.embeddings.index2word[:10])) 
       vKing = self.embeddings['king']
       print ('Length of our vectorization of "king" (see ctor of Word2Vec in make_embeddings) : {}'.format(len(vKing)))
       print ('First full feature vector of "king":  {}'.format(vKing))
#

    def load_data(self, input):
        self.df = read_json(input)

        return self.df
    
    def make_embeddings(self, input):
        if 'GoogleNews' in self.embeddings_file: # Google's pre-trained model
            raise ValueError("attempting to overwrite the Google corpus.")
            
        print('Loading input {}'.format(input))
        time0 = time.time()
        self.df = self.load_data(input)
        print('  ... took {} sec'.format(time.time()-time0))
        sentences = self.df['title'] + ' ' + self.df['body']
        sentences = [self.analyze(s) for s in sentences.tolist()]
        time0 = time.time()
        print('Fitting embeddings ... (hard coded dimenstions is {})'.format(EmbeddingsTool.NDIM))
        self.embeddings = word2vec.Word2Vec(sentences, size=EmbeddingsTool.NDIM, min_count=10, workers=4)
        print('  ... took {} sec'.format(time.time()-time0))
        time0 = time.time()
        print('Saving embeddings to output {}'.format(self.embeddings_file))
        self.embeddings.wv.save(self.embeddings_file)
        print('  ... took {} sec'.format(time.time()-time0))
        
        return
    
    def query(self, q):
        matches = []
        time0 = time.time()
        print('Querying embeddings ...')
        if len(q.split()) == 1:
            q = self.analyze(q)[0]
            matches = self.embeddings.most_similar(positive=['{}'.format(q)])
        else:
            matches = eval('self.embeddings.{}'.format(q))
        print('  ... took {} sec'.format(time.time()-time0))
        
        return matches

    def csv_dump(self):
        for word in self.embeddings.index2word:
            try:
                print ('{}, '.format(word), end='')
                for feat in self.embeddings[word]:
                    print('{}, '.format(feat), end='')
                print('')
            except UnicodeError: # non-ascii characters!
                continue

def main():
    retval = 0
    description = 'Embedding Tool, create, use,  CBOW embeddings and possibly other things.'
    epilog = '''Examples:
    # this creates a an embeddings file from the reddit jokes data store
    python embedtool.py --make_embeddings reddit_jokes.json --embeddings_file jokes_cbow.model

    # this shows how to load from disk and use that embeddings file (simplified)
    python embedtool.py --query king --embeddings_file jokes_cbow.model
    
    # If you add more than a single word, the tool assumes it's a valid KeyedVector API
    # See here:  https://radimrehurek.com/gensim/models/word2vec.html
    # Here are some more examples
        python embedtool.py --query "most_similar(positive=['woman', 'king'], negative=['man'])"  --embeddings_file jokes_cbow.model
        python embedtool.py --query "most_similar_cosmul(positive=['woman', 'king'], negative=['man'])"  --embeddings_file jokes_cbow.model
        python embedtool.py --query "doesnt_match('breakfast cereal lunch dinner'.split())"  --embeddings_file jokes_cbow.model
        python embedtool.py --query "similarity('woman', 'man')"  --embeddings_file jokes_cbow.model

    # you can also use the giant pretrained model from google, see here
    # http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
    # you'll need to d/l first (1.6 GB) and note it will take several minutes to load
    python embedtool.py --query king --embeddings_file GoogleNews-vectors-negative300.bin.gz
    '''
    parser = argparse.ArgumentParser(description=description, epilog=epilog, 
     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--make_embeddings', dest='make_embeddings')
    parser.add_argument('--embeddings_file', dest='embeddings_file', required=True)
    parser.add_argument('--query', dest='query', 
                         help="query the specified embeddings model: 'king' or 'most_similar(positive=['king', 'man'], negative=[woman])'")
    parser.add_argument('--export', dest='export', action='store_true',
                         help="export as CSV to std out (redirect as required).")  
    parser.add_argument('--info', dest='info', action='store_true')

    args = parser.parse_args()
    make_embeddings = args.make_embeddings
    embeddings_file = args.embeddings_file
    query = args.query
    export = args.export
    info = args.info

    tool = EmbeddingsTool(embeddings_file=embeddings_file)

    if make_embeddings:   
        tool.make_embeddings(make_embeddings)
    
    if query:
        m = tool.query(query)
        print('{}'.format(m))

    if export:
        tool.csv_dump()

    if info:
        tool.info()
        
    return retval


if __name__ == '__main__':
    retval = main()
    sys.exit(retval)
