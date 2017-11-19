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

from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
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
        self.doc2vec = kwargs['doc2vec']
        time0 = time.time()
        print('loading embeddings file {}'.format(self.embeddings_file))
        if os.path.exists(self.embeddings_file):
            if 'GoogleNews' in self.embeddings_file: # Google's pre-trained model
               self.embeddings = KeyedVectors.load_word2vec_format(self.embeddings_file, binary=True)
            else:  # Some other model, presumably the one we generated in "make_embeddings"
                if self.doc2vec:
                    self.embeddings = doc2vec.Doc2Vec.load(self.embeddings_file)
                else:
                    self.embeddings = KeyedVectors.load(self.embeddings_file)
                    
        print('  ... took {} sec'.format(time.time()-time0))
           
        self.analyze = CountVectorizer(stop_words='english').build_analyzer()
        return
    def info(self):
        def info_wv(self, wv):
            print('Word Embeddings.')
            print('  Length:  {}.  First 3 words {}'.format(len(wv.vocab), wv.index2word[:3]))
            print('  Example word embedding.  wv["{}"]:'.format('king'))
            print('    V length:  {}, embedding follows:  {}'.format(
             len(wv['king']), wv['king']))
            
        if self.doc2vec:
            print('Document Embeddings.')
            d2v = self.embeddings
            print('  Length:  {}.  First 3 entries {}'.format(len(d2v.docvecs.doctags), 
             d2v.docvecs.doctags.keys()[:3]))
            tag0 = d2v.docvecs.doctags.keys()[0]
            print('  Example document embedding.  d2v.docvecs["{}"]:'.format(tag0))
            print('    V length:  {}, embedding follows:  {}'.format(
             len(d2v.docvecs[tag0]), d2v.docvecs[tag0]))
            print ('The word embeddings are available via the "wv" member:')
            info_wv(self, d2v.wv)
        else:
            info_wv(self, self.embeddings)
#

    def load_data(self, input):
        self.df = read_json(input)

        return self.df
    
    def make_embeddings(self):
        if 'GoogleNews' in self.embeddings_file: # Google's pre-trained model
            raise ValueError("attempting to overwrite the Google corpus.")
            
        sentences = self.df['title'] + ' ' + self.df['body']
        self.df['tokenlist'] = [self.analyze(s) for s in sentences.tolist()]
        time0 = time.time()
        print('Fitting embeddings ... (hard coded dimensions is {})'.
              format(EmbeddingsTool.NDIM))
        if self.doc2vec:
            taggeddocs = []
            for i, tokenlist in enumerate(self.df.tokenlist):
                td = TaggedDocument(tokenlist, [unicode(self.df.id[i])])
                #print('{}, {}, {}'.format(i, self.df.id[i], tokenlist[:4]))
                taggeddocs.append(td)
            self.embeddings = doc2vec.Doc2Vec(taggeddocs, 
             size=EmbeddingsTool.NDIM, window = 8, min_count=5, workers=4)
        else:
            self.embeddings = word2vec.Word2Vec(sentences, 
             size=EmbeddingsTool.NDIM, min_count=10, workers=4)
        print('  ... took {} sec'.format(time.time()-time0))
        time0 = time.time()
        print('Saving embeddings to output {}'.format(self.embeddings_file))
        if self.doc2vec:
            print('writing doc2vec')
            self.embeddings.delete_temporary_training_data(
             keep_doctags_vectors=True, keep_inference=True)
            self.embeddings.save(self.embeddings_file)
        else:
            self.embeddings.wv.save(self.embeddings_file)
        print('  ... took {} sec'.format(time.time()-time0))
        
        return
    
    def query(self, q):
        matches = []
        if self.doc2vec:
            d2v = self.embeddings.docvecs
            if (len(q.split()) == 1):
                q = self.analyze(q)[0]
                matches = d2v.most_similar('{}'.format(q))
            else:
                matches = eval('d2v.{}'.format(q))
        else:           
            if len(q.split()) == 1:
                q = self.analyze(q)[0]
                matches = self.embeddings.most_similar(positive=['{}'.format(q)])
            else:
                matches = eval('self.embeddings.{}'.format(q))
        
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
    # word2vec embeddings
    python embedtool.py --make_embeddings reddit_jokes.json --embeddings_file jokes_cbow.model

    # doc2vec embeddings
    python embedtool.py --make_embeddings reddit_jokes.json --doc2vec --embeddings_file reddit.d2v.model


    # this shows how to load from disk and use that embeddings file (simplified)
    python embedtool.py --query king --embeddings_file jokes_cbow.model
    
    # If you add more than a single word, the tool assumes it's a valid KeyedVector API
    # See here:  https://radimrehurek.com/gensim/models/word2vec.html
    # Here are some more examples
        python embedtool.py --query "most_similar(positive=['woman', 'king'], negative=['man'])"  --embeddings_file jokes_cbow.model
        python embedtool.py --query "most_similar_cosmul(positive=['woman', 'king'], negative=['man'])"  --embeddings_file jokes_cbow.model
        python embedtool.py --query "doesnt_match('breakfast cereal lunch dinner'.split())"  --embeddings_file jokes_cbow.model
        python embedtool.py --query "similarity('woman', 'man')"  --embeddings_file jokes_cbow.model

        python embedtool.py --query 4q6b2d  --embeddings_file reddit.d2v.model --doc2vec


    # you can also use the giant pretrained model from google, see here
    # http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
    # you'll need to d/l first (1.6 GB) and note it will take several minutes to load
    python embedtool.py --query king --embeddings_file GoogleNews-vectors-negative300.bin.gz


    # Info and Export also work:
    python embedtool.py --export --embeddings_file jokes_cbow.model
    python embedtool.py --info --embeddings_file jokes_cbow.model
   
    '''
    parser = argparse.ArgumentParser(description=description, epilog=epilog, 
     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--make_embeddings', dest='make_embeddings', action = 'store_true')
    parser.add_argument('-i', dest='input_')
    parser.add_argument('--doc2vec', dest='doc2vec', action='store_true')
    parser.add_argument('--embeddings_file', dest='embeddings_file', required=True)
    parser.add_argument('--query', dest='query', 
                         help="query the specified embeddings model: 'king' or 'most_similar(positive=['king', 'man'], negative=[woman])'")
    parser.add_argument('--export', dest='export', action='store_true',
                         help="export as CSV to std out (redirect as required).")  
    parser.add_argument('--info', dest='info', action='store_true',
                         help='dump out some info about the embeddings (vector size, vocab size ...)')

    args = parser.parse_args()
    make_embeddings = args.make_embeddings
    input_ = args.input_
    doc2vec = args.doc2vec
    embeddings_file = args.embeddings_file
    query = args.query
    export = args.export
    info = args.info

    tool = EmbeddingsTool(embeddings_file=embeddings_file, doc2vec=doc2vec)
    
    if input_:
        tool.load_data(input_)

    if make_embeddings:   
        tool.make_embeddings()
    
    if query:
        matches = tool.query(query)
        print('{}'.format(matches))
        if doc2vec:
            for m, _ in matches:
                mask = tool.df['id'] == m
                print ('{}.  {} {}'.format(m, tool.df.loc[mask].title.to_string(), 
                 tool.df.loc[mask].body.to_string()))

    if export:
        tool.csv_dump()

    if info:
        tool.info()
        
    return retval


if __name__ == '__main__':
    retval = main()
    sys.exit(retval)
