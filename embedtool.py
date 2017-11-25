#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
import time

from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.doc2vec import TaggedDocument

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
from pandas import read_json


class EmbeddingsTool():
    '''Utility class to build, query and provide example usage of
    text embeddings.
    '''
    NDIM = 100

    def __init__(self, *args, **kwargs):
        '''Returns EmbeddingsTools that is ready for use
        Keyword Arguments:
        embeddings_file -- the file from (to) which the embeddings
         are serialized.
        doc2vec -- if True, use Doc2Vec embeddings, otherwise use Word2Vec.
        '''

        self.embeddings_file = kwargs['embeddings_file']
        self.doc2vec = kwargs['doc2vec']
        time0 = time.time()
        print('loading embeddings file {}'.format(self.embeddings_file))
        if os.path.exists(self.embeddings_file):
            # Google's Pre-trained model.
            if 'GoogleNews' in self.embeddings_file:
                self.embeddings = KeyedVectors.load_word2vec_format(
                    self.embeddings_file, binary=True)
            else:  # One of our "make_embeddings" models
                if self.doc2vec:
                    self.embeddings = doc2vec.Doc2Vec.load(
                        self.embeddings_file)
                else:
                    self.embeddings = KeyedVectors.load(self.embeddings_file)

        print('  ... took {} sec'.format(time.time() - time0))

        self.analyze = CountVectorizer(stop_words='english').build_analyzer()
        return

    def info(self):
        '''Dump out information about the specified embeddings file.'''
        def info_wv(self, wv):
            '''Dump out information about a Word2Vec embedding '''
            print('Word Embeddings.')
            print('  Length:  {}.  First 3 words {}'.format(
                len(wv.vocab), wv.index2word[:3]))
            print('  Example word embedding.  wv["{}"]:'.format('king'))
            print('    V length:  {}, embedding follows:  {}'.format(
                len(wv['king']), wv['king']))

        if self.doc2vec:
            print('Document Embeddings.')
            d2v = self.embeddings
            print('  Length:  {}.  First 3 entries {}'.format(
                len(d2v.docvecs.doctags), d2v.docvecs.doctags.keys()[:3]))
            tag0 = d2v.docvecs.doctags.keys()[0]
            print(
                '  Example document embedding.  d2v.docvecs["{}"]:'.format(tag0))
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
        '''Build the embeddings and serialize to disk.'''
        if 'GoogleNews' in self.embeddings_file:  # Google's pre-trained model
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
                taggeddocs.append(td)
            self.embeddings = doc2vec.Doc2Vec(
                alpha=0.025, min_alpha=0.025,
                size=EmbeddingsTool.NDIM,
                window=8,
                min_count=5,
                workers=4)
            # to do:  continue training
            self.embeddings .build_vocab(taggeddocs)
            print('  ... build_vocab took {} sec'.format(time.time() - time0))
            for epoch in range(10):
                self.embeddings .train(taggeddocs,
                total_examples=self.embeddings.corpus_count,
                epochs=self.embeddings.iter)
                print('  ... training epoch {} through {} sec'.format(epoch, time.time() - time0))
                self.embeddings .alpha -= 0.002  # decrease the learning rate
                self.embeddings .min_alpha = self.embeddings.alpha  # fix the learning rate, no decay
        else:
            self.embeddings = word2vec.Word2Vec(
                sentences, size=EmbeddingsTool.NDIM, min_count=10, workers=4)
        print('  ... took {} sec'.format(time.time() - time0))
        time0 = time.time()
        print('Saving embeddings to output {}'.format(self.embeddings_file))
        if self.doc2vec:
            print('writing doc2vec')
            self.embeddings.delete_temporary_training_data(
                keep_doctags_vectors=True, keep_inference=True)
            self.embeddings.save(self.embeddings_file)
        else:
            self.embeddings.wv.save(self.embeddings_file)
        print('  ... took {} sec'.format(time.time() - time0))

        return

    def query(self, q):
        '''Execute Queries against the model.

        q -- this is the query string.
          It can be either a single term (e.g. 'king') which will use the
          'most_similar' API OR
          It can be a call to the method on either KeyedVectors (Word2Vec)
          or DocvecsArray (Doc2Vec), extended as necessary
          (e.g. "doesnt_match('breakfast cereal lunch dinner'.split())")
         See https://radimrehurek.com/gensim/models/doc2vec.html and
             https://radimrehurek.com/gensim/models/word2vec.html
             and the example usages of this tool (--help)
        '''
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
                matches = self.embeddings.most_similar(
                    positive=['{}'.format(q)])
            else:
                matches = eval('self.embeddings.{}'.format(q))

        return q, matches

    def csv_dump(self):
        ''' Dump Word2Vec embeddings to CSV file. '''
        for word in self.embeddings.index2word:
            try:
                print('{}, '.format(word), end='')
                for feat in self.embeddings[word]:
                    print('{}, '.format(feat), end='')
                print('')
            except UnicodeError:  # non-ascii characters!
                continue


def main():
    retval = 0
    description = 'Embedding Tool, create, use, various text embeddings.'
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

        # Find duplicate jokes based on doc2vec similarity
        python embedtool.py -i reddit_jokes.json --query 1ah5m2  --embeddings_file reddit.d2v.model --doc2vec

        # Blonde jokes
        python embedtool.py -i reddit_jokes.json --query 1b36gi  --embeddings_file reddit.d2v.model --doc2vec


    # you can also use the giant pretrained model from google, see here
    # http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
    # you'll need to d/l first (1.6 GB) and note it will take several minutes to load
    python embedtool.py --query king --embeddings_file GoogleNews-vectors-negative300.bin.gz


    # Info and Export also work:
    python embedtool.py --export --embeddings_file jokes_cbow.model
    python embedtool.py --info --embeddings_file jokes_cbow.model

    '''
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--make_embeddings',
        dest='make_embeddings',
        action='store_true')
    parser.add_argument('-i', dest='input_')
    parser.add_argument('--doc2vec', dest='doc2vec', action='store_true')
    parser.add_argument(
        '--embeddings_file',
        dest='embeddings_file',
        required=True)
    parser.add_argument(
        '--query',
        dest='query',
        help="query the specified embeddings model: 'king' or 'most_similar(positive=['king', 'man'], negative=[woman])'")
    parser.add_argument(
        '--export',
        dest='export',
        action='store_true',
        help="export as CSV to std out (redirect as required).")
    parser.add_argument(
        '--info',
        dest='info',
        action='store_true',
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
        src, matches = tool.query(query)
        print('{}'.format(matches))
        if doc2vec:
            i = np.where(tool.df.id == src)[0][0]
            s = tool.df.title.iloc[i] + ' ' + tool.df.body.iloc[i]
            print('Source:  {}. {}\nMost Similar:\n'.format(i, s))

            for m, _ in matches:
                i = np.where(tool.df.id == m)[0][0]
                s = tool.df.title.iloc[i] + ' ' + tool.df.body.iloc[i]
                print('{}. {}'.format(i, s))

    if export:
        tool.csv_dump()

    if info:
        tool.info()

    return retval


if __name__ == '__main__':
    retval = main()
    sys.exit(retval)
