
from __future__ import print_function
import os
import time

from gensim.models.keyedvectors import KeyedVectors
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
from pandas import read_json
class bs_embeddings():
    # These two files are assumed to be in the CWD.
    JOKES_D2V_MODEL='jokes.d2v.model'
    GOOGLE_W2V_MODEL='GoogleNews-vectors-negative300.bin.gz'
    def __init__(self, *args, **kwargs):

        self.tf_vectorizer = CountVectorizer(
        max_df=0.95, min_df=2, max_features=20000,
        stop_words='english')

        # Always load our D2V model, build it if necessary.
        if not os.path.exists(bs_embeddings.JOKES_D2V_MODEL):
            self.force_build_model(100, 10) #default is 100 dimensions, 10 epochs of training
        else:
            self.jokes_d2v_model = doc2vec.Doc2Vec.load(bs_embeddings.JOKES_D2V_MODEL)

        # If the Google W2V model is available, load it.
        self.google_model = None
        if os.path.exists(bs_embeddings.GOOGLE_W2V_MODEL):
            print('loading the google model...')
            self.google_model = KeyedVectors.load_word2vec_format(
                    bs_embeddings.GOOGLE_W2V_MODEL, binary=True)
            print('    ... Done.')

    # Overwrite JOKES_D2V_MODEL file.
    def force_build_model(self, ndim, nepochs):
        jokes_json = 'reddit_jokes.json'
        if os.path.exists(jokes_json):
            self.df  = read_json(jokes_json)
            self.df['sentences'] = self.df['title'] + ' ' + self.df['body']
            self.df.drop_duplicates(subset='sentences', inplace=True)

            # Strips out punctuation, Lower cases, removes English stop words
            # and white space.  Leaves numbers
            analyze = self.tf_vectorizer.build_analyzer()

            self.df['tokenlist'] = [analyze(s) for s in self.df.sentences.tolist()]
            print('Building D2V model ...')
            taggeddocs = []
            time0 = time.time()
            for i, tokenlist in enumerate(self.df.tokenlist):
                td = TaggedDocument(tokenlist, [unicode(self.df.iloc[i].id)])
                taggeddocs.append(td)
            self.embeddings = doc2vec.Doc2Vec(
                alpha=0.025, min_alpha=0.025,
                size=ndim,
                window=8,
                min_count=5,
                workers=4)
            self.embeddings .build_vocab(taggeddocs)
            print('  ... build_vocab took {} sec'.format(
                    time.time() - time0))
            for epoch in range(nepochs):
                self.embeddings .train(
                    taggeddocs,
                    total_examples=self.embeddings.corpus_count,
                    epochs=self.embeddings.iter)
                print('  ... training epoch {} through {} sec'.format(
                    epoch, time.time() - time0))
                self.embeddings.alpha -= 0.002
                self.embeddings.min_alpha = self.embeddings.alpha

            self.embeddings.save(bs_embeddings.JOKES_D2V_MODEL)
            self.jokes_d2v_model = doc2vec.Doc2Vec.load(bs_embeddings.JOKES_D2V_MODEL)
        else:
            msg = '{} not found while attempting to build Doc2Vec model'.format(jokes_json)
            raise ValueError(msg)

    def mean_w2v_jokes(self, tokenizedjokes):
        result = []
        ndim = len(self.jokes_d2v_model.wv.syn0[0])
        for tokenlist in tokenizedjokes:
            joke_vec = np.zeros(ndim)
            for token in tokenlist:
                if token in self.jokes_d2v_model.wv.vocab:
                    joke_vec += self.jokes_d2v_model.wv[token]
            joke_vec /= float(len(tokenlist))
            result.append(joke_vec)
        return np.array(result)

    def mean_w2v_google(self, tokenizedjokes):
        if not self.google_model:
            print ('!! Google model not loaded.  Get it from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit')
            return None
        result = []
        ndim = len(self.google_model.syn0[0])
        for tokenlist in tokenizedjokes:
            joke_vec = np.zeros(ndim)
            for token in tokenlist:
                if token in self.google_model.vocab:
                    joke_vec += self.google_model[token]
            joke_vec /= float(len(tokenlist))
            result.append(joke_vec)
        return np.array(result)


    def d2v(self, ids):
        result = []
        for id in ids:
            d2v = self.jokes_d2v_model.docvecs[id]
            result.append(d2v)
        return result
