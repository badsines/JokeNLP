from collections import defaultdict
import time
import numpy as np
from pandas import read_json, DataFrame
import pickle

import bs_embeddings

def looks_like_duplicate(prev, curr):
    if prev == None:
        return False
    common = [e for e in curr.tokenlist if e in prev.tokenlist]
    l = max(len(curr.tokenlist), len(prev.tokenlist))
    cf = len(common)/float(l) if l > 0 else 1
    if curr.joke == prev.joke:
        return True # identical jokes
    if l == 0:
        return curr.joke == prev.joke # very short jokes must match exactly
    if l < 5 and cf == 1.0:
        return True
    elif cf > 0.9:
        return True
    return False

print('Loading bs_embeddings object...')
bse = bs_embeddings.bs_embeddings()

# Now we load the joke database and configure it.
print('Loading reddit_jokes dataframe ...')
df = read_json('data/reddit_jokes.json')
df['joke'] = df['title'] + ' ' + df['body']
analyze = bse.tf_vectorizer.build_analyzer()
df['tokenlist'] = [sorted(l) for l in [analyze(s) for s in df.joke.tolist()]]
df.sort_values('tokenlist', inplace=True)
N=0
# Iteratively pass through DF removing adjacent rows that appear to be duplicates
# We need to do this multiple times b/c we do not require exact matches to be duplicates.
while True:
    print('serializing to disk...')
    df.to_pickle('jokes.df.pickle')
    print('serializing from disk...')
    with open('jokes.df.pickle', 'rb') as pickle_file:
        df = pickle.load(pickle_file)
    N = N+1
    print('***************  Starting iteration N = {}  *************'.format(N))
    prev = None
    initCount = len(df.index)
    dupCount = 0
    exactDupCount = 0
    duplicates = []
    updates = defaultdict(int)
    for curr in df.itertuples():
        line = ' '.join(curr.tokenlist)
        if not looks_like_duplicate(prev, curr):
           printsource = True
           src = curr
        else:
           if printsource:
               print('*** Duplicated: {}'.format(src.joke.encode('utf-8')[:128] ))
               #print('*** {}'.format(src.tokenlist))
               printsource = False
           print('   *** Matched  {}.'.format(curr.joke.encode('utf-8')[:128]))
           #print('    {}'.format(curr.tokenlist))
           if curr.score > src.score:
               updates[src.Index] = curr.score
           duplicates.append(curr.Index)
           dupCount += 1
           if curr.joke == src.joke:
               exactDupCount += 1
        prev = curr

    print('Init Count {}, Duplicates {}, Exact Duplicates {}'.format(initCount, dupCount, exactDupCount))
    print('Ready to remove {} duplicates and update {} rows.'.format(len(duplicates), len(updates)))
    if len(duplicates) == 0:
        print ("\n\n *** Yay done:  serialized from disk and found no duplicates\n\n")
        break;
    df.drop(duplicates, inplace=True)
    updates_df = DataFrame.from_dict(updates, orient='index')
    updates_df.columns = ['score']
    df.update(updates_df)

    # verify that scores got updated.
    for id in updates:
        #print('updates[{}]=={}, df.loc[{}].score=={}'.format(id, updates[id], id, df.loc[id].score))
        assert updates[id] == df.loc[id].score
    print('Final Count:  {}'.format(len(df.index)))

print('Rebuilding the Doc2Vec Model with duplicates removed.')
bse.force_build_model(300, 25, df=df)

# Add Vectorizations to rows that remain and save dataframe that can be used
# for futher analysis
df['mean_w2v_jokes'] = bse.mean_w2v_jokes(df.tokenlist).tolist()
df['d2v'] = bse.d2v(df.id)
df.to_pickle('data/jokes.df.pickle')

print('... Now, go figure out how to predict the funny ones!')
