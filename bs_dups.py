import time
import numpy as np
from pandas import read_json
import pickle

import bs_embeddings

def looks_like_duplicate(tokenlist1, tokenlist2):
    common = [e for e in tokenlist1 if e in tokenlist2]
    l = max(len(tokenlist1), len(tokenlist2))
    cf = len(common)/float(l)
    if cf > 0.5 :
        return True
    #if cf > 0.25:
    #    print('SIMILAR NOT DUP')
    #    print('  ** distance:  {}, common token factor {}'.format(dist, cf))
    #    #print('    {}'.format(tokenlist1))
    #    #print('    {}'.format(tokenlist2))
    #    print('    ** {}'.format(joke1))
    #    print('    ** {}'.format(joke2))

bse = bs_embeddings.bs_embeddings()

# This is the D2V model built with duplicates included.
print('Running force_build_model ...')
bse.force_build_model(300, 25)

# Now we load the joke database and configure it.
df = read_json('reddit_jokes.json')
df['joke'] = df['title'] + ' ' + df['body']
#df.drop_duplicates(subset='joke', inplace=True)
analyze = bse.tf_vectorizer.build_analyzer()
df['tokenlist'] = [analyze(s) for s in df.joke.tolist()]



L0 = df.shape[0]

# Now we remove duplicates based a cosine distance
# criteria plus some scrutiny about the gross overlap
# of the duplicate tokenlists.
low_bar = 0.90
ndups = 0
ndups2 = 0
ids = df.id
todrop = []
for ii, id in enumerate(ids):
    if ii % 500 == 0:
        print('(progress {}):  Joke {} of {}'.format(time.strftime("%Y-%m-%d %H:%M"), ii, df.shape[0]))
    if id in df.id.values:
        i = np.where(df.id == id)[0][0] # np.where returns a tuple of ndarrays 
    else:
        continue
    joke1 = df.iloc[i].joke.encode('utf-8')
    l = len(df.iloc[i].tokenlist)
    matches = bse.jokes_d2v_model.docvecs.most_similar(id)
    if matches[0][1] > low_bar:
        #print('Source Joke (id {}, len {}, score {}):  {}'.format(id, l, df.score.iloc[i], joke1))
        for m, dist in matches:
            if m in df.id.values:
                j = np.where(df.id == m)[0][0]
            else:
                continue
            joke2 = df.joke.iloc[j].encode('utf-8')
            if looks_like_duplicate(df.iloc[i].tokenlist, df.iloc[j].tokenlist):
                ndups += 1
                if joke1 == joke2:
                    ndups2 += 1
                # SettingWithCopyWarning here.  But why?
                df.iloc[j].score = max(df.iloc[i].score, df.iloc[j].score)
                l = len(df.iloc[j].tokenlist)
                #if l < 7:  # print out some short duplicates so we can convince ourselves its working.
                    #print('  Matched Joke: ({}, id {}, len {}, score {}) {}'.format(dist, m, l, df.score.iloc[j], joke2))
                if j in df.index:
                    todrop.append(j)
        #print('*'*80)
    if len(todrop) > 500 or ii == len(ids)-1:
        todrop = list(set(todrop))
        df.drop(todrop, axis='index', inplace=True)
        todrop = []
# Add Vectorizations to rows that remain and save dataframe that can be used
# for futher analysis
df['mean_w2v_jokes'] = bse.mean_w2v_jokes(df.tokenlist).tolist()
df['d2v'] = bse.d2v(df.id)
df.to_pickle('jokes.df.pickle')

with open('jokes.df.pickle', 'rb') as pickle_file:
    df = pickle.load(pickle_file)

print('Initially, we had {} jokes.'.format(L0))
print('We removed {} duplicates of which {} where exact matches'.format(ndups, ndups2))
print('Afterwards, we have {} jokes.'.format(df.shape[0]))
print('... Now, go figure out how to predict the funny ones!')
