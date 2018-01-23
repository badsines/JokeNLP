from pandas import read_json

import bs_embeddings

# this is the API entry point.  It loads our model from disk
# building it as necessary.  This assumes that reddit_jokes.json and optionally,
# GoogleNews-vectors-negative300.bin.gz are in the current directory.
bse = bs_embeddings.bs_embeddings()

# Now, load our dataset
df = read_json('reddit_jokes.json')
print(df.head())

# Preprocessing.  Nothing happening with our embeddings here.
df['sentences'] = df['title'] + ' ' + df['body']
#df.drop_duplicates(subset='sentences', inplace=True)
analyze = bse.tf_vectorizer.build_analyzer()
df['tokenlist'] = [analyze(s) for s in df.sentences.tolist()]

# Use bs_embeddings object to get the vectorizations of jokes
df['mean_w2v_jokes'] = bse.mean_w2v_jokes(df.tokenlist).tolist()
#df['mean_w2v_google'] = bse.mean_w2v_google(df.tokenlist).tolist()
df['d2v'] = bse.d2v(df.id)
print(df.head())


# Now, suppose we want to redo the vectorization but with more
# dimensions and more training.

# Force a rebuild of our D2V model (200 dimensions, 15 epochs):
bse.force_build_model(300, 25)
# Use bs_embeddings object to get the vectorizations of jokes
df['mean_w2v_jokes'] = bse.mean_w2v_jokes(df.tokenlist).tolist()
df['d2v'] = bse.d2v(df.id)
print(df.head())
