# jokes_NLP
Take a dataset of jokes from reddit and other sources and write a Deep Learning algorithm to score them.

## the_wordnet_blue_sky_idea.ipynb
This notebook is an experiment to see if the topic of the joke can be determined through lexicographical analysis.
For each word in the title and each word in the body of the joke, a synset is created.  Then the length of the synsets
are compared.  The synset of the title words that is closest to a synset in the body is considered the topic.

I have discovered that it is really hard for a human to determine the topic of a joke.  For example, in the joke
"Time flies like an arrow.  Fruit flies like a banana", what is the topic?

