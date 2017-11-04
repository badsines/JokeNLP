#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that creates an object "preprocess" with typical methods for cleaning 
text.

"""

# Load libraries
import pandas as pd
from nltk.data import load
from nltk.corpus import stopwords

class preprocess(object):
    
    
    def __init__(self, make_lowercase=True, rm_punctuation=True, 
                 rm_whitespace=True, rm_stopwords=True):
       self.make_lowercase = make_lowercase
       self.rm_punctuation = rm_punctuation
       self.rm_whitespace = rm_whitespace
       self.rm_stopwords = rm_stopwords

    
    def cleanData(self, data):
        """
        Cleans data by making text lowercase, removes punctuations, removes 
        extra white space, and removing english stop words.
        
        params:
            data: a column in a panda dataframe.
        
        output:
            data: a panda dataframe with the new clean column.
        """
        
        # Make text lowercase
        if (self.make_lowercase):
            data = data.str.lower()
        
        # Remove punctuation
        if (self.rm_punctuation):
            data= data.str.replace('[^\w\s]','')
        
        # Remove extra whitespace
        if (self.rm_whitespace):
            data = data.str.strip()
        
        # Remove stop words
        if (self.rm_stopwords):
            
            # Get stop words
            stop = stopwords.words('english')
            
            # Create function to remove stop words
            stop_word_func = lambda x: ' '.join(
                            [word for word in x.split() if word not in (stop)]
                            )
            
            # Remove stop words from the body column
            data = data.apply(stop_word_func)
        
        return data


    def removeDuplicates():
        pass
    
    
    def tokenizeText(self, data):
        """
        Breaks text into english words.
        
        params:
            data: a panda dataframe with column containing
            text.
            
        output:
            words: a list containing the words of each text.
        """
        
        # Load tokenizer
        tokenizer = load('tokenizers/punkt/english.pickle')
        
        # Split the text into English words
        words = []
        for text in data:
            sentence = tokenizer.tokenize(text)
            for s in sentence:
                words.append(s.split())
        
        return words
 
# Testing below. Delete before merging.
jokeData = pd.read_json("reddit_jokes.json")
prep = preprocess()       
jokeData['body'] = prep.cleanData(jokeData['body'])
jokeData['title'] = prep.cleanData(jokeData['title'])
words = prep.tokenizeText(jokeData['title'])
print(words[1])
print(jokeData.head(n=5))
