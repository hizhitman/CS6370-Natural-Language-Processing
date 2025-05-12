from util import *

# Add your import statements here
from nltk.corpus import stopwords
import nltk
import json
import math
from collections import defaultdict

nltk.download('stopwords', quiet=True)

class StopwordRemoval():
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def fromList(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """

        # Fill in code here
        
        stopwordRemovedText = []
        for sentence in text:
            filtered_sentence = [word for word in sentence if word.lower() not in self.stop_words]   # Remove stopword sentencewise
            stopwordRemovedText.append(filtered_sentence)                                            # Append modified sentence
        return stopwordRemovedText
