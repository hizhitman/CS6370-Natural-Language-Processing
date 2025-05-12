from util import *

# Add your import statements here

from nltk.stem import PorterStemmer

class InflectionReduction:
	def __init__(self):
		self.stemmer = PorterStemmer()     # Porter Stemmer
        
    # Applies Porter's stemming algorithm to tokenized text
	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""
		
		#Fill in code here
		
		reducedText = []         
		for sentence in text:
			stemmed_sentence = [self.stemmer.stem(word) for word in sentence]    # Apply Porter Stemming to each sentence in text
			reducedText.append(stemmed_sentence)                                 # Append stemmed sentence (as a list of stemmed words)
		
		return reducedText


