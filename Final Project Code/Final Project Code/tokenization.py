from util import *

# Add your import statements here
import re
from nltk.tokenize import TreebankWordTokenizer	

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach
		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence
		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		tokenizedText = []
		
		for sentence in text:
			# Split sentence into words/punctuation using regex
			# - This pattern keeps apostrophes in contractions with [\w']+
			# - Splits at punctuation marks with [^\w\s]
			# - Naturally splits at word boundaries by separating matches
			tokens = re.findall(r"[\w']+|[^\w\s]", sentence)
			
			tokenizedText.append(tokens)
		
		return tokenizedText

	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizer = TreebankWordTokenizer()         # Penn Treebank Tokenizer
		tokenizedText = []
		for sentence in text:
			tokens = tokenizer.tokenize(sentence)   # Tokenize input sentence using Penn Treebank Tokenizer
			tokenizedText.append(tokens)

		return tokenizedText
