from util import *

# Add your import statements here
import re
from nltk.tokenize import PunktSentenceTokenizer

class SentenceSegmentation():
	def __init__(self):
		# List of abbreviations
		self.abbreviations = {'dr', 'mr', 'mrs', 'ms', 'prof', 'ph.d', 'u.s.a', 'u.k', 'inc', 'temp.', 'etc'}

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach
		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)
		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		# Step 1: Tokenize text into words with punctuation
		tokens = re.findall(r'\S+|\n', text)
		
		segmentedText = []
		current_sentence = []
		
		i = 0
		while i < len(tokens):
			token = tokens[i]
			current_sentence.append(token)
			
			# Rule 1: Check if token ends with a sentence terminator
			if re.search(r'[.!?]$', token):
				# Rule 2: Check if the token is a common abbreviation
				normalized_token = token.lower().rstrip('.!?')
				is_abbreviation = normalized_token in self.abbreviations
				
				# Rule 3: Check if next token starts with a capital letter
				next_token = tokens[i+1] if i+1 < len(tokens) else None
				next_token_capitalized = next_token and next_token[0].isupper() if next_token else False
				
				# Determine if this is a sentence boundary
				# It's a boundary if:
				# - It's not an abbreviation, OR
				# - It's an abbreviation but the next token is capitalized
				if not is_abbreviation or (is_abbreviation and next_token_capitalized):
					segmentedText.append(' '.join(current_sentence))
					current_sentence = []
			
			i += 1
		
		# Add any remaining tokens as the last sentence
		if current_sentence:
			segmentedText.append(' '.join(current_sentence))
		
		return segmentedText
	
	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		tokenizer = PunktSentenceTokenizer()      # Punkt Sentence Tokenizer
		segmentedText = tokenizer.tokenize(text)  # Applying Punkt Sentence Tokenizer on text
		
		return segmentedText
