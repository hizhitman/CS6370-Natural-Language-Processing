from util import *

# Add your import statements here
import numpy as np

class Evaluation():
    
    
	def __init__(self):
		self.precision_matrix = np.zeros((225, 11))


	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters

		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here

		relevant_documents = 0
		for idx in range(k):
			doc = query_doc_IDs_ordered[idx]
			if int(doc) in true_doc_IDs:
				relevant_documents += 1
		
		# print(precision, relevant_documents, k)
		precision = relevant_documents/k


		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code 

		
		meanPrecision = 0
		number_queries = len(query_ids)
		
		for idx in range(number_queries):
			# for each query id, we would want to find the true_doc_IDs
			query_id = query_ids[idx]
			query_true_doc_IDs = []
			for qrel in qrels:
				if int(qrel['query_num']) == query_id:
					query_true_doc_IDs.append(int(qrel['id']))
			
			# precision for a particular query
			query_precision = self.queryPrecision(doc_IDs_ordered[idx], query_id, query_true_doc_IDs, k)
			meanPrecision += query_precision
		
		meanPrecision /= number_queries

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here

		if len(true_doc_IDs)==0:
			return 0

		relevant_docs = 0
		for idx in range(min(k,len(query_doc_IDs_ordered))):
			doc = query_doc_IDs_ordered[idx]
			if int(doc) in true_doc_IDs:
				relevant_docs += 1
		recall = relevant_docs/len(true_doc_IDs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here

		meanRecall = 0
		number_queries = len(query_ids)
		for idx in range(number_queries):
			# for each query id, we would want to find the true_doc_IDs
			query_id = int(query_ids[idx])
			query_true_doc_IDs = []
			for qrel in qrels:
				if int(qrel['query_num']) == query_id:
					query_true_doc_IDs.append(int(qrel['id']))
			
			# recall for a particular query
			query_recall = self.queryRecall(doc_IDs_ordered[idx], query_id, query_true_doc_IDs, k)
			meanRecall += query_recall
		
		meanRecall /= number_queries

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here

		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k) 

		if precision == 0 and recall == 0:
			return 0
	
		fscore = 2*precision*recall/(precision + recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		meanFscore = 0
		num_queries = len(query_ids)
		for idx in range(num_queries):
			query_id = int(query_ids[idx])
			query_true_doc_IDs = []
			for qrel in qrels:
				if int(qrel['query_num']) == query_id:
					query_true_doc_IDs.append(int(qrel['id']))
		
			query_fscore = self.queryFscore(doc_IDs_ordered[idx], query_id, query_true_doc_IDs, k)
			meanFscore += query_fscore
			
		meanFscore /= num_queries



		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : list 
			The list of dicts (Extra added - qrels needed)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		DCG = 0.0
		for i in range(1, min(k, len(query_doc_IDs_ordered))+1):
			# find relevance of document with query
			rel = 0
			for qrel in qrels:
				if (int(qrel['query_num']) == query_id) and (int(qrel['id']) == int(query_doc_IDs_ordered[i-1])):
					rel = 5 - int(qrel['position'])
			DCG += rel/np.log2(i+1)
        
		#print(DCG)
		true_rel = []
		for i in range(0, len(true_doc_IDs)):
			# find relevance of document with query
			rel = 0
			for qrel in qrels:
				if int(qrel['query_num']) == query_id and int(qrel['id']) == true_doc_IDs[i]:
					rel = 5-int(qrel['position'])
			true_rel.append(rel)
		
		true_rel = np.array(true_rel, dtype= int)
		true_rel = -np.sort(-true_rel)
		IDCG = 0.0
		for i in range(min(k, len(true_rel))):
			rel = true_rel[i]
			IDCG += rel/np.log2(i+2)
		
		nDCG = DCG/IDCG
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		meanNDCG = 0
		num_queries = len(query_ids)
		for idx in range(num_queries):
			# for each query id, we would want to find the true_doc_IDs
			query_id = int(query_ids[idx])
			query_true_doc_IDs = []
			for qrel in qrels:
				if int(qrel['query_num']) == query_id:
					query_true_doc_IDs.append(int(qrel['id']))
			
			# NDCG for a particular query
			query_NDCG = self.queryNDCG(doc_IDs_ordered[idx], query_id, query_true_doc_IDs, qrels, k)
			meanNDCG += query_NDCG
		
		meanNDCG /= num_queries

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		if len(true_doc_IDs) == 0:
			return 0
		relevant_docs=0
		avgPrecision = 0
		for idx in range(min(k, len(query_doc_IDs_ordered))):
			doc = int(query_doc_IDs_ordered[idx])
			if int(doc) in true_doc_IDs:
				avgPrecision += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, idx + 1)
				relevant_docs+=1
		
		if relevant_docs != 0:
			avgPrecision /= relevant_docs
		else:
			avgPrecision = 0

		return avgPrecision



	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries
        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value
        Returns
        -------
        float
            The MAP value as a number between 0 and 1
        """
		num_queries = len(query_ids)
        
        # No need to check or resize since we've fixed the size in __init__
		meanAveragePrecision = 0
        
		for idx in range(num_queries):
            # For each query id, find the true_doc_IDs
			query_id = int(query_ids[idx])
			query_true_doc_IDs = []
            
			for qrel in q_rels:
				if int(qrel['query_num']) == query_id:
					query_true_doc_IDs.append(int(qrel['id']))
            
            # Compute average precision and store in the matrix
			ap = self.queryAveragePrecision(doc_IDs_ordered[idx], query_id, query_true_doc_IDs, k)
			self.precision_matrix[idx, k] = ap
			meanAveragePrecision += ap
        
		if num_queries > 0:
			meanAveragePrecision /= num_queries
        
		return meanAveragePrecision

	def get_precision_matrix(self):
		"""
        Returns the precision matrix containing MAP values for each query at each k
        
        Returns
        -------
        numpy.ndarray
            A 2D matrix where rows represent queries and columns represent k values
        """
		return self.precision_matrix
