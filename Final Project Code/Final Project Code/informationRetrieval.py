#from util import *

# Add your import statements here

import numpy as np
from collections import defaultdict
import math
from math import log10, sqrt
#from numpy.linalg import norm

#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import Word2Vec, LdaModel
from gensim.corpora import Dictionary
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class InformationRetrieval():

    def __init__(self, model_type=0):
        
        # Initialize specific model variables based on model_type
        if model_type == 1:  # TF-IDF
            self._init_tfidf()
        elif model_type == 2:  # LSA
            self._init_lsa()
        elif model_type == 3:  # CRN
            self._init_crn()
        elif model_type == 4:  # Word2Vec
            self._init_word2vec()
        elif model_type == 5:  # Word2Vec with LDA
            self._init_word2vec_lda()
        elif model_type == 6:  # Doc2Vec 
            self._init_doc2vec()
        elif model_type == 7:  # BM25
            self._init_bm25()
        elif model_type == 8:  # BERT 
            self._init_bert()
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be an integer between 0 and 4.")
    
    # TF-IDF
    def _init_tfidf(self):
        """Initialize TF-IDF specific variables"""
        self.index = None
        self.vocabulary = {}
        self.tf_idf = {}
        
    # LSA
    def _init_lsa(self):
        """Initialize LSA specific variables"""
        self.vectorizer = None  # TF-IDF vectorizer
        self.svd = None         # TruncatedSVD model
        self.lsa_matrix = None  # Document vectors in LSA space
        self.docIDs = None      # Document IDs
        
    # CRN
    def _init_crn(self):
        """Initialize CRN specific variables"""
        self.doc_ids = []
        self.vocab = []
        self.tfidf_vectors = []
        self.pmi_matrix = []
        self.doc_vectors = []
        self.term_to_idx = {}
    
    # W2V
    def _init_word2vec(self):
        """Initialize Word2Vec specific variables"""
        self.word2vec_model = None
        self.doc_embeddings = None
        self.docIDs = None
        
    # W2V_LDA
    def _init_word2vec_lda(self):
        """Initialize Word2Vec with LDA specific variables"""
        self.word2vec_model = None
        self.lda_model = None
        self.dictionary = None
        self.doc_embeddings = None
        self.IDs = None
        
    # D2V
    def _init_doc2vec(self):
        """Initialize Doc2Vec specific variables"""
        self.d2v_model = None
        self.doc_embeddings = None
        self.docIDs = None 
    
    # BM25
    def _init_bm25(self):
        """Initialize BM25 specific variables"""
        self.bm25_fitted_model = None
        self.docIDs = None 
        
    # BERT
    def _init_bert(self):
        """Initialize BERT specific variables"""
        self.bert_model = None
        self.doc_embeddings = None
        self.docIDs = None 

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list...             A list of lists of lists where each sub-list is a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        # Store document IDs and documents for later use
        self.docIDs = docIDs
        self.docs = docs

        vocab_dict = {} # Document Frequency - Number of documents that contain each word
        tf_matrix = []  # Term Frequency     - Number of terms in each document

        # For each document
        for doc in docs:
            tf = defaultdict(int)  # Term frequency for current document
            words_seen = set()     # Track unique words in this document for DF counting

            # For each word in each sentence
            for sentence in doc:
                for word in sentence:
                    tf[word] += 1
                    if word not in words_seen:
                        vocab_dict[word] = vocab_dict.get(word, 0) + 1
                        words_seen.add(word)
            tf_matrix.append(tf)   # Stores TF of words, document wise

        # Compute Inverse Document Frequency 
        N = len(docs)
        idf_dict = {word: log10(N / df) for word, df in vocab_dict.items()}

        # Initialize TF-IDF Matrix
        tf_idf = defaultdict(lambda: [0] * N)

        # Compute TF-IDF for each word in each document
        for word in vocab_dict:
            for doc_num in range(N):
                tf_idf[word][doc_num] = tf_matrix[doc_num][word] * idf_dict[word]

        # Compute the L2 norm (Euclidean length) of each document's TF-IDF vector
        doc_norms = [sqrt(sum(tf_idf[word][i] ** 2 for word in tf_idf)) for i in range(N)]

        # Store the computed data structures in the class for use in retrieval
        self.idf_dict = idf_dict
        self.tf_idf = dict(tf_idf)
        self.vocab = vocab_dict
        self.doc_norms = doc_norms

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        # List to store the ranked document IDs for each query
        doc_IDs_ordered = []

        # Retrieve precomputed data from the index
        tf_idf = self.tf_idf          # TF-IDF matrix for all documents
        idf_dict = self.idf_dict      # IDF values for all words
        docIDs = self.docIDs          # Document IDs
        doc_norms = self.doc_norms    # L2 norms of document TF-IDF vectors
        N = len(self.docs)            # Total number of documents

        # For each query
        for query in queries:

            # Compute TF of query
            tf_query = defaultdict(int)
            for sentence in query:
                for word in sentence:
                    if word in self.vocab:   # If a query word is not in self.vocab, it is ignored since it has no IDF value
                        tf_query[word] += 1

            # Compute TF-IDF weights for the query
            query_tf_idf = {}
            for word in tf_query:
                if word in idf_dict:  # Words not in the corpus are ignored as they have no IDF Value
                    query_tf_idf[word] = tf_query[word] * idf_dict[word]

            # Compute the L2 norm of the query's TF-IDF vector
            query_norm = sqrt(sum(val ** 2 for val in query_tf_idf.values()))

            # Dictionary to store cosine similarity scores for each document
            scores = {}
            # Compute cosine similarity between the query and each document
            for i in range(N):
                score = 0
                for word in query_tf_idf:
                    if word in tf_idf:
                        score += tf_idf[word][i] * query_tf_idf[word]

                if doc_norms[i] == 0 or query_norm == 0:
                    scores[docIDs[i]] = 0
                else:
                    scores[docIDs[i]] = score / (doc_norms[i] * query_norm)

            # Sort documents by descending similarity score
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ranked_doc_ids = [doc_id for doc_id, _ in sorted_docs]  # Extract Document IDs
            doc_IDs_ordered.append(ranked_doc_ids)  # Append the ranked list for this query

        return doc_IDs_ordered

    def buildIndex_LSA(self, docs, docIDs):
        """
        Builds the LSA index from preprocessed documents.

        Parameters
        ----------
        docs : list of lists
            Preprocessed documents (tokenized and cleaned)
        docIDs : list
            Unique document identifiers
        """
        
        # Prepare TF-IDF Matrix in the same way as before
        
        # Store document IDs and documents for later use
        self.docIDs = docIDs
        self.docs = docs

        vocab_dict = {} # Document Frequency - Number of documents that contain each word
        tf_matrix = []  # Term Frequency     - Number of terms in each document

        # For each document
        for doc in docs:
            tf = defaultdict(int)  # Term frequency for current document
            words_seen = set()     # Track unique words in this document for DF counting

            # For each word in each sentence
            for sentence in doc:
                for word in sentence:
                    tf[word] += 1
                    if word not in words_seen:
                        vocab_dict[word] = vocab_dict.get(word, 0) + 1
                        words_seen.add(word)
            tf_matrix.append(tf)   # Stores TF of words, document wise

        # Compute Inverse Document Frequency 
        N = len(docs)
        idf_dict = {word: log10(N / df) for word, df in vocab_dict.items()}

        # Create a list of all unique words in vocabulary
        vocab_list = list(vocab_dict.keys())

        # Initialize TF-IDF Matrix 
        tf_idf_matrix = np.zeros((len(vocab_list), N))

        # Compute TF-IDF for each word in each document
        for word_idx, word in enumerate(vocab_list):
            for doc_num in range(N):
                tf_idf_matrix[word_idx, doc_num] = tf_matrix[doc_num][word] * idf_dict[word]

        # Apply Truncated SVD (LSA)
        n_components = min(200, min(tf_idf_matrix.shape) - 1) 
        svd = TruncatedSVD(n_components=n_components)
        lsa_matrix = svd.fit_transform(tf_idf_matrix.T)

        # Store key components for later use
        self.vocab_list = vocab_list
        self.lsa_matrix = lsa_matrix
        self.svd = svd
        self.idf_dict = idf_dict 

    def rank_LSA(self, queries):
        """
        Ranks documents by relevance to each query using LSA.

        Parameters
        ----------
        queries : list of lists
            Tokenized and cleaned queries

        Returns
        -------
        list of lists
            Ordered document IDs for each query
        """
        # List to store ranked document IDs for each query
        doc_IDs_ordered = []

        for query in queries:
            # Compute query TF vector
            query_tf = defaultdict(int)
            for sentence in query:
                for word in sentence:
                    query_tf[word] += 1

            # Create query vector in the original vocabulary space
            query_vec = np.zeros(len(self.vocab_list))
            for word, count in query_tf.items():
                if word in self.vocab_list:
                    word_idx = self.vocab_list.index(word)
                    # Use pre-computed IDF from the original index
                    query_vec[word_idx] = count * (log10(len(self.docs) / (self.idf_dict.get(word, 1))))

            # Project query to LSA space
            query_lsa = self.svd.transform(query_vec.reshape(1, -1))

            # Compute cosine similarities between query and documents in LSA space
            similarities = np.dot(self.lsa_matrix, query_lsa.T).flatten()

            # Normalize similarities to ensure they are between -1 and 1
            norm_query = np.linalg.norm(query_lsa)
            norm_docs = np.linalg.norm(self.lsa_matrix, axis=1)

            # Avoid division by zero
            epsilon = 1e-10
            norm_similarities = np.zeros_like(similarities)

            # Compute Normalized similarities        
            valid_indices = (norm_query > epsilon) & (norm_docs > epsilon)
            norm_similarities[valid_indices] = similarities[valid_indices] / (norm_query * norm_docs[valid_indices])

            # Create list of (doc_id, similarity) and sort
            doc_scores = list(zip(self.docIDs, norm_similarities))
            sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
            
            # Extract document IDs
            ranked_doc_ids = [doc_id for doc_id, _ in sorted_docs]
            doc_IDs_ordered.append(ranked_doc_ids)

        return doc_IDs_ordered
        
    def buildIndex_word2vec_LDA(self, docs, docIDs, word2vec_dims=400, lda_topics=10, 
                  word2vec_lr=0.03, word2vec_epochs=500):
        """
        Builds hybrid Word2Vec + LDA document index
        """
        self.IDs = docIDs
        
        # Word2Vec training
        print(f'Building Word2Vec model (dims={word2vec_dims})')
        tokenized_docs = [[word for sublist in doc for word in sublist] for doc in docs]
        self.word2vec_model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=word2vec_dims,
            window=2,
            workers=4,
            sg=1
        )
        self.word2vec_model.train(tokenized_docs, total_examples=len(tokenized_docs), 
                                epochs=word2vec_epochs, start_alpha=word2vec_lr)

        # LDA training
        print(f'Building LDA model (topics={lda_topics})')
        self.dictionary = Dictionary(tokenized_docs)
        corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]
        self.lda_model = LdaModel(
            corpus=corpus,
            num_topics=lda_topics,
            id2word=self.dictionary,
            passes=15
        )

        # Generate hybrid embeddings
        self.doc_embeddings = self._generate_hybrid_embeddings(
            docs, 
            word2vec_dims, 
            lda_topics
        )
    
    def _generate_hybrid_embeddings(self, docs, w2v_dims, lda_dims):
        """Combines Word2Vec and LDA vectors"""
        embeddings = []
        for doc in docs:
            # Flatten document tokens
            flat_doc = [word for sublist in doc for word in sublist]
            
            # Word2Vec component
            w2v_vec = np.zeros(w2v_dims)
            valid_terms = 0
            for term in flat_doc:
                if term in self.word2vec_model.wv:
                    w2v_vec += self.word2vec_model.wv[term]
                    valid_terms += 1
            if valid_terms > 0:
                w2v_vec /= valid_terms

            # LDA component
            bow = self.dictionary.doc2bow(flat_doc)
            lda_vec = np.array([prob for _, prob in 
                              self.lda_model.get_document_topics(bow, minimum_probability=0)])
            
            # Combine vectors
            hybrid = np.concatenate((w2v_vec, lda_vec))
            embeddings.append(hybrid)
            
        return embeddings
        
    def rank_word2vec_LDA(self, queries):
        """
        Ranks documents using hybrid Word2Vec+LDA similarity
        """
        ranked_doc_IDs = []
        query_embeddings = self._generate_hybrid_embeddings(
            queries,
            self.word2vec_model.vector_size,
            self.lda_model.num_topics
        )

        for q_vec in query_embeddings:
            similarities = [cosine_similarity([q_vec], [doc_vec])[0][0] 
                          for doc_vec in self.doc_embeddings]
            ranked_indices = np.argsort(similarities)[::-1]
            ranked_doc_ids = [self.IDs[i] for i in ranked_indices]
            ranked_doc_IDs.append(ranked_doc_ids)
            
        return ranked_doc_IDs

    def buildIndex_CRN(self, docs, doc_ids):
        """Optimized index construction with PMI pre-calculation"""
        print("Build Index")
        self.doc_ids = doc_ids
        num_docs = len(docs)
        
        # Build vocabulary and document frequencies
        doc_freq = defaultdict(int)
        term_docs = defaultdict(set)
        
        # First pass: collect term statistics
        for doc_idx, doc in enumerate(docs):
            term_counts = defaultdict(int)
            for sentence in doc:
                for term in sentence:
                    term_counts[term] += 1
                    term_docs[term].add(doc_idx)
            
            for term in term_counts:
                doc_freq[term] += 1
        
        # Create vocabulary list and mapping
        self.vocab = list(doc_freq.keys())
        self.term_to_idx = {term: idx for idx, term in enumerate(self.vocab)}
        vocab_size = len(self.vocab)
        
        # Precompute TF-IDF vectors
        self.tfidf_vectors = np.zeros((num_docs, vocab_size))
        for doc_idx, doc in enumerate(docs):
            term_counts = defaultdict(int)
            for sentence in doc:
                for term in sentence:
                    term_counts[term] += 1
            
            for term, count in term_counts.items():
                term_idx = self.term_to_idx[term]
                idf = math.log(num_docs / doc_freq[term])
                self.tfidf_vectors[doc_idx, term_idx] = count * idf
        
        # Precompute PMI matrix
        self.pmi_matrix = np.zeros((vocab_size, vocab_size))
        N = num_docs
        
        # Precompute document sets for each term
        term_doc_sets = {term: term_docs[term] for term in self.vocab}
        
        for i, term_i in enumerate(self.vocab):
            docs_i = term_doc_sets[term_i]
            p_i = len(docs_i) / N
            
            for j, term_j in enumerate(self.vocab[i:], start=i):
                docs_j = term_doc_sets[term_j]
                p_j = len(docs_j) / N
                
                # Fast intersection using set lookups
                joint_count = len(docs_i & docs_j)
                p_ij = joint_count / N
                
                if p_ij > 0 and p_i > 0 and p_j > 0:
                    pmi = math.log2(p_ij / (p_i * p_j))
                    self.pmi_matrix[i, j] = pmi
                    self.pmi_matrix[j, i] = pmi  # Symmetric
        
        # Precompute document vectors in PMI space
        self.doc_vectors = np.dot(self.pmi_matrix, self.tfidf_vectors.T).T

    def rank_CRN(self, queries):
        """Efficient ranking using precomputed vectors"""
        print("Rank")
        results = []
        vocab_size = len(self.vocab)
        
        for query in queries:
            # Build query vector
            query_vec = np.zeros(vocab_size)
            term_counts = defaultdict(int)
            
            for sentence in query:
                for term in sentence:
                    term_counts[term] += 1
            
            for term, count in term_counts.items():
                if term in self.term_to_idx:
                    term_idx = self.term_to_idx[term]
                    query_vec[term_idx] = count
            
            # Project query using PMI matrix
            query_proj = np.dot(self.pmi_matrix, query_vec)
            
            # Compute cosine similarities
            similarities = cosine_similarity(
                [query_proj],
                self.doc_vectors
            )[0]
            
            # Sort documents by similarity
            sorted_indices = np.argsort(similarities)[::-1]
            results.append([str(self.doc_ids[i]) for i in sorted_indices])
        #print('results',results)
        return results

    def buildIndex_word2vec(self, docs, docIDs, word2vec_dims=100, word2vec_lr=0.03, word2vec_epochs=100):
        """
        Builds the Word2Vec-based document index.

        Parameters
        ----------
        docs : list
            A list of documents, where each document is a list of sentences, and each sentence is a list of tokens.
        docIDs : list
            A list of document IDs corresponding to the documents.
        """
        self.docIDs = docIDs

        # Merge all sentences in each document into one flat list of words
        merged_docs = [[word for sentence in doc for word in sentence] for doc in docs]

        print(f"Training Word2Vec model with dim={word2vec_dims}, lr={word2vec_lr}, epochs={word2vec_epochs}")
        self.word2vec_model = Word2Vec(
            sentences=merged_docs,
            vector_size=word2vec_dims,
            window=2,
            workers=4,
            sg=1,
            alpha=word2vec_lr
        )
        self.word2vec_model.train(merged_docs, total_examples=len(merged_docs), epochs=word2vec_epochs)

        self.doc_embeddings = self.generate_doc_embeddings(merged_docs, word2vec_dims)
        
    def generate_doc_embeddings(self, docs, dims):
        """
        Generates document embeddings by averaging word vectors.
        
        Parameters
        ----------
        docs : list
            List of documents, where each document is a list of words.
        dims : int
            Dimensionality of word vectors.
            
        Returns
        -------
        list
            List of document embeddings.
        """
        embeddings = []
        for doc in docs:
            doc_vec = np.zeros(dims)
            valid_terms = 0
            
            for term in doc:
                if term in self.word2vec_model.wv:
                    doc_vec += self.word2vec_model.wv[term]
                    valid_terms += 1
                    
            if valid_terms > 0:
                doc_vec /= valid_terms
                
            embeddings.append(doc_vec)
        return embeddings
        
    def rank_word2vec(self, queries):
        """
        Ranks documents based on Word2Vec cosine similarity with queries.

        Parameters
        ----------
        queries : list
            A list of queries, where each query is a list of sentences (which are lists of tokens).

        Returns
        -------
        list
            A list of lists of document IDs ranked by relevance for each query.
        """
        dims = len(self.doc_embeddings[0])
        merged_queries = [[word for sentence in query for word in sentence] for query in queries]

        query_embeddings = self.generate_doc_embeddings(merged_queries, dims)

        ranked_doc_IDs = []
        for query_vec in query_embeddings:
            sims = [cosine_similarity([query_vec], [doc_vec])[0][0] for doc_vec in self.doc_embeddings]
            ranked_indices = np.argsort(sims)[::-1]
            ranked_doc_IDs.append([self.docIDs[i] for i in ranked_indices])
        return ranked_doc_IDs

    def buildIndex_doc2vec(self, docs, docIDs, vector_size=100, window=5, min_count=1, epochs=50, dm=0):
        self.docIDs = docIDs  
        tagged_docs = [TaggedDocument(words=[word for sent in doc for word in sent], tags=[docID]) 
                       for doc, docID in zip(docs, docIDs)]
        tagged_docs = []
        for doc, docID in zip(docs, docIDs):
          words = [word for sentence in doc for word in sentence]
          tagged_docs.append(TaggedDocument(words=words, tags=[docID]))
        
        # Training Doc2Vec model
        self.d2v_model = Doc2Vec(tagged_docs, vector_size=vector_size, window=window, 
                                 min_count=min_count, epochs=epochs, dm=dm)
        
        # To Generate document embeddings
        self.doc_embeddings = np.array([self.d2v_model.dv[docID] for docID in docIDs]) 

    def rank_doc2vec(self, queries):
        ranked_doc_IDs = []
        tokenized_queries = [[word for sentence in query for word in sentence] for query in queries]

        for query in tokenized_queries:
            query_embedding = self.d2v_model.infer_vector(query)
            score = cosine_similarity([query_embedding], self.doc_embeddings)[0] # cosine similarity
            ranked_indices = np.argsort(score)[::-1] #ranking based on score
            ranked_doc_IDs.append([self.docIDs[i] for i in ranked_indices]) #modifying to given doc ids
        
        return ranked_doc_IDs 
    
    def buildIndex_bert(self, docs, docIDs):
        self.bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.doc_embeddings = self.bert_model.encode(docs)
        self.docIDs = docIDs
    
    def rank_bert(self, queries):
        doc_IDs_ordered = []
        query_embeddings = self.bert_model.encode(queries)
        for query_embedding in query_embeddings:
            scores = {}
            for i in range(len(self.docIDs)):
                scores[self.docIDs[i]] = np.dot(query_embedding, self.doc_embeddings[i])
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ranked_doc_ids = [doc_id for doc_id, _ in sorted_docs]
            doc_IDs_ordered.append(ranked_doc_ids)
        return doc_IDs_ordered
    
    def buildIndex_bm25(self, docs, docIDs):
        for i in range(len(docs)):
            docs[i] = [word for sentence in docs[i] for word in sentence]
        self.bm25_fitted_model = BM25Okapi(docs)
        self.docIDs = docIDs
    
    def rank_bm25(self, queries):
        doc_IDs_ordered = []
        for query in queries:
            scores = self.bm25_fitted_model .get_scores(query[0])
            doc_scores = dict(zip(self.docIDs, scores))
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            ranked_doc_ids = [doc_id for doc_id, _ in sorted_docs]
            doc_IDs_ordered.append(ranked_doc_ids)
        return doc_IDs_ordered