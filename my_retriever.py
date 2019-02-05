#==============================================================================
# Importing
import math

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self, index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        self.numDocs = Retrieve.get_doc_set(self)

    # Method performing retrieval for specified query. Gathers candidate docids then calls appropriate function
    # Returns a list of 10 docids. 
    def forQuery(self, query):
        candidates = Retrieve.get_candidates(self, query) #get all docids that share 1 term with given query
        if self.termWeighting == 'tfidf':
            return(Tfidf.run_tfidf(self, query, candidates))
        elif self.termWeighting == 'tf':
            return(Tf.run_tf(self, query, candidates))
        elif self.termWeighting == 'binary':
            return(Binary.run_binary(self, query, candidates))

    # Adds |D| as an instance variable, only calculated once each query set. T = ~0.01
    def get_doc_set(self):
        full_docs = set()
        for term in self.index:
            full_docs = full_docs | set(self.index[term].keys())
        return(len(full_docs))

    # Pulls all docid numbers from self.index if it shares at least 1 term with the query. Set union, so unique. 
    def get_candidates(self, query):
        candidates = set()
        for term in query:
            if term in self.index:
                candidates = candidates | set(self.index[term].keys())
        return(candidates)

    # Page rank methods are coded in seperate classes to simulate 'pluggable' modules with loose coupling / high cohesion.
    # i.e. you can plugin or take out any class without effecting the others in any way. General functions placed above.
    # The same can be performed with a few if statments, but opted to make it more 'realistic' and also slightly faster. 
    # A compact version can also be found in the document upload. 
    #============================= Binary ===============================#
class Binary:
    def run_binary(self, query, candidates):
        v_lengths, candidate_scores = Binary.binary_get_candidate_scores(self, query, candidates)
        distance_dict = {}
        for doc_id in candidate_scores:
            distance_dict[doc_id] = candidate_scores[doc_id] / math.sqrt(v_lengths[doc_id])
        return(sorted(distance_dict, key=distance_dict.get, reverse=True)[:10]) #Sorts and returns biggest 10

    # Calculates Candidate term scores and the document vector length. 
    def binary_get_candidate_scores(self, query, candidates):
        candidate_scores = dict.fromkeys(candidates, 0) # Dictionary Initialisation
        v_lengths = dict.fromkeys(candidates, 0)
        for term in self.index: #Looping through the full index. 
            for doc_id in self.index[term]: 
                if doc_id in candidates:
                    if term in query: # Term is only kept if it is also in query. 
                        candidate_scores[doc_id] += 1
                    v_lengths[doc_id] += 1
        return v_lengths, candidate_scores

    #=============================   TF   ==============================#
class Tf:
    def run_tf(self, query, candidates):
        query_scores = Tf.get_query_score(self, query)
        v_lengths, candidate_scores = Tf.tf_get_candidate_scores(self, query, candidates, query_scores)
        distance_dict = {}
        for doc_id in candidate_scores:
            distance_dict[doc_id] = candidate_scores[doc_id] / math.sqrt(v_lengths[doc_id])
        return(sorted(distance_dict, key=distance_dict.get, reverse=True)[:10]) #Sorts and returns biggest 10

    # Calculates Candidate term scores and the document vector length. 
    def tf_get_candidate_scores(self, query, candidates, query_scores):
        candidate_scores = dict.fromkeys(candidates, 0) # Dictionary Initialisation
        v_lengths = dict.fromkeys(candidates, 0)
        for term in self.index: #Looping through the full index. 
            for doc_id in self.index[term]: 
                if doc_id in candidates:
                    score = (self.index[term][doc_id]) 
                    if term in query: # Term is only kept if it is also in query. 
                        candidate_scores[doc_id] += (score * query_scores[term])
                    v_lengths[doc_id] += score*score # Document vector score. Not square rooted here. Done later
        return v_lengths, candidate_scores

    # Returns TF scores for the terms in query
    def get_query_score(self, query):
        query_scores = {}
        for term in query:
            if term in self.index:
                query_scores[term] = query[term]  
        return(query_scores)

    #============================= TF:IDF ===============================#
class Tfidf:
    def run_tfidf(self, query, candidates):
        query_scores = Tfidf.get_query_score(self, query)
        v_lengths, candidate_scores = Tfidf.tfidf_get_candidate_scores(self, query, candidates, query_scores)
        distance_dict = {}
        for doc_id in candidate_scores:
            distance_dict[doc_id] = candidate_scores[doc_id] / math.sqrt(v_lengths[doc_id])
        return(sorted(distance_dict, key=distance_dict.get, reverse=True)[:10]) #Sorts and returns biggest 10

    # Calculates Candidate term scores and the document vector length. 
    def tfidf_get_candidate_scores(self, query, candidates, query_scores):
        candidate_scores = dict.fromkeys(candidates, 0) # Dictionary Initialisation
        v_lengths = dict.fromkeys(candidates, 0)
        for term in self.index: #Looping through the full index. 
            idf = (math.log10(self.numDocs/len(self.index[term]))) #IDF score for given Term
            for doc_id in self.index[term]: 
                if doc_id in candidates:
                    score = (self.index[term][doc_id] * idf) # TFIDF score here.
                    if term in query: # Term is only kept if it is also in query. 
                        candidate_scores[doc_id] += (score * query_scores[term])
                    v_lengths[doc_id] += score*score # Document vector score. Not square rooted here. Done later
        return v_lengths, candidate_scores

    # Returns TFIDF scores for the terms in query
    def get_query_score(self, query):
        query_scores = {}
        for term in query:
            if term in self.index: # Query TFIDF scores
                query_scores[term] = (query[term] * (math.log10(self.numDocs/len(self.index[term])))) 
        return(query_scores)
