import math, re, sys
import time

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

        if self.termWeighting == 'tf':
            return(Tf.run_tf(self, query, candidates))
        elif self.termWeighting == 'tfidf':
            return(Tfidf.run_tfidf(self, query, candidates))
        elif self.termWeighting == 'binary':
            return(range(0,10))

    # Adds |D| as an instance variable, only calculated once. 
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

    #======================== Binary ==========================#
#class Binary:

    #========================   TF   ==========================#
class Tf:
    def run_tf(self, query, candidates):
        candidate_scores = {}
        for term in self.index: # Loops through the entire index once & looks at each documentID in each term.
            for doc_id in self.index[term]: 
                if doc_id in candidates:
                    if doc_id in candidate_scores: 
                        candidate_scores[doc_id] = candidate_scores[doc_id] + self.index[term][doc_id]
                    else:
                        candidate_scores[doc_id] = self.index[term][doc_id]
        return({key: candidate_scores[key] for key in sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:10]})

    #======================== TF:IDF ==========================#
class Tfidf:
    # Main function - Runs the tfidf page ranker and calls functions.
    # returns an array of ten document ids sorted in decending order
    def run_tfidf(self, query, candidates):
        candidate_scores = Tfidf.get_candidate_scores(self, query, candidates)
        query_scores = Tfidf.get_query_score(self, query)
        distance_dict = {}
        for candidate in candidate_scores:
            distance_dict[candidate] = Tfidf.get_distance(query_scores, candidate, candidate_scores)
        return({key: distance_dict[key] for key in sorted(distance_dict, key=distance_dict.get, reverse=True)[:10]}) #sorts dictionary, returns biggest 10. 

    # Returns a nested dictionary: documentID -> term -> tfidf. A term is ONLY kept if useful for later
    # Also calculates a vector length for each term/document relation and adds it to the same dictionary
    def get_candidate_scores(self, query, candidates):
        candidate_scores = {}
        for term in self.index: # Loops through the entire index once & looks at each documentID in each term.
            idf = (math.log10(self.numDocs/len(self.index[term]))) 
            for doc_id in self.index[term]: 
                if doc_id in candidates:
                    score = (self.index[term][doc_id] * idf) # Tfidf score of the document / term releationship. 
                    if doc_id in candidate_scores: # Note - vector length is NOT square rooted here. get_distance instead.
                        candidate_scores[doc_id]['v_length'] = candidate_scores[doc_id]['v_length'] + (score**2) # A little messy but speeds up process. 
                    else:
                        candidate_scores[doc_id] = {'v_length': (score**2)}
                    if term in query: # ONLY add the term to the dictionary if it is useful later. i.e if it shares the term with query. 
                        candidate_scores[doc_id][term] = score
        return candidate_scores
    
    # Calculates the word values in query.
    # Returns a dictionary term -> tfidf. 
    def get_query_score(self, query):
        query_scores = {}
        v_length = 0
        for term in query:
            if term in self.index:
                query_scores[term] = (query[term] * (math.log10(self.numDocs/len(self.index[term])))) #Tfidf score. 
        return(query_scores)

    # Returns a single float that represents the similarity between the query and given doc
    # Takes the queries tfidf scores dictionary, a documentID integer and the complete candidateID dictionary. 
    def get_distance(query_scores, candidate, candidate_scores):
        distance_dict = {}
        tfidf_vector = 0
        for term in query_scores:
            if term in candidate_scores[candidate] and term != "v_length":
                tfidf_vector += (query_scores[term] * candidate_scores[candidate][term])
        tfidf_vector = (tfidf_vector/math.sqrt(candidate_scores[candidate]["v_length"])) # rooted here to avoid unnecessary loops.  
        return tfidf_vector
