from rank_bm25 import BM25Okapi
import re
import numpy as np

class BM25Retriever():
    def __init__(self, chunks):
        self.tokenized_chunks = [
            re.split(r'[\n ]+', chunk) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        self.chunks = chunks

    def search(self, query, topk):
        tokenized_query = re.split(r'[\n ]+', query) 
        
        best_arg = np.argsort(self.bm25.get_scores(query))[-topk:]
        return [self.chunks[arg] for arg in best_arg]