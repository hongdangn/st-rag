from FlagEmbedding import FlagReranker
import numpy as np

class Ranker:
    def __init__(self, topk=2):
        self.reranker = FlagReranker(
            'BAAI/bge-reranker-large',
            use_fp16=True,
            devices=["cuda:0"]
        )
        self.topk=topk

    def get_final_docs(self, query, docs):
        queries = [query] * len(docs)
        pairs = [[query, doc] for query, doc in zip(queries, docs)]
        scores = self.reranker.compute_score(pairs, normalize=True)

        best_arg = np.argsort(scores)[-self.topk:]
        return [docs[id] for id in best_arg]


