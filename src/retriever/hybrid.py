class HybridRetriever(): 
    def __init__(self, retrievers, topks):
        assert len(retrievers) == len(topks)
        self.retrievers = retrievers
        self.topks = topks

    def search(self, query):
        related_docs = []

        for id in range(len(self.topks)):
            related_docs.extend(
                self.retrievers[id].search(query, topk=self.topks[id])
            )

        return list(set(related_docs))