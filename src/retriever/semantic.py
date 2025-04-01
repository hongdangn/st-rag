class SemanticRetriever():
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(self, query, topk):
        docs = self.vector_store.similarity_search(query, k=topk)
        return [doc.page_content for doc in docs]