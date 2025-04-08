from retriever import BM25Retriever, SemanticRetriever, HybridRetriever
# from reranker import Ranker
from prompter import formulate_prompt
import google.generativeai as genai

class QuestionAnswer:
    def __init__(self, vector_store, chunks):
        self.retriever = HybridRetriever(
            retrievers=[BM25Retriever(chunks), SemanticRetriever(vector_store)],
            topks=[2, 3]
        )

        # self.reranker = Ranker(topk=3)
        generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 3,
            "max_output_tokens": 2048,
        }

        self.gemini = genai.GenerativeModel(model_name="gemini-2.0-flash",
                                       generation_config=generation_config,)

    def get_final_prompt(self, query):
        final_docs = self.retriever.search(query)
        # final_docs = self.reranker.get_final_docs(query, docs)

        return formulate_prompt(query, final_docs)
    
    def get_answer(self, query):
        final_prompt = self.get_final_prompt(query)

        gemini_response = self.gemini.generate_content(
            final_prompt
        )
        
        return gemini_response.candidates[0].content.parts[0].text
