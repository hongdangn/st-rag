import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import google.generativeai as genai


def embed_chunks(chunks):
    return genai.embed_content(
        model="models/embedding-001",
        content=chunks,
    )["embedding"]

def create_vector_store():
    index = faiss.IndexFlatL2(
        len(
            genai.embed_content(model="models/embedding-001", 
                                content="hello world")["embedding"]
            )
    )

    vector_store = FAISS(
        embedding_function=embed_chunks,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    return vector_store