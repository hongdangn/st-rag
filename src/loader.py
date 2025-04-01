import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 32,
    length_function = len
)

def pdfloader_chunker(uploaded_file):
    file = PyPDF2.PdfReader(uploaded_file)
    pages = [page.extract_text() for page in file.pages] 

    all_chunks = []
    for page in pages:
        all_chunks.extend(splitter.split_text(page))

    return all_chunks