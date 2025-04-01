import streamlit as st
import google.generativeai as genai
from vector_store import create_vector_store
from loader import pdfloader_chunker
from qa import QuestionAnswer

genai.configure(api_key="AIzaSyC-45MSDZKaXsINcdEf3gwx8ozNnypdeLw")

# Page title
st.set_page_config(page_title='🦜🔗 Ask the PDF App')
st.title('🦜🔗 Ask the Doc App')

with st.sidebar:
    st.header("Please upload a PDF file.")
    uploaded_file = st.file_uploader('Choose a PDF file', type='pdf')   

if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'qa' not in st.session_state:
    st.session_state.qa = None

if uploaded_file is not None:
    if st.session_state.current_file_name != uploaded_file.name:
        if "messages" in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Please ask my anything about your file."}]

        storer = create_vector_store()
        
        st.session_state.current_file_name = uploaded_file.name
        all_chunks = pdfloader_chunker(uploaded_file)
        
        from uuid import uuid4
        uuids = [str(uuid4()) for _ in range(len(all_chunks))]
        
        storer.add_texts(texts=all_chunks, ids=uuids)
        st.session_state.vector_store = storer

        st.session_state.qa = QuestionAnswer(storer, all_chunks)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Please ask my anything about your file."}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        response = st.session_state.qa.get_answer(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

else:
    st.warning("Please upload your PDF file!!")
