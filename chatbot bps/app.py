import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os  # Import the 'os' module to handle file paths

load_dotenv(".env")

# Display logo and header
logo_path = "bps.png"

# Check if the logo file exists
if os.path.exists(logo_path):
    st.image(logo_path, width=100)  # Adjust width as needed
else:
    st.warning("Logo file not found. Make sure 'bps_logo.png' is in the same directory as your script.")

st.header("BADAN STATISTIK PROVINSI ACEH")

pdf = st.file_uploader("Upload PDF BPS yang kamu Miliki", type="pdf")

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split menjadi chunk
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

pertanyaan = st.text_input("Tanya Tentang BPS yang sudah diupload")
if pertanyaan:
    # No need to use 'input' here, as 'st.text_input' is already capturing the user's input
    docs = knowledge_base.similarity_search(pertanyaan)

    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=pertanyaan)
    st.write(response)
