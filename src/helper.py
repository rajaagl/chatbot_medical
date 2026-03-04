# src/helper.py - Version avec HuggingFace locale (pas de PyTorch)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
import os

# Désactiver le téléchargement automatique de PyTorch
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'

# Extract Data From the PDF File
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    # Utilise HuggingFaceEmbeddings avec configuration spéciale pour éviter PyTorch
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings