# src/ingestion.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
#import warnings
#warnings.filterwarnings("ignore")


import os

# chemin vers les PDF
DATA_DIR = "./data"

def ingest_documents():
    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            pages = loader.load_and_split()
            docs.extend(pages)

    # chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")
    vectordb.persist()
    print(f"Nombre de chunks générés : {len(chunks)}")
    print("✅ Vector store créé avec succès dans chroma_db")

if __name__ == "__main__":
    ingest_documents()
 #   input("Press Enter to exit...")