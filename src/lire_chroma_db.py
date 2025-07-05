from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

docs = vectordb.get()
print(f"Nombre de documents dans la base : {len(docs['documents'])}")
print("Exemple de chunk :")
print(docs['documents'][0][:500])  # n’affiche que les 500 premiers caractères du premier chunk