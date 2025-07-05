# src/extractive_app.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="app_debug.log",   # ton fichier de log
    filemode="w",               # écrase à chaque run, sinon mets "a" pour append
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    # même embeddings que l'indexation
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever()

    while True:
        question = input("\nPose ta question (ou q pour quitter) : ")
        if question.lower() == "q":
            break

        # top 3 chunks les plus proches
        results = retriever.get_relevant_documents(question, k=3)

        if not results:
            print("Aucun résultat trouvé.")
        else:
            for idx, doc in enumerate(results, 1):
                print("\n" + "="*60)
                print(f"Chunk {idx}")
                print(f"Source : {doc.metadata.get('source', 'inconnu')}  page : {doc.metadata.get('page', 'inconnue')}")
                print("-"*60)
                print(doc.page_content.strip())
                print("="*60)

if __name__ == "__main__":
    main()