# src/ingestion.py

from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
from use_image2pdf import extract_text_from_pdf_images
from multi_process_unstruct import run_unstructured_ocr

os.environ["PATH"] += os.pathsep + r"C:\Program Files\poppler\Library\bin"

DATA_DIR = "./data"


def ingest_documents():
    # liste des pdf réellement sur disque
    pdf_files_on_disk = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    
    # embeddings identiques
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # charger ChromaDB existant
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # récupérer la liste des sources déjà indexées
    existing_sources = set()
    try:
        existing_sources = set([
            m["source"]
            for m in vectordb.get()["metadatas"]
        ])
        # print(f"Documents déjà indexés dans la base : {existing_sources}")
    except Exception:
        print("Pas de vectordb existant, premier index.")
    
    # liste des pdf nouveaux
    new_pdfs = [f for f in pdf_files_on_disk if os.path.join(DATA_DIR, f) not in existing_sources]
    print("new_pdfs   ", new_pdfs)
    # liste des pdfs supprimés
    removed_pdfs = existing_sources - set(os.path.join(DATA_DIR, f) for f in pdf_files_on_disk)
    print("removed_pdfs   ", removed_pdfs)


    # supprimer de la collection les documents qui n'existent plus
    to_delete_ids = []  # toujours initialisé

    if removed_pdfs:
        print(f"Suppression des documents retirés du dossier data: {removed_pdfs}")
        to_delete_ids = [
            id_
            for id_, meta in zip(vectordb.get()["ids"], vectordb.get()["metadatas"])
            if meta["source"] in removed_pdfs
        ]



    if to_delete_ids:
        vectordb.delete(ids=to_delete_ids)
        print(f"{len(to_delete_ids)} chunks supprimés de ChromaDB.")
    else:
        print("Aucun chunk à supprimer trouvé dans ChromaDB.")


    # indexer les nouveaux pdfs
    if new_pdfs:
        docs = []
        for file in new_pdfs:
            pdf_path = os.path.join(DATA_DIR, file)
            loader = PyPDFLoader(pdf_path)  # Cas 1: PDF texte natif, PyPDFLoader arrive à le parser
            pages = loader.load_and_split()     
            
            if len(pages) == 0:
                print(f"⚠️ Aucun texte détecté dans {file} → tentative OCR avec UnstructuredPDFLoader...")
                # CAS 2 : OCR haute résolution avec Unstructured
                # PDF image scannée → UnstructuredPDFLoader
                #PDF est une image scannée (pas du texte natif) et que tu veux laisser LangChain + unstructured gérer l’OCR à ta place.
                print("⏳ OCR Unstructured en cours... (patiente)")
                
                pages_content = run_unstructured_ocr(pdf_path, timeout=120)
                if not pages_content:
                    print("⚠️ Unstructured KO → fallback pytesseract")
                    texts = extract_text_from_pdf_images(pdf_path)
                    if texts:
                        from langchain_core.documents import Document
                        pages = [Document(page_content=t, metadata={"source": pdf_path}) for t in texts]
                        print(f"✅ OCR pytesseract a extrait {len(pages)} pages de texte.")
                    else:
                        print(f"❌ Aucun texte récupéré même avec pytesseract.")
                else:
                    # reconstruire proprement
                    from langchain_core.documents import Document
                    pages = [Document(page_content=t, metadata={"source": pdf_path}) for t in pages_content]
                    print(f"✅ OCR Unstructured a extrait {len(pages)} pages depuis {file}")
        
            docs.extend(pages)
                
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        if len(chunks) > 0:
            print(f"Exemple metadata d’un chunk : {chunks[0].metadata}")
            print(f"Nombre de chunks générés : {len(chunks)}")
            vectordb.add_documents(chunks)
            print(f"{len(chunks)} chunks ajoutés pour les fichiers {new_pdfs}")
        else:
            print("Aucun chunk à ajouter (vérifier le contenu des PDF).")
    else:
        print("Aucun nouveau fichier à indexer.")
    
    
    # afficher la liste finale des documents présents dans la base
    all_sources = set([
        m["source"]
        for m in vectordb.get()["metadatas"]
    ])
    print("✅ Liste finale des documents indexés :")
    for s in all_sources:
        print("-", s)

if __name__ == "__main__":
    ingest_documents()