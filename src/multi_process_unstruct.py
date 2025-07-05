# multi_process_unstruct.py
from langchain_community.document_loaders import UnstructuredPDFLoader
import multiprocessing
import os
import logging

# configure le logger une seule fois
logging.basicConfig(
    filename="unstructured_ocr.log", 
    filemode="w",   # recrée le fichier à chaque exécution
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def unstructured_worker(pdf_path, queue):
    # corrige le PATH dans le processus enfant
    os.environ["PATH"] += os.pathsep + r"C:\Program Files\poppler\Library\bin"

    try:
        queue.put(f"🟡 Unstructured_OCR démarré sur {pdf_path}")
        loader = UnstructuredPDFLoader(pdf_path, strategy="hi_res")
        pages = loader.load()
        simple_pages = [p.page_content for p in pages]
        queue.put(f"✅ Unstructured_OCR terminé avec {len(simple_pages)} pages")
        queue.put(simple_pages)  # résultat final
        logging.info(f"OCR terminé pour {pdf_path}, {len(simple_pages)} pages extraites")
    except Exception as e:
        queue.put(e)
        logging.error(f"Erreur OCR worker: {e}")

def run_unstructured_ocr(pdf_path, timeout=120):
    """
    Lance le worker Unstructured dans un process séparé.
    Retourne list[str] ou [].
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=unstructured_worker, args=(pdf_path, queue))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        msg = f"⏰ Timeout Unstructured OCR dépassé ({timeout}s)"
        print(msg)
        logging.warning(msg)
        return []

    # lire tout ce qui a été mis dans la queue
    result = None
    while not queue.empty():
        item = queue.get()
        if isinstance(item, Exception):
            print(f"❌ Exception OCR Unstructured : {item}")
            logging.error(f"Exception OCR Unstructured : {item}")
            result = []
        else:
            # on suppose que c'est la liste finale de textes
            result = item
    
    if result is None:
        msg = "⚠️ Queue vide après OCR Unstructured"
        print(msg)
        logging.warning(msg)
        return []
    
    return result