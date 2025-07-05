# KYC-RAG

Un projet de démonstration de **RAG (Retrieval Augmented Generation)** appliqué à la conformité bancaire (KYC / LCB-FT), avec ingestion de documents PDF, vectorisation dans ChromaDB et requêtes basées sur LangChain.

---

## Objectifs

- Indexer et chunker des documents réglementaires
- Utiliser ChromaDB pour la recherche sémantique
- Intégrer un pipeline OCR pour traiter les PDF scannés
- Fournir une interface Streamlit pour poser des questions

---

## Architecture

- **LangChain**
- **ChromaDB** pour le stockage vectoriel
- **HuggingFace Embeddings** pour la vectorisation
- **Streamlit** comme interface
- **PyPDFLoader / UnstructuredPDFLoader** pour la lecture des PDF
- **pytesseract + pdf2image** pour l’OCR des PDF scannés

---

## Structure du projet

```plaintext
kyc-rag/
│
├── src/
│   ├── ingestion.py
│   ├── ingestion_simple.py
│   ├── multi_process_unstruct.py
│   ├── use_image2pdf.py
│   ├── app.py
│   ├── extractive_app.py
│   ├── lire_chroma_db.py
│   └── test_use_image2pdf.py
│
├── chroma_db/            (base de vecteurs persistée)
├── data/                 (vos fichiers PDF)
├── requirements.txt
└── README.md

## Installation

1. Clonez le projet :

```bash
git clone https://github.com/votre-compte/KYC-RAG.git
cd KYC-RAG

python -m venv .venv_rag
source .venv_rag/bin/activate     # sous Linux/mac
# ou
.venv_rag\Scripts\activate        # sous Windows

pip install -r requirements.txt


---

## Utilisation

```markdown
## Utilisation

### 1️⃣ Lancer l’ingestion

```bash
python src/ingestion.py

streamlit run src/app.py


---

## Points d’attention

```markdown
## Points d’attention

- Pour l’OCR, pytesseract + pdf2image sont utilisés en fallback
- Sur Windows, attention aux limites de `multiprocessing`

## Licence

MIT
Made with ❤️ by Faycal