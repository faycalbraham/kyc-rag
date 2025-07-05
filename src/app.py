# src/app.py

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="app_debug.log",   # ton fichier de log
    filemode="w",               # écrase à chaque run, sinon mets "a" pour append
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# embeddings identiques
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# récupérateur
retriever = vectordb.as_retriever()

# modèle LLM
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# chaîne RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # simple
    retriever=retriever,
    verbose=True
)

# boucle interactive
while True:
    question = input("Pose ta question (ou q pour quitter) : ")
    if question.lower() == "q":
        break
    # récupération des documents contextuels
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"Question: {question}\n\nContexte:\n{context}"
    print("="*30)
    print("PROMPT ENVOYÉ AU LLM :")
    print(prompt)
    print("="*30)
    result = qa_chain.run(question)
    print(f"Réponse : {result}")
