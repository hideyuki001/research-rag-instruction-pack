from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

import numpy as np

# ---------------------------
# Minimal RAG setup
# ---------------------------
def build_rag_chain(doc_path=None):
    # Document loading
    if doc_path.endswith(".pdf"):
        loader = PyPDFLoader(doc_path)
    else:
        loader = TextLoader(doc_path)
    docs = loader.load()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Embeddings + Vector DB
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)

    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0)

    # RAG pipeline
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa


def rag(doc_path=None, question="What are the key points?"):
    qa = build_rag_chain(doc_path)
    result = qa.invoke({"query": question})
    return result


# ---------------------------
# EUQS Evaluation
# ---------------------------
def compute_euqs_v2(metrics: dict):
    """
    metrics = {
      "base_confidence": 0.8,
      "delta_s": 0.75,
      "rope_average": 0.7,
      ...
    }
    """
    base_score = np.mean([
        metrics["base_confidence"],
        metrics["delta_s"],
        metrics["rope_average"],
        metrics["refiner_score"],
        metrics["cn_jp_srank"],
        metrics["temporal_stability"],
        metrics["erdf_alignment"]
    ])
    adjusted = base_score * metrics["context_coherence_factor"] * metrics["ambient_consistency_factor"]
    return round(adjusted, 3)


def quality_gate(score: float):
    if score < 0.70:
        return "degrade"
    elif score < 0.85:
        return "refine"
    else:
        return "ship"
