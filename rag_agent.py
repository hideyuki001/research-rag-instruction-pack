# rag_agent.py
"""
Research RAG Instruction Pack
Minimal LangChain RAG implementation + EUQS evaluation
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import numpy as np

# --------------------------
# 1) RAG Setup
# --------------------------

def load_vectorstore(pdf_path="docs.pdf"):
    """Load PDF, split, embed, and store in FAISS vector DB."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    emb = OpenAIEmbeddings(model="text-embedding-3-large")
    vs = FAISS.from_documents(chunks, emb)
    return vs.as_retriever(search_kwargs={"k": 5})


def build_rag_chain(retriever):
    """Build RAG chain with fixed 6-part output."""
    TEMPLATE = """You are a RAG agent for research & education.
Constraints: No investment advice / No profit guarantees.
Always include sources, timestamps, and uncertainty if relevant.

Format output as:
1. Conclusion (TL;DR)
2. Reasons (3–6 items)
3. Evidence (sources, timestamp, uncertainty)
4. Counter-conditions
5. Next step
6. JSON log

Context: {context}
Question: {question}
Answer:"""

    prompt = PromptTemplate.from_template(TEMPLATE)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    combine = create_stuff_documents_chain(llm, prompt)
    rag = create_retrieval_chain(retriever, combine)
    return rag


# --------------------------
# 2) EUQS Evaluation
# --------------------------

def _norm(x):
    return (x - 0.5) / 0.25  # normalize around 0.5

def compute_euqs_v2(metrics: dict):
    """
    Compute EUQS v2 score.
    metrics = {
        base_confidence: float,
        delta_s: float,
        rope_average: float,
        refiner_score: float,
        cn_jp_srank: float,
        temporal_stability: float,
        erdf_alignment: float,
        context_coherence_factor: float,
        ambient_consistency_factor: float
    }
    """
    weights = dict(
        base_confidence=0.20,
        delta_s=0.15,
        rope_average=0.15,
        refiner_score=0.15,
        cn_jp_srank=0.15,
        temporal_stability=0.10,
        erdf_alignment=0.10,
    )

    z = {k: _norm(metrics.get(k, 0.5)) for k in weights}
    s = {k: 1 / (1 + np.exp(-1.5 * z[k])) for k in weights}

    context = np.clip(metrics.get("context_coherence_factor", 1.0), 0.85, 1.05)
    ambient = np.clip(metrics.get("ambient_consistency_factor", 1.0), 0.85, 1.05)

    score = (sum(weights[k] * s[k] for k in weights)) * context * ambient
    return score

def quality_gate(score: float):
    if score >= 0.85:
        return "ship"
    elif score >= 0.70:
        return "refine"
    else:
        return "degrade"


# --------------------------
# 3) Exported Objects
# --------------------------

# Default retriever and RAG chain
retriever = load_vectorstore("docs.pdf")  # PDFは必要に応じて差し替え
rag = build_rag_chain(retriever)
