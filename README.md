# Research RAG Instruction Pack
Research RAG Instruction Pack

[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-orange)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-black)](https://platform.openai.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/hideyuki001/research-rag-instruction-pack/pulls)

🚀 LangChain-based RAG framework for research & education
This repository provides a minimal yet practical implementation of Retrieval-Augmented Generation (RAG) with LangChain.
It integrates the 5P principles and a custom EUQS evaluation framework to ensure quality from the Proof-of-Concept stage.

🔹 Features

5P Principles

Parsimonious — simple but powerful explanations

Pragmatic — outputs ready for practical use

Probabilistic — uncertainty, sources, and timestamps are explicit

Pedagogical — dual explanation (beginner → expert)

Protective — ethics, compliance, risk prevention

Fixed Output Format

Conclusion (TL;DR)

Reasons (3–6 items)

Evidence (sources, timestamp, uncertainty)

Counter-conditions (when the conclusion fails)

Next step (primary sources / actions)

JSON Log

EUQS Quality Evaluation

Metrics: base_confidence, delta_s, rope_average, refiner_score, cn_jp_srank, temporal_stability, erdf_alignment

Adjusted by: context_coherence_factor × ambient_consistency_factor

Quality gate: <0.70=degrade / 0.70–0.85=refine / ≥0.85=ship

Minimal LangChain Setup (LCEL)

Document loading (PDF, text)

Chunking & vectorization (FAISS + OpenAI embeddings)

Retrieval + LLM response (ChatOpenAI)

🔹 Installation
git clone https://github.com/hideyuki001/research-rag-instruction-pack.git
cd research-rag-instruction-pack
pip install -r requirements.txt


requirements.txt example:

langchain
langchain-community
langchain-openai
langchain-text-splitters
faiss-cpu
numpy

🔹 Usage Example
from rag_agent import rag

response = rag.invoke({"question": "What are the key points of this PDF?"})
print(response)


Sample output:

1. Conclusion (TL;DR)
2. Reasons
3. Evidence (sources, timestamp, uncertainty)
4. Counter-conditions
5. Next step
6. JSON log

🔹 EUQS Example
from rag_agent import compute_euqs_v2, quality_gate

metrics = {
  "base_confidence": 0.8,
  "delta_s": 0.75,
  "rope_average": 0.7,
  "refiner_score": 0.8,
  "cn_jp_srank": 0.72,
  "temporal_stability": 0.65,
  "erdf_alignment": 0.7,
  "context_coherence_factor": 1.0,
  "ambient_consistency_factor": 0.95
}

score = compute_euqs_v2(metrics)
print(score, quality_gate(score))

🔹 Roadmap

v1.0: Minimal RAG + EUQS evaluation

v1.1: Translation memory retrieval mode

v2.0: LangSmith integration + AWS deployment

🔹 License

MIT License (c) 2025 Hideyuki Okabe

🔹 Related Work

QA Synth Pro v2.0

Chrono Vision Framework (in progress)
