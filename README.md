# 🚀 Gemini Intelligence Benchmarking: RAG vs. Long Context

This project provides a professional framework to compare two dominant AI architectures: **Retrieval-Augmented Generation (RAG)** and **Long Context Windows**, using Google Gemini 1.5 models.

## 📋 Project Overview
The goal is to analyze the trade-offs between:
- **RAG:** Using FAISS vector database and embeddings (Cost-efficient & Fast).
- **Long Context:** Utilizing Gemini's massive 1M+ token window (High Precision & Holistic understanding).

## 🛠️ Tech Stack
- **LLMs:** Google Gemini 1.5 Flash & Pro.
- **Orchestration:** LangChain.
- **Vector DB:** FAISS.
- **Processing:** PyPDF, Tiktoken.
- **UI/Analysis:** Pandas & Rich Terminal.

## 🚀 How to Run on Google Colab
1. Open the `.ipynb` file in this repository.
2. Click on the **"Open in Colab"** badge.
3. Add your `GEMINI_API_KEY` to the secrets or environment variables.
4. Upload your PDF documents and run the cells!

## 📊 Key Metrics Tracked
- **Context Tokens:** How much data is actually sent to the model.
- **Latency:** Time taken for the model to generate a response.
- **Response Quality:** Comparison of groundedness vs. holistic reasoning.

---
*Developed with ❤️ using Google Gemini API*
