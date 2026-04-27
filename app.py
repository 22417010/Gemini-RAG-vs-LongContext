import streamlit as st
import os
from google import genai
import time
import tiktoken
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd

st.set_page_config(page_title="Gemini RAG vs Long Context", layout="wide")

# العنوان والوصف
st.title("🚀 Gemini Intelligence Benchmarking")
st.markdown("### RAG vs. Long Context Comparison Demo")

# شريط جانبي للإعدادات
with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    api_key = st.text_input("🔑 أدخل Gemini API Key", type="password")
    
    rag_model = st.selectbox("نموذج RAG", ["gemini-1.5-flash"])
    long_context_model = st.selectbox("نموذج Long Context", ["gemini-1.5-pro"])
    
    chunk_size = st.slider("حجم Chunk", 500, 2000, 1000)
    top_k = st.slider("عدد النتائج المسترجعة", 1, 10, 5)

# تحميل الملفات
st.header("📤 رفع الملفات")
uploaded_files = st.file_uploader("اختر ملفات PDF", type="pdf", accept_multiple_files=True)

documents = []
if uploaded_files:
    for file in uploaded_files:
        reader = PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        documents.append({"source": file.name, "text": text})
    st.success(f"✅ تم تحميل {len(documents)} ملف")

# بناء Vector Store
if documents and st.button("🔨 بناء قاعدة البيانات"):
    with st.spinner("جاري بناء Vector Store..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
        texts, metadatas = [], []
        
        for doc in documents:
            chunks = splitter.split_text(doc["text"])
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({"source": doc["source"], "chunk_id": i})
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        st.session_state.vector_store = vector_store
        st.success(f"✅ تم بناء {vector_store.index.ntotal} chunks")

# نموذج المقارنة
st.header("🤖 المقارنة الذكية")
question = st.text_area("اكتب سؤالك:", height=100)

if question and api_key and documents:
    if st.button("🚀 شغّل المقارنة"):
        client = genai.Client(api_key=api_key)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚡ RAG (Flash)")
            with st.spinner("جاري المعالجة..."):
                start = time.time()
                
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": top_k})
                context_chunks = retriever.invoke(question)
                rag_context = "\n\n".join([d.page_content for d in context_chunks])
                
                rag_prompt = f"Answer based ONLY on context:\nContext:\n{rag_context}\n\nQuestion: {question}"
                rag_res = client.models.generate_content(model=rag_model, contents=rag_prompt)
                rag_time = time.time() - start
                
                st.write(rag_res.text)
                st.metric("الوقت المستغرق", f"{rag_time:.2f}s")
        
        with col2:
            st.subheader("🧠 Long Context (Pro)")
            with st.spinner("جاري المعالجة..."):
                start = time.time()
                
                full_context = "\n\n".join([f"Doc {d['source']}: {d['text']}" for d in documents])
                lc_prompt = f"Answer based on all docs:\nDocs:\n{full_context}\n\nQuestion: {question}"
                lc_res = client.models.generate_content(model=long_context_model, contents=lc_prompt)
                lc_time = time.time() - start
                
                st.write(lc_res.text)
                st.metric("الوقت المستغرق", f"{lc_time:.2f}s")
        
        # جدول المقارنة
        st.header("📊 النتائج المقارنة")
        
        def count_tokens(text):
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        
        comparison = {
            "المقياس": ["عدد التوكنز", "الوقت (ثانية)"],
            "RAG": [count_tokens(rag_context), f"{rag_time:.2f}"],
            "Long Context": [count_tokens(full_context), f"{lc_time:.2f}"]
        }
        
        st.dataframe(pd.DataFrame(comparison))
