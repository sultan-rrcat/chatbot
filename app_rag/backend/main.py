import os
from llama_cpp import Llama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -----------------------------
# Load Chroma vector store
vectorstore = Chroma(
    persist_directory="new_rag_data/chroma_faultbook_index",
    embedding_function=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True}
    )
)

# -----------------------------
# Load LLaMA.cpp local GGUF model
print("loading llm...")
llm = Llama(
    model_path=r"C:\\Users\\Administrator\\Desktop\\chatbot\\backend\\models\\DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf",
    n_ctx=8192,
    n_batch=64,
    n_threads=8,
    verbose=False
)
print("llm_loaded...")

# -----------------------------
# CLI-based loop for querying while testing
while True:
    user_query = input("\n❓ Your question (or type 'exit'): ").strip()
    if user_query.lower() == 'exit':
        break
    
    # Retrieve top 4 relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    results = retriever.invoke(user_query)

    # Ensure results are Document objects
    if isinstance(results, list) and hasattr(results[0], 'page_content'):
        retrieved_docs = results
    else:
        # If Chroma returns strings instead, wrap in Document manually
        from langchain_core.documents import Document
        retrieved_docs = [Document(page_content=doc) if isinstance(doc, str) else doc for doc in results]

    # Prepare context correctly
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(context)

    # Prepare prompt
    prompt = f"""
You are a helpful fault logbook assistant. Answer the user's query using the following context:

{context}

Question: {user_query}

Answer:
"""

    # Generate using LLaMA.cpp
    output = llm(prompt, max_tokens=512, temperature=0.1, top_p=0.9)

    print("\n💡 Answer:", output["choices"][0]["text"].strip())