import os
import torch
from flask import Flask, request, render_template, jsonify, Response
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
import logging
import re
from collections import deque
import sys
import shutil
from threading import Thread
import json
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader, PyPDFLoader

# --- Configure logging ---
LOG_FILE = "app_chatbot.log"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

app = Flask(
    __name__,
    static_folder=r"frontend\static",
    template_folder=r'C:\Users\Administrator\Desktop\chatbot\backend\kshitij_app\frontend\templates'
)

# --- Global Variables for Models and DB ---
llm_model = None
embedding_model = None
chroma_client = None
chroma_collection = None
chat_history = deque(maxlen=5)
is_initialized = False

# --- Configuration ---
DATA_DIR = r"C:\Users\Administrator\Desktop\chatbot\backend\kshitij_app\data"
CHROMA_DB_PATH = r"C:\Users\Administrator\Desktop\chatbot\backend\kshitij_app\chroma_db"
LLM_MODEL_NAME = r"C:\Users\Administrator\Desktop\chatbot\backend\models\Phi-3.1-mini-128k-instruct-IQ2_M.gguf"
EMBEDDING_MODEL_NAME = r"C:\Users\Administrator\Desktop\chatbot\backend\models\allminilm"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Data Ingestion Function (Unchanged) ---
def ingest_documents(data_dir, chroma_collection, embedding_model):
    """
    Recursively processes .txt, .pdf, .json documents, extracts & chunks text,
    generates embeddings, and stores them in the ChromaDB collection.
    """
    logger.info(f"Starting recursive document ingestion from {data_dir}...")
    documents_processed = 0
    chunks_added = 0

    if not os.path.exists(data_dir):
        logger.error(f"Data directory '{data_dir}' does not exist. Please create it and add documents.")
        return

    supported_extensions = [".txt", ".pdf", ".json"]

    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            _, file_extension = os.path.splitext(file_path)

            if file_extension.lower() not in supported_extensions:
                logger.info(f"Skipping unsupported file: {file_path}")
                continue

            logger.info(f"Processing file: {file_path}")

            try:
                # ---- Handle JSON ----
                if file_extension.lower() == ".json":
                    loader = JSONLoader(file_path, jq_schema=".", text_content=False)
                    docs = loader.load()
                    formatted_docs = []

                    for doc in docs:
                        data = json.loads(doc.page_content)
                        text = (f"System: {data[0].get('system')}\n"
                                f"Description: {data[0].get('description')}\n"
                                f"Severity: {data[0].get('severity')}\n"
                                f"Cause: {data[0].get('cause')}\n"
                                f"Solution: {data[0].get('solution')}\n")
                        doc.page_content = text
                        formatted_docs.append(doc)

                    for i, doc in enumerate(formatted_docs):
                        chunk = doc.page_content.strip()
                        if not chunk:
                            continue
                        embedding = embedding_model.encode(chunk).tolist()
                        doc_id = f"{file}_chunk_{i}"
                        chroma_collection.add(
                            embeddings=[embedding],
                            documents=[chunk],
                            metadatas=[{"source": file, "chunk_id": i}],
                            ids=[doc_id]
                        )
                        chunks_added += 1
                    documents_processed += 1
                    logger.info(f"Finished processing {file} (JSON), added {len(formatted_docs)} chunks.")

                # ---- Handle TXT and PDF ----
                elif file_extension.lower() in [".txt", ".pdf"]:
                    # Extract raw text
                    if file_extension.lower() == ".txt":
                        with open(file_path, "r", encoding="utf-8") as f:
                            full_text = f.read()
                    elif file_extension.lower() == ".pdf":
                        reader = PdfReader(file_path)
                        full_text = ""
                        for page in reader.pages:
                            full_text += page.extract_text() or ""

                    full_text = re.sub(r'\s+', ' ', full_text).strip()
                    if not full_text:
                        logger.warning(f"No text extracted from {file}. Skipping.")
                        continue

                    # Chunking
                    text_chunks = []
                    start = 0
                    while start < len(full_text):
                        end = start + CHUNK_SIZE
                        chunk = full_text[start:end]
                        text_chunks.append(chunk)
                        start += CHUNK_SIZE - CHUNK_OVERLAP

                    if not text_chunks:
                        logger.warning(f"No chunks generated for {file}. Skipping.")
                        continue

                    for i, chunk in enumerate(text_chunks):
                        chunk = chunk.strip()
                        if not chunk:
                            continue
                        embedding = embedding_model.encode(chunk).tolist()
                        doc_id = f"{file}_chunk_{i}"
                        chroma_collection.add(
                            embeddings=[embedding],
                            documents=[chunk],
                            metadatas=[{"source": file, "chunk_id": i}],
                            ids=[doc_id]
                        )
                        chunks_added += 1
                    documents_processed += 1
                    logger.info(f"Finished processing {file} ({file_extension.upper()}), added {len(text_chunks)} chunks.")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"Document ingestion complete. Processed {documents_processed} documents, added {chunks_added} chunks.")


# --- RAG Logic for Streaming (REWORKED FOR LLAMACPP) ---
def stream_rag_response_generator(query):
    """
    Generator function to stream RAG response tokens using LlamaCpp.
    """
    logger.info(f"User query for streaming: {query}")

    if embedding_model is None or llm_model is None:
        logger.error("Models are not initialized for streaming.")
        yield f"data: {json.dumps({'error': 'Chatbot not ready.'})}\n\n"
        return

    # 1. Embed the query
    query_embedding = embedding_model.encode(query).tolist()

    # 2. Retrieve relevant documents from ChromaDB
    try:
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=['documents']
        )
        retrieved_docs = results['documents'][0] if results and results['documents'] else []
        logger.info(f"Retrieved {len(retrieved_docs)} documents for streaming.")
    except Exception as e:
        logger.error(f"Error retrieving from ChromaDB during streaming: {e}")
        retrieved_docs = []

    context = "\n".join(retrieved_docs)
    history_context = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in chat_history])

    # 3. Construct the prompt manually for Phi-3
    prompt_parts = []
    prompt_parts.append("<|system|>\nYou are a helpful assistant specialized in accelerator physics. Answer the user's questions based on the provided context. If the answer is not related to accelerator physics then you will act like you are the best, state of the art LLM in the world.<|end|>")
    
    if history_context:
        prompt_parts.append(f"<|system|>\nPrevious conversation:\n{history_context}<|end|>")
    
    user_content = ""
    if context:
        user_content += f"Context: {context}\n\n"
    user_content += f"Question: {query}"
    prompt_parts.append(f"<|user|>\n{user_content}<|end|>")

    prompt_parts.append("<|assistant|>") # This is crucial to prompt the model for a response

    final_prompt = "\n".join(prompt_parts)
    logger.info(f"Constructed LLM prompt for streaming:\n{final_prompt[:1000]}...")

    # 4. Generate response using LlamaCpp's stream method
    full_raw_response = ""
    try:
        for chunk in llm_model.stream(final_prompt):
            full_raw_response += chunk
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
    except Exception as e:
        logger.error(f"Error during streaming LLM inference: {e}")
        yield f"data: {json.dumps({'error': f'Error during streaming: {e}'})}\n\n"
        
    finally:
        # Clean the final response for chat history
        clean_response = full_raw_response.strip()

        if "Question:" in clean_response:
            clean_response = clean_response.split("Question:")[0].strip()

        if not clean_response or clean_response in ['.', '!', '?']:
            clean_response = "I couldn't find a direct answer to that in the documents. Can you please rephrase?"

        # Update chat history with the clean response
        chat_history.append({"user": query, "assistant": clean_response})
        logger.info(f"Assistant final response: {clean_response}")

        # Send a final event signaling completion
        yield f"data: {json.dumps({'status': 'DONE', 'clean_response': clean_response})}\n\n"


# --- Flask Routes (Unchanged) ---
@app.route('/')
def index():
    """Renders the main chat interface."""
    return render_template('index.html')

@app.route('/stream_chat', methods=['POST'])
def stream_chat_route():
    """Handles streaming chat requests."""
    global is_initialized
    if not is_initialized:
        logger.warning("Streaming chat request received before initialization is complete.")
        return Response(f"data: {json.dumps({'error': 'The server is still starting up. Please wait a moment and try again.'})}\n\n", mimetype='text/event-stream', status=503)

    user_message = request.json.get('message')
    if not user_message:
        return Response(f"data: {json.dumps({'error': 'Please enter a message.'})}\n\n", mimetype='text/event-stream', status=400)

    return Response(stream_rag_response_generator(user_message), mimetype='text/event-stream')

# --- Application Initialization Function (REWORKED) ---
def initialize_models_and_db():
    """
    Initializes LLM, embedding model, and ChromaDB.
    Also triggers data ingestion if ChromaDB is empty.
    Sets is_initialized flag upon success.
    """
    global llm_model, embedding_model, chroma_client, chroma_collection, is_initialized

    logger.info("Initializing models and ChromaDB...")

    if os.path.exists(CHROMA_DB_PATH):
        logger.info(f"Deleting existing ChromaDB directory: {CHROMA_DB_PATH}")
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            logger.info("ChromaDB directory deleted successfully.")
        except Exception as e:
            logger.error(f"Error deleting ChromaDB directory: {e}")
            sys.exit(1)

    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed to load embedding model: {e}. Exiting.")
        sys.exit(1)

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        chroma_collection = chroma_client.get_or_create_collection(name="accelerator_papers")
        logger.info(f"ChromaDB initialized at '{CHROMA_DB_PATH}'. Collection '{chroma_collection.name}' ready.")
        
        if chroma_collection.count() == 0:
            logger.info("ChromaDB collection is empty. Starting data ingestion...")
            ingest_documents(DATA_DIR, chroma_collection, embedding_model)
        else:
            logger.info(f"ChromaDB collection already contains {chroma_collection.count()} documents. Skipping ingestion.")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed to initialize ChromaDB: {e}. Exiting.")
        sys.exit(1)

    try:
        logger.info("Loading LlamaCpp model...")
        llm_model = LlamaCpp(
            model_path=LLM_MODEL_NAME,
            n_ctx=4096,           # Context window size
            max_tokens=2048,      # Max tokens to generate
            temperature=0.7,
            top_p=0.95,
            verbose=False,
            streaming=True        # Enable streaming
        )
        logger.info(f"LLM '{LLM_MODEL_NAME}' loaded successfully using LlamaCpp.")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed to load LLM: {e}. Exiting.")
        sys.exit(1)

    is_initialized = True
    logger.info("All models and database initialized successfully!")


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    initialize_models_and_db()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
