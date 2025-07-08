import os
import json
from flask import Flask, request, Response, render_template
from llama_cpp import Llama
from langchain_core.prompts import ChatPromptTemplate
from textwrap import dedent
import logging
import time
import psutil
import subprocess

# Assuming you have these modules from your project
import config
from intent_classifier import IntentClassifier
from rag_setup import RagSetup


# --- Flask App Initialization ---
app = Flask(
    __name__,
    static_folder=config.FLASK_APP_STATIC_FOLDER,
    template_folder=config.FLASK_APP_TEMPLATE_FOLDER,
)

# --- Logging Setup ---
config.setup_logging()
logger = logging.getLogger(__name__)
logger.info("Logging started...")

# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)  # or logging.CRITICAL to suppress all

# --- Global Variables for Models and Services ---
llm_model = None
classifier_obj = None
rag_obj = None

# --- Inference Logic ---
import time

def stream_inference_generator(user_prompt):
    """
    Generator function to stream response tokens using Llama.cpp.
    Includes timing and performance logging.
    """

    start_time = time.time()
    token_count = 0

    try:
        for chunk in llm_model(prompt=user_prompt, max_tokens=512, temperature=0.8, top_p=0.9, stream=True):
            token_start_time = time.time()

            token = chunk["choices"][0]["text"]
            token_count += 1

            # Token-wise speed logging
            token_time = time.time() - token_start_time

            yield f"data: {json.dumps({'chunk': token})}\n\n"

        total_time = time.time() - start_time
        logger.info(f"Total tokens: {token_count}, Total time: {total_time:.2f}s, Avg Tokens/Sec: {token_count/total_time:.3f}")

        # Optional: Log CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        logger.info(f"CPU usage: {cpu_percent}%")

        # Optional: Log GPU usage (NVIDIA only)
        try:
            gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,nounits,noheader"]).decode("utf-8").strip()
            logger.info(f"GPU stats: {gpu_info}")
        except Exception as gpu_err:
            logger.warning(f"GPU monitoring failed: {gpu_err}")

    except Exception as e:
        logger.error("Error during Llama.cpp streaming", exc_info=True)
        error_message = f"Error generating response. Please try again later."
        yield f"data: {json.dumps({'error': error_message})}\n\n"

        logger.error(f"Error during Llama.cpp streaming: {e}")
        error_message = f"Error generating response: {e}"
        yield f"data: {json.dumps({'error': error_message})}\n\n"


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main chat interface."""
    return render_template('index.html')

@app.route('/stream_chat', methods=['POST'])
def stream_chat():
    """Handles the streaming chat request from the frontend."""
    user_message = request.json.get('message')
    if not user_message:
        return Response(f"data: {json.dumps({'error': 'Please enter a message.'})}\n\n", mimetype='text/event-stream', status=400)

    logger.info(f"Received user message: {user_message}")

    def generate_response():
        # 1. Classify Intent
        intent = classifier_obj.classify_query(user_message)
        logger.info(f"Classified intent: {intent}")

        context = ""
        unique_sources = []
        prompt_template_str = ""
        final_prompt = user_message

        # 2. Handle different intents
        if intent in ("INTENT1_REALTIME", "INTENT2_ANALYTICAL"):
            yield f"data: {json.dumps({'chunk': 'This feature is currently under development.'})}\n\n"

        elif intent == "INTENT3_FAULTINFO":
            try:
                context, _ = rag_obj.retrieve_from_collection(config.FAULT_INFO_COLLECTION, user_message)

                prompt_template_str = dedent("""
                    You are an expert fault-diagnosis assistant for industrial systems.
                    You must answer the user's question using ONLY the information provided in the context below.
                    Do not use any outside knowledge, assumptions, or fabrications. If the answer is not contained in the context, respond with:
                    "The provided context does not contain sufficient information to answer this question."

                    The context may contain technical logs, fault codes, descriptions, and recommended actions. You should analyze them carefully before responding.

                    Context (Top 3 retrieved rows from the table):
                    {context}

                    Question:
                    {question}

                    Answer (based only on the context above):
                """)

            except Exception as e:
                logger.error(f"Error retrieving from FAULT_INFO collection: {e}")
                yield f"data: {json.dumps({'error': f'Error fetching fault info: {e}'})}\n\n"
                return

        elif intent == "INTENT4_DOMAININFO":
            try:
                context, unique_sources = rag_obj.retrieve_from_collection(config.DOMAININFO_COLLECTION, user_message)
                prompt_template_str = dedent("""
                    You are a helpful assistant. Use the provided context to answer the question.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:
                """)
            except Exception as e:
                logger.error(f"Error retrieving from DOMAININFO collection: {e}")
                # Fallback to general model if retrieval fails
                context = ""

        # 3. Format the prompt if a template was chosen
        if prompt_template_str and context:
            prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
            formatted_messages = prompt_template.format_messages(context=context, question=user_message)
            final_prompt = formatted_messages[0].content
            logger.info(f"Using RAG-based prompt.\nContex Retrieved: {context}")

        # 4. Stream the LLM response
        for token_data in stream_inference_generator(final_prompt):
            yield token_data

        # 5. Send sources at the end if they exist
        if unique_sources:
            sources_data = {"sources": [f"Source [{i+1}]: {src}" for i, src in enumerate(unique_sources)]}
            print(sources_data)
            yield f"data: {json.dumps(sources_data)}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'status': 'DONE'})}\n\n"

    return Response(generate_response(), mimetype='text/event-stream')

@app.route('/frontend_log', methods=['POST'])
def frontend_log():
    data = request.get_json()
    logger.info(f"[FRONTEND LOG] {data}")
    return {"status": "logged"}, 200

# --- Application Initialization ---
def initialize_services():
    """Initializes the LLM and other services."""
    global llm_model, classifier_obj, rag_obj
    logger.info("Starting service initialization...")

    try:
        llm_model = Llama(
            model_path=config.TINYLLAMA_MODEL_PATH, 
            n_ctx=4096,
            n_gpu_layers=100, 
            verbose=False)
        logger.info("LLM model initialized successfully.")
    except Exception as e:
        logger.critical("Failed to initialize LLM model.", exc_info=True)
        exit(1)

    try:
        classifier_obj = IntentClassifier()
        logger.info("Intent classifier initialized successfully.")
    except Exception as e:
        logger.critical("Failed to initialize intent classifier.", exc_info=True)
        exit(1)

    try:
        rag_obj = RagSetup()
        logger.info("RAG setup initialized successfully.")
    except Exception as e:
        logger.critical("Failed to initialize RAG setup.", exc_info=True)
        exit(1)

    logger.info("All services initialized successfully.")



if __name__ == "__main__":
    initialize_services()
    # use_reloader=False is important to prevent re-initialization on each change in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)