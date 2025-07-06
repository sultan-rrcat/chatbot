import os
import json
from flask import Flask, request, Response, render_template
from llama_cpp import Llama
from langchain_core.prompts import ChatPromptTemplate
from textwrap import dedent
import logging

# Assuming you have these modules from your project
import config
from intent_classifier import IntentClassifier
from rag_setup import RagSetup

# --- Flask App Initialization ---
app = Flask(
    __name__,
    static_folder=r"C:\Users\offic\Documents\chatbot\app_home\frontend\static",
    template_folder=r'C:\Users\offic\Documents\chatbot\app_home\frontend\templates',
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
def stream_inference_generator(user_prompt):
    """
    Generator function to stream response tokens using Llama.cpp.
    """
    try:
        # Llama.cpp's stream method returns a generator
        for chunk in llm_model(prompt=user_prompt, max_tokens=1024, temperature=0.7, top_p=0.9, stream=True):
            token = chunk["choices"][0]["text"]
            # Yield each token formatted for Server-Sent Events (SSE)
            yield f"data: {json.dumps({'chunk': token})}\n\n"
    except Exception as e:
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
                    You are a helpful assistant. Use the following context to answer the question.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:
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
            logger.info(f"Using RAG-based prompt: {final_prompt}")

        # 4. Stream the LLM response
        for token_data in stream_inference_generator(final_prompt):
            yield token_data

        # 5. Send sources at the end if they exist
        if unique_sources:
            sources_data = {"sources": [f"Source [{i+1}]: {src}" for i, src in enumerate(unique_sources)]}
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
    logger.info("Initializing services...")
    try:
        llm_model = Llama(model_path=config.PHI3_MODEL_PATH, n_ctx=8096, verbose=False, n_gpu_layers=1000)
        classifier_obj = IntentClassifier()
        rag_obj = RagSetup()   
        logger.info("All services initialized successfully!")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR during initialization: {e}", exc_info=True)
        # Exit if critical components fail to load
        exit(1)


if __name__ == "__main__":
    initialize_services()
    # use_reloader=False is important to prevent re-initialization on each change in debug mode
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)