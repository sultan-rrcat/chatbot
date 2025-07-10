import os
import json
from flask import Flask, request, Response, render_template
from textwrap import dedent
import logging
import time
import psutil
import subprocess
from threading import Thread
import date_parser

# Hugging Face Transformers and PyTorch imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig

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
logger.info("📝 Logging has been successfully set up.")

# --- Global Variables for Models and Services ---
model = None
tokenizer = None
device = None
classifier_obj = None
rag_obj = None

# --- Inference Logic ---
def stream_inference_generator(messages):
    """
    Generator function to stream response tokens using Hugging Face Transformers.
    It uses a TextIteratorStreamer and runs generation in a separate thread.
    Includes timing and performance logging.
    """
    start_time = time.time()
    token_count = 0

    # Use a streamer for non-blocking, token-by-token generation
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    try:
        # Apply the chat template for the specific model
        # This formats the input conversation correctly
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        # Keyword arguments for the model's generate method
        generation_kwargs = dict(
            input_ids=inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )

        # Run generation in a separate thread to avoid blocking the main thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they become available
        for token in streamer:
            token_count += 1
            yield f"data: {json.dumps({'chunk': token})}\n\n"

        # Wait for the thread to finish
        thread.join()

        total_time = time.time() - start_time
        if total_time > 0:
            logger.info(f"✅ Total Tokens: {token_count}, Total Time: {total_time:.2f}s, Avg Tokens/Sec: {token_count/total_time:.3f}")

        # Optional: Log CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        logger.info(f"🖥️ CPU Usage: {cpu_percent}%")

        # Optional: Log GPU usage (NVIDIA only)
        try:
            gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,nounits,noheader"]).decode("utf-8").strip()
            logger.info(f"📈 GPU Stats: {gpu_info}")
        except Exception as gpu_err:
            logger.warning(f"⚠️ GPU monitoring failed: {gpu_err}")

    except Exception as e:
        logger.error("❌ Error during Transformers streaming", exc_info=True)
        error_message = f"Error generating response. Please try again later."
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

    user_message = date_parser.normalize_dates_in_text(user_message)
    
    def generate_response():
        # 1. Classify Intent
        intent = classifier_obj.classify_query(user_message)

        context = ""
        unique_sources = []
        system_prompt = ""
        
        # This will hold the conversation history for the model
        messages = []

        # 2. Handle different intents
        if intent in ("INTENT1_REALTIME", "INTENT2_ANALYTICAL"):
            yield f"data: {json.dumps({'chunk': 'This feature is currently under development.'})}\n\n"
            # Signal completion even for placeholder messages
            yield f"data: {json.dumps({'status': 'DONE'})}\n\n"
            return

        elif intent == "INTENT3_FAULTINFO":
            try:
                context, _ = rag_obj.retrieve_from_collection(config.FAULT_INFO_COLLECTION, user_message)
                system_prompt = dedent("""
                    You are an expert fault-diagnosis assistant for industrial systems.
                    You must answer the user's question using ONLY the information provided in the context below.
                    Do not use any outside knowledge, assumptions, or fabrications. If the answer is not contained in the context, respond with:
                    "The provided context does not contain sufficient information to answer this question."
                    The context may contain technical logs, fault codes, descriptions, and recommended actions. You should analyze them carefully before responding.
                """)
                messages.append({"role": "system", "content": system_prompt})
                # Add the user's question with context
                user_content_with_context = f"Context:\n{context}\n\nQuestion:\n{user_message}"
                messages.append({"role": "user", "content": user_content_with_context})
                logger.info(f"Using RAG-based prompt for FAULT_INFO. Context Retrieved: {context}")

            except Exception as e:
                logger.error(f"❌ Error retrieving from FAULT_INFO collection: {e}")
                yield f"data: {json.dumps({'error': f'Error fetching fault info: {e}'})}\n\n"
                return

        elif intent == "INTENT4_DOMAININFO":
            try:
                context, unique_sources = rag_obj.retrieve_from_collection(config.DOMAININFO_COLLECTION, user_message)
                system_prompt = "You are a helpful assistant. Use the provided context to answer the question."
                messages.append({"role": "system", "content": system_prompt})
                # Add the user's question with context
                user_content_with_context = f"Context:\n{context}\n\nQuestion:\n{user_message}"
                messages.append({"role": "user", "content": user_content_with_context})
                logger.info("Using RAG-based prompt for DOMAININFO.")

            except Exception as e:
                logger.error(f"❌ Error retrieving from DOMAININFO collection: {e}")
                # Fallback to general model if retrieval fails
                messages.append({"role": "user", "content": user_message})
        
        else: # General chat, no specific intent or RAG
            messages.append({"role": "user", "content": user_message})


        # 4. Stream the LLM response
        for token_data in stream_inference_generator(messages):
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
    logger.info(f"🖥️ [FRONTEND LOG] {data}")
    return {"status": "logged"}, 200

# --- Application Initialization ---
def initialize_services():
    """Initializes the LLM, Tokenizer, and other services."""
    global model, tokenizer, device, classifier_obj, rag_obj
    logger.info("Starting service initialization...")

    try:
        # --- Device Setup ---
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")

        # --- Quantization Configuration (Optional but Recommended) ---
        # Use 4-bit quantization to reduce memory footprint
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model_id = config.MISTRAL_MODEL_PATH
        
        logger.info(f"Loading tokenizer for model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        logger.info(f"Loading model: {model_id}. This may take a while...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config if torch.cuda.is_available() else None, # Only quantize on GPU
            device_map="auto" # Automatically map model layers to available devices
        )
        logger.info(f"✅ LLM Model: {model_id} Initialized Successfully.")

    except Exception as e:
        logger.critical("❌ Failed to initialize LLM model.", exc_info=True)
        exit(1)

    try:
        classifier_obj = IntentClassifier()
        logger.info("✅ Intent Classifier Initialized Successfully.")
    except Exception as e:
        logger.critical("❌ Failed to initialize intent classifier.", exc_info=True)
        exit(1)

    try:
        rag_obj = RagSetup()
        logger.info("✅ RAG Setup Initialized Successfully.")
    except Exception as e:
        logger.critical("❌ Failed to initialize RAG setup.", exc_info=True)
        exit(1)

    logger.info("🎉 All services initialized successfully.")



if __name__ == "__main__":
    initialize_services()
    # use_reloader=False is important to prevent re-initialization on each change in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
