import config
import logging
import os
import json
import pandas as pd
import numbers
from flask import Flask, request, Response, render_template, jsonify, session
from textwrap import dedent
import time
import psutil
import subprocess
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig

from NL2SQL import NL2SQL
from intent_classifier import IntentClassifier
from rag_setup import RagSetup
import date_parser
import uuid
from sqlalchemy import create_engine, text
from datetime import datetime, timezone

# --- Flask App Initialization ---
app = Flask(
    __name__,
    static_folder=config.FLASK_APP_STATIC_FOLDER,
    template_folder=config.FLASK_APP_TEMPLATE_FOLDER,
)

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "indus-faultbook-assistant")

# --- Logging Setup ---
config.setup_logging()
logger = logging.getLogger(__name__)
logger.info("Logging has been successfully set up.")

# --- Global Variables for Models and Services ---
mistral_base_model = None
mistral_tokenizer = None
device = None
classifier_obj = None
rag_obj = None
nl2sql_obj = None

engine = create_engine(config.SQLALCHEMY_CHATBOT_CONNECTION_STRING)


def ensure_message_alternation(messages):
    # Remove consecutive 'system' or 'user' roles
    cleaned_messages = []
    last_role = None
    for msg in messages:
        if msg["role"] == last_role and msg["role"] in ["system", "user"]:
            continue  # skip duplicate system/user
        cleaned_messages.append(msg)
        last_role = msg["role"]

    # Ensure the alternation pattern
    roles_sequence = [msg["role"] for msg in cleaned_messages if msg["role"] != "system"]
    if len(roles_sequence) > 0 and roles_sequence[0] != "user":
        raise ValueError("First message after system must be from 'user'.")
    for idx in range(1, len(roles_sequence)):
        if roles_sequence[idx] == roles_sequence[idx - 1]:
            raise ValueError(f"Invalid alternation in messages at position {idx}: {roles_sequence}")

    return cleaned_messages


# --- Inference Logic ---
def stream_inference_generator(messages):
    """
    Generator function to stream response tokens using Hugging Face Transformers.
    It uses a TextIteratorStreamer and runs generation in a separate thread.
    Includes timing and performance logging.
    """
    start_time = time.time()
    token_count = 0

    current_process = psutil.Process(os.getpid())

    # Use a streamer for non-blocking, token-by-token generation
    streamer = TextIteratorStreamer(mistral_tokenizer, skip_prompt=True, skip_special_tokens=True)

    try:
        # Apply the chat template for the specific model
        # This formats the input conversation correctly
        inputs = mistral_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        # Keyword arguments for the model's generate method
        generation_kwargs = dict(
            input_ids=inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            temperature=1,
            top_p=0.9,
            top_k=50
        )

        current_process.cpu_percent(interval=None)

        # Run generation in a separate thread to avoid blocking the main thread
        thread = Thread(target=mistral_base_model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they become available
        for token in streamer:
            token_count += 1
            yield f"data: {json.dumps({'chunk': token})}\n\n"

        # Wait for the thread to finish
        thread.join()

        total_time = time.time() - start_time
        if total_time > 0:
            logger.info(f"Total Tokens: {token_count}, Total Time: {total_time:.2f}s, Avg Tokens/Sec: {token_count/total_time:.3f}")

        # Optional: Log CPU usage
        cpu_usage = current_process.cpu_percent(interval=1)
        logger.info(f"Process CPU Usage: {cpu_usage}%")

        cpu_usage = psutil.cpu_percent(interval=1)
        logger.info(f"System CPU usage: {cpu_usage}")

        cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
        avg_cpu_usage = sum(cpu_usage) / len(cpu_usage)
        logger.info(f"Average CPU usage: {avg_cpu_usage:.2f}%")

        # Optional: Log GPU usage (NVIDIA only)
        try:
            gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,nounits,noheader"]).decode("utf-8").strip()
            logger.info(f"GPU Stats: {gpu_info}")
        except Exception as gpu_err:
            logger.warning(f"GPU monitoring failed: {gpu_err}")

    except Exception as e:
        logger.error("Error during Transformers streaming", exc_info=True)
        error_message = f"Error generating response. Please try again later."
        yield f"data: {json.dumps({'error': error_message})}\n\n"


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main chat interface and starts a session."""
    return render_template('index.html')


@app.route('/stream_chat', methods=['POST'])
def stream_chat():
    """Handles the streaming chat request from the frontend."""
    user_message = request.json.get('message')
    if not user_message:
        return Response(f"data: {json.dumps({'error': 'Please enter a message.'})}\n\n", mimetype='text/event-stream', status=400)

    user_message = date_parser.normalize_dates_in_text(user_message)
    session_id = session.get('session_id')
    def generate_response(session_id = session_id):
        # 1. Classify Intent
        intent_result = classifier_obj.classify_query(user_message)
        intent = intent_result['classified_intent']

        context = ""
        unique_sources = []
        system_prompt = ""
        sql_query_to_send = None
        
        # This will hold the conversation history for the model
        messages = []

        # 2. Handle different intents
        if intent in (config.CLASS_LABELS[0]):
            # db = nl2sql_obj.connect_db(config.PYODBC_CONNECTION_STRING)
            sql_text = nl2sql_obj.generate_query_using_llm(user_message)

            if sql_text:
                clean_sql = nl2sql_obj.extract_sql(sql_text)
                validation_status, _ = nl2sql_obj.validate_sql_syntax(clean_sql)

                if validation_status:
                    try:
                        df_result = nl2sql_obj.execute_query(clean_sql)
                        num_rows = len(df_result)
                        logger.info(f"Total number of rows in the result = {num_rows}")
                        sql_query_to_send = clean_sql

                        def is_zero_count_result(df: pd.DataFrame) -> bool:
                            if len(df) == 1 and df.shape[1] == 1:
                                value = df.iloc[0,0]
                                return isinstance(value, numbers.Number) and value ==0
                            return False

                        if num_rows == 0 or (num_rows == 1 and is_zero_count_result(df_result)):
                            logger.info("No data found prompting...")
                            # No data found — update system prompt accordingly
                            system_prompt = dedent("""
You are an expert data assistant integrated into a larger chatbot. You are responsible for querying the system's database to answer user questions.

**Your Current Situation:**
You have just run a search in the database based on the user's request, but it returned no matching records.

**Your Task:**
1.  **Take Ownership:** Inform the user directly and clearly that you could not find results for their specific request. Do not talk about SQL or technical details. You are the one who performed the search.
2.  **Diagnose & Suggest:** Proactively suggest a few likely reasons why the search came up empty. Think like an analyst (e.g., is it a spelling issue, a date range issue, or a data availability issue?).
3.  **Offer Action, Not Just Advice:** Frame your suggestions as actions *you can take* for the user. Instead of telling them what to do, ask if they'd like *you* to try a different approach. This is key to sounding authoritative.
4.  **CRITICAL:** Do not expose the SQL query to the user unless they specifically ask for it. You are the expert; you handle the technical details.

**Example Interaction:**
* **User Prompt:** "List recent 5 issues logged by bhavba"
* **Your Ideal Response:** "I searched for the five most recent issues logged by 'bhavba' but didn't find any matching records. 

    This could be due to a couple of reasons:
    * The name might be spelled differently in the database.
    * There may not be any issues logged by that user in the recent past.

    Would you like me to try searching for similar names or expand the search to include all entries from the last 90 days?"
        """)

                            messages.append({"role": "system", "content": system_prompt})
                            messages.append({
                                "role": "user",
                                "content": f"Based on my user prompt, you found no results. Now, guide me to a solution.\n\nUser Prompt: {user_message}"
                            })
                        else:
                            cols_to_check = ["fault_description", "first_observation", "action_taken"]
                            if num_rows > 9 and any(col in df_result.columns for col in cols_to_check):
                                preview_rows = df_result.head(3)
                            else:
                                preview_rows = df_result

                            system_prompt = (
                                "You are a helpful assistant skilled at summarizing database results. "
                                "Provide a clear, concise natural language summary of the data returned. "
                                "If results are too many, summarize trends; if few, highlight specifics."
                            )
                            messages.append({"role": "system", "content": system_prompt})
                            messages.append({
                                "role": "user",
                                "content": f"User prompt: {user_message}\nGenerated SQL: {clean_sql}\nTotal rows fetched: {num_rows}\nTop rows(for preview):\n{preview_rows}\n\nPlease summarize the result."
                            })

                    except Exception as e:
                        error_msg = f"There was an error executing your query: {str(e)}"
                        logger.error(error_msg)
                        yield f"data: {json.dumps({'chunk': error_msg})}\n\n"
                        yield f"data: {json.dumps({'chunk': 'Would you like to rephrase your question or try a different query?'})}\n\n"
                        yield f"data: {json.dumps({'status': 'DONE'})}\n\n"
                        return
                else:
                    intent = config.CLASS_LABELS[1]
            else:
                intent = config.CLASS_LABELS[1]

        if intent == config.CLASS_LABELS[1]:
            try:
                logger.info("Using RAG")
                context, _ = rag_obj.retrieve_from_collection(config.FAULT_INFO_COLLECTION, user_message)
                
                system_prompt = dedent("""
You are a specialized AI assistant for the Accelerator Control System, with deep expertise in diagnosing and resolving system faults. Your role is to provide clear, direct, and actionable answers to operators and engineers.

**Your Task:**
1.  **Synthesize, Don't State:** Analyze the technical data provided below to construct a comprehensive response. Connect different pieces of information to give the user a complete picture.
2.  **Speak as the Expert:** Answer confidently and directly, as if this information is your own knowledge.
3.  **CRITICAL:** **Do not mention the context.** Avoid phrases like "Based on the provided context," "According to the information," or anything similar. Just answer the question.
4.  **If Information is Missing:** If the data is insufficient to answer the question, simply state that you don't have enough specific information on that issue and, if possible, suggest what kind of information would be helpful.
""")
                
                messages.append({"role": "system", "content": system_prompt})
                
                # Add the user's question with context
                user_content_with_context = f"Use the following technical data to answer the user's question.\n\nTechnical Data:\n{context}\n\nUser Question:\n{user_message}"
                messages.append({"role": "user", "content": user_content_with_context})
                
                logger.info(f"Using RAG-based prompt for FAULT_INFO. Context Retrieved: {context}")

            except Exception as e:
                logger.error(f"Error retrieving from FAULT_INFO collection: {e}")
                yield f"data: {json.dumps({'error': f'Error fetching fault info: {e}'})}\n\n"
                return

        elif intent == config.CLASS_LABELS[2]:
            try:
                context, unique_sources = rag_obj.retrieve_from_collection(config.DOMAININFO_COLLECTION, user_message)
                system_prompt = "You are a helpful assistant. Use the provided context to answer the question."
                messages.append({"role": "system", "content": system_prompt})
                # Add the user's question with context
                user_content_with_context = f"Context:\n{context}\n\nQuestion:\n{user_message}"
                messages.append({"role": "user", "content": user_content_with_context})
                logger.info("Using RAG-based prompt for DOMAININFO.")

            except Exception as e:
                logger.error(f"Error retrieving from DOMAININFO collection: {e}")
                # Fallback to general model if retrieval fails
                messages.append({"role": "user", "content": user_message})
        
        else: # General chat, no specific intent or RAG
            messages.append({"role": "user", "content": user_message})

        # 4. Stream the LLM response
        try:
            safe_messages = ensure_message_alternation(messages)
        except ValueError as e:
            logger.error(f"Message alternation error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        chatbot_response = ''
        for token_data in stream_inference_generator(safe_messages):
            yield token_data  # Keep streaming full SSE line to frontend

            # Parse and extract the "chunk" for accumulation
            if token_data.startswith("data: "):
                try:
                    chunk_json = json.loads(token_data[len("data: "):])
                    chunk_text = chunk_json.get("chunk", "")
                    chatbot_response += chunk_text
                except json.JSONDecodeError:
                    pass  # Skip invalid JSON

        # Send SQL Query if it was generated
        if sql_query_to_send:
            sql_data = {"sql_query": sql_query_to_send}
            yield f"data: {json.dumps(sql_data)}\n\n"

        # 5. Send sources at the end if they exist
        sources_data = {}
        if unique_sources:
            sources_data = {"sources": [f"Source [{i+1}]: {src}" for i, src in enumerate(unique_sources)]}
            yield f"data: {json.dumps(sources_data)}\n\n"

        sources_data_json = json.dumps(sources_data)
        connection = engine.connect()
        row = connection.execute(text("SELECT MAX(turn_number) from messages where session_id = :session_id"), {"session_id": session_id}).fetchone()
        connection.close()
        
        current_turn_val = row[0] if row and row[0] is not None else 0
        next_turn = current_turn_val + 1

        timestamp = datetime.now()
        classified_intent = intent_result["classified_intent"]
        mistral_intent = intent_result["mistral_intent"]
        tinyllama_intent = intent_result["tinyllama_intent"]
        bert_intent = intent_result["bert_intent"]
        generated_sql = sql_query_to_send if sql_query_to_send else None
        rag_chunks = context if context else None
        connection = engine.connect()

        insert_sql = text("""
    INSERT INTO messages (
        session_id,
        turn_number,
        timestamp,
        user_prompt,
        classified_intent,
        mistral_intent,
        tinyllama_intent,
        bert_intent,
        chatbot_response,
        generated_sql,
        unique_sources,
        rag_chunks
    )
    VALUES (
        :session_id,
        :turn_number,
        :timestamp,
        :user_prompt,
        :classified_intent,
        :mistral_intent,
        :tinyllama_intent,
        :bert_intent,
        :chatbot_response,
        :generated_sql,
        :unique_sources,
        :rag_chunks
    )
""")
        try:
            connection.execute(insert_sql, {
                "session_id": session_id,
                "turn_number": next_turn,
                "timestamp": timestamp,
                "user_prompt": user_message,
                "classified_intent": classified_intent,
                "mistral_intent": mistral_intent,
                "tinyllama_intent": tinyllama_intent,
                "bert_intent": bert_intent,
                "chatbot_response": chatbot_response,
                "generated_sql": generated_sql,
                "unique_sources": sources_data_json,
                "rag_chunks": rag_chunks
            })
            connection.commit()
        except Exception as e:
            logger.info(f"Error executing query: {e}")
        finally:
            connection.close()

        # Signal completion
        yield f"data: {json.dumps({'status': 'DONE'})}\n\n"

    return Response(generate_response(), mimetype='text/event-stream')

@app.route('/start_session')
def start_session():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    start_time = datetime.now()

    insert_sql = text("""
        INSERT INTO sessions (session_id, start_time)
        VALUES (:session_id, :start_time)
    """)
    connection = engine.connect()
    try:
        connection.execute(insert_sql, {
                "session_id": session_id,
                "start_time": start_time
        })
        connection.commit()
    except Exception as e:
        logger.error(f"Error executing SQL on SQL Server: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        connection.close()

    return jsonify({"status": "success", "session_id": session_id})


@app.route('/end_session', methods=['POST'])
def end_session():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"status":"error", "message":"Session ID not found"}), 404

    end_time = datetime.now()

    update_sql = text("""
    UPDATE sessions
    set end_time = :end_time
    WHERE session_id = :session_id
    """)
    connection = engine.connect()
    try:
        connection.execute(update_sql, {
                "end_time":end_time,
                "session_id":session_id
            })

        connection.commit()
        return jsonify({"status": "success", "session_id": session_id, "end_time": str(end_time)})
    
    except Exception as e:
        logger.error(f"Error updating end_time for session {session_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        connection.close()

@app.route('/frontend_log', methods=['POST'])
def frontend_log():
    data = request.get_json()
    logger.info(f"===[FRONTEND LOG]=== {data}")
    return {"status": "logged"}, 200

# --- Application Initialization ---
def initialize_services():
    """Initializes the LLM, mistral_tokenizer, and other services."""
    global mistral_base_model, mistral_tokenizer, device, classifier_obj, rag_obj, nl2sql_obj
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
        
        logger.info(f"Loading mistral_tokenizer for model: {config.MISTRAL_BASE_MODEL_PATH}")
        mistral_tokenizer = AutoTokenizer.from_pretrained(config.MISTRAL_BASE_MODEL_PATH)
        mistral_tokenizer.pad_token = mistral_tokenizer.eos_token

        tinyllama_tokenizer = AutoTokenizer.from_pretrained(config.TINYLLAMA_BASE_MODEL_PATH)
        tinyllama_tokenizer.pad_token = tinyllama_tokenizer.eos_token

        logger.info(f"Loading model: {config.MISTRAL_BASE_MODEL_PATH}. This may take a while...")

        # For NL2SQL:
        base_model_nl2sql = AutoModelForCausalLM.from_pretrained(
            config.MISTRAL_BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # For Summarization and general use:
        mistral_base_model = AutoModelForCausalLM.from_pretrained(
            config.MISTRAL_BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto"
        )

        tinyllama_base_model = AutoModelForCausalLM.from_pretrained(
            config.TINYLLAMA_BASE_MODEL_PATH,
            quantization_config = bnb_config,
            device_map = "auto"
        )

        logger.info(f"LLM Model: {config.MISTRAL_BASE_MODEL_PATH} Initialized Successfully.")

    except Exception as e:
        logger.critical("Failed to initialize LLM model.", exc_info=True)
        exit(1)

    try:
        classifier_obj = IntentClassifier(mistral_base_model, mistral_tokenizer, tinyllama_base_model, tinyllama_tokenizer)
        logger.info("Intent Classifier Initialized Successfully.")
    except Exception as e:
        logger.critical("Failed to initialize intent classifier.", exc_info=True)
        exit(1)

    try:
        nl2sql_obj = NL2SQL(base_model_nl2sql, mistral_tokenizer)
        logger.info("NL2SQL Model Initialized Successfully.")
    except Exception as e:
        logger.critical("Failed to initialize NL2SQL Model.", exc_info=True)
        exit(1)

    try:
        rag_obj = RagSetup()
        logger.info("RAG Setup Initialized Successfully.")
    except Exception as e:
        logger.critical("Failed to initialize RAG setup.", exc_info=True)
        exit(1)

    logger.info("All services initialized successfully.")

if __name__ == "__main__":
    initialize_services()
    # use_reloader=False is important to prevent re-initialization on each change in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
