import logging
import os
from logging.handlers import RotatingFileHandler

# Using os.path.join for better cross-platform compatibility, even if primarily for Windows.
# ==============================================================================
# --- 📁 FILE & DIRECTORY PATHS ---
# ==============================================================================
BASE_DIR = r"C:\Users\admin\Documents\chatbot"
APP_DIR = r"C:\Users\admin\Documents\chatbot\app"

# ==============================================================================
# --- 🧠 MODEL PATHS ---
# ==============================================================================
MODELS_BASE_DIR = r"C:\Users\admin\Documents\chatbot\models"

# --- Embedder and Classifier Models ---
ALLMINILM_EMBEDDER_MODEL_PATH = r"C:\Users\admin\Documents\chatbot\models\embedders\allminilm"
NOMIC_EMBED_TEXT_V1_EMBEDDER_MODEL_PATH = r"C:\Users\admin\Documents\chatbot\models\embedders\nomic-embed-text-v1"
BERT_INTENT_CLASSIFIER_MODEL = r"C:\Users\admin\Documents\chatbot\models\intent_classifier_model\bert-based-intent-classifier"
SQL_GEN_MODEL = r"C:\Users\admin\Documents\chatbot\models\embedders\prem-1B-SQL"

# --- Large Language Models (LLMs) ---
DEEPSEEK_MODEL_PATH = r"C:\Users\admin\Documents\chatbot\models\llms\DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf"
PHI3_MINI_MODEL_PATH = r"C:\Users\admin\Documents\chatbot\models\llms\Phi-3.1-mini-128k-instruct-IQ2_M.gguf"

MISTRAL_BASE_MODEL_PATH = r"C:\Users\admin\Documents\chatbot\models\llms\mistral_7B"
TINYLLAMA_BASE_MODEL_PATH = r"C:\Users\admin\Documents\chatbot\models\llms\tiny_llama"

# ---  Adapter Layer Path ---
NL2SQL_ADAPTER_PATH = r"C:\Users\admin\Documents\chatbot\models\nl2sql_model\mistral_7b_v03_it_sql_adapter"
MISTRAL_LLM_CLASSIFIER_ADAPTER_PATH = r"C:\Users\admin\Documents\chatbot\models\intent_classifier_model\mistral_7b_v03_it_classifier_adapter"
TINYLLAMA_LLM_CLASSIFIER_ADAPTER_PATH = r"C:\Users\admin\Documents\chatbot\models\intent_classifier_model\tiny_llama_it_classifier_adapter"

# ==============================================================================
# --- ⚙️ RAG & VECTOR DB SETTINGS ---
# ==============================================================================
# --- Source Document Directories for RAG ---
RAG_DATA_DIR = r"C:\Users\admin\Documents\chatbot\app\backend\rag_data"
ACC_PY_DOC_DIR = os.path.join(RAG_DATA_DIR, "acc_py_docs")
DB_SCHEMA_DOC_DIR = os.path.join(RAG_DATA_DIR, "db_schema_docs")
FAULT_DOC_DIR = os.path.join(RAG_DATA_DIR, "fault_docs")

# --- Database & Log Directories ---
CHROMA_DB_DIR = r"C:\Users\admin\Documents\chatbot\app\backend\chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- Collection Names for ChromaDB ---
DOMAININFO_COLLECTION = "DOMAININFO_COLLECTION"
DBSCHEMA_COLLECTION = "DBSCHEMA_COLLECTION"
FAULT_INFO_COLLECTION = "FAULT_INFO_COLLECTION"
INTENT_CLASSIFICATION_COLLECTION = "INTENT_CLASSIFICATION_COLLECTION"


# ==============================================================================
# --- 🎯 INTENT CLASSIFICATION ---
# ==============================================================================
CLASS_LABELS = {
    0: "INTENT1_FAULT_SQL",
    1: "INTENT2_FAULT_RAG",
    2: "INTENT3_DOMAIN_RAG",
    3: "INTENT4_GENERAL_INFO"
}

INTENT_CLASSIFICATION_TRAIN_DATA = r"C:\Users\admin\Documents\chatbot\others\intent_data_v2.csv"
# ==============================================================================
# --- 🗄️ DATABASE CONNECTION (SQL Server) ---
# ==============================================================================
SERVER = 'DESKTOP-FDPS0T7\\SQLEXPRESS'
DATABASE = 'flogbook'
DRIVER = 'ODBC Driver 17 for SQL Server'

# --- Connection strings for different libraries ---
PYODBC_CONNECTION_STRING = f"mssql+pyodbc://{SERVER}/{DATABASE}?driver={DRIVER}&trusted_connection=yes"
SQLALCHEMY_CONNECTION_STRING = f"mssql+pyodbc://@{SERVER}/{DATABASE}?driver={DRIVER}&trusted_connection=yes"
SQLALCHEMY_CHATBOT_CONNECTION_STRING = f"mssql+pyodbc://@{SERVER}/chatbot?driver={DRIVER}&trusted_connection=yes"


# ==============================================================================
# --- 🌐 FLASK APP CONFIGURATION ---
# ==============================================================================
FLASK_APP_BASE_DIR = r"C:\Users\admin\Documents\chatbot\app\frontend"
FLASK_APP_STATIC_FOLDER = os.path.join(FLASK_APP_BASE_DIR, "static")
FLASK_APP_TEMPLATE_FOLDER = os.path.join(FLASK_APP_BASE_DIR, "templates")


# ==============================================================================
# --- 📝 LOGGING SETUP ---
# ==============================================================================
LOG_DIRECTORY = r'C:\Users\admin\Documents\chatbot\app\backend\logs'
LOG_FILE = os.path.join(LOG_DIRECTORY, "app.log")

def setup_logging():
    """
    Configures logging to output to both a file and the console.
    The file handler uses UTF-8 encoding to support all characters.
    """
    # Ensure the log directory exists
    os.makedirs(LOG_DIRECTORY, exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the lowest level for the logger

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define a consistent format for all log messages
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Console Handler ---
    # Logs messages to the console (useful for real-time debugging)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO) # Set level for console output
    # console_handler.setFormatter(log_format)
    # logger.addHandler(console_handler)

    # --- File Handler ---
    # Logs messages to a file, with UTF-8 encoding for emoji support
    # RotatingFileHandler prevents the log file from growing indefinitely
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=20*1024*1024, # 20 MB
        backupCount=2,
        encoding='utf-8' # CRITICAL: This ensures emojis are written correctly
    )
    file_handler.setLevel(logging.DEBUG) # Log more detailed info to the file
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
