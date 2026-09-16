import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Merged museum config (learning record — not expected to run as-is).
# Canonical source: app_home/backend/config.py (latest/modular phase).
# Legacy values from app_rag + app_kshitij are preserved below as comments
# and deprecated aliases. Original absolute paths are kept in comments for
# provenance; active values are now relative to this file so the single app/
# folder is portable. See app/docs/provenance.md and app/.env.example.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent          # app/backend/
APP_DIR = BASE_DIR.parent                            # app/
RAG_DATA_DIR = BASE_DIR / "rag_data"

# Canonical document dirs (were absolute C:\Users\offic\Documents\chatbot\...).
# Originals:
#   ACC_PY_DOC_DIR    = r"C:\Users\offic\Documents\chatbot\app_home\backend\rag_data\acc_docs"
#   DB_SCHEMA_DOC_DIR = r"C:\Users\offic\Documents\chatbot\app_home\backend\rag_data\db_docs"
#   FAULT_DOC_DIR     = r"C:\Users\offic\Documents\chatbot\app_home\backend\rag_data\fault_docs"
ACC_PY_DOC_DIR = os.environ.get("ACC_PY_DOC_DIR", str(RAG_DATA_DIR / "acc_docs"))
DB_SCHEMA_DOC_DIR = os.environ.get("DB_SCHEMA_DOC_DIR", str(RAG_DATA_DIR / "db_docs"))
FAULT_DOC_DIR = os.environ.get("FAULT_DOC_DIR", str(RAG_DATA_DIR / "fault_docs"))

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# Kshitij-era variant used CHUNK_OVERLAP = 50 (see legacy/kshitij_app_monolith.py).
KSHITIJ_CHUNK_OVERLAP = 50

DOMAININFO_COLLECTION = "ACC_PY_DOCS"
DBSCHEMA_COLLECTION = "DB_SCHEMA_DOCS"
FAULT_INFO_COLLECTION = "FAULT_INFO_COLLECTION"
# Kshitij-era single collection: "accelerator_papers".
KSHITIJ_COLLECTION = "accelerator_papers"
# RAG-era Chroma dir: "new_rag_data/chroma_faultbook_index" (relative to CWD).
RAG_CLI_PERSIST_DIR = "new_rag_data/chroma_faultbook_index"

# NOTE (preserved bug, museum): app_home/backend/rag_setup.py references
# config.CHROMA_DB_DIR 3x, but app_home/backend/config.py never defined it.
# Defined here as the canonical Chroma dir so the reference resolves.
# Original repo ignored: app_home/backend/chroma_db (see .gitignore).
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", str(BASE_DIR / "chroma_db"))
# Kshitij-era absolute: r"C:\Users\Administrator\Desktop\chatbot\backend\kshitij_app\chroma_db"
KSHITIJ_CHROMA_DB_PATH = str(BASE_DIR / "chroma_db_kshitij")

# Original: r'C:\Users\offic\Documents\chatbot\app_home\backend\logs\app.log'
LOG_DIRECTORY = os.environ.get("LOG_DIRECTORY", str(BASE_DIR / "logs" / "app.log"))
# Kshitij-era variant: plain "app_chatbot.log" in CWD.
KSHITIJ_LOG_FILE = "app_chatbot.log"

# Model paths: originals were local absolute paths (git-ignored `models/`).
#   EMBEDDER  = r"C:\Users\offic\Documents\chatbot\models\embedding\bge-en"
#   INTENT    = r"...\models\intent_classifier\intent_classifier_model"
#   DEEPSEEK  = r"...\models\llms\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
#   LLAMA31   = r"...\models\llms\Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
#   PHI3      = r"...\models\llms\Phi-3-mini-4k-instruct-q4.gguf"
# RAG-era:    EMBEDDER_MODEL_PATH = r"C:\Users\Administrator\Desktop\chatbot\backend\models\allminilm"
# Kshitij-era: LLM = Phi-3.1-mini-128k-instruct-IQ2_M.gguf, EMBEDDING = .../allminilm
MODELS_DIR = APP_DIR / "models"
EMBEDDER_MODEL_PATH = os.environ.get("EMBEDDER_MODEL_PATH", str(MODELS_DIR / "embedding" / "bge-en"))
INTENT_CLASSIFIER_MODEL = os.environ.get("INTENT_CLASSIFIER_MODEL", str(MODELS_DIR / "intent_classifier" / "intent_classifier_model"))
DEEPSEEK_MODEL_PATH = os.environ.get("DEEPSEEK_MODEL_PATH", str(MODELS_DIR / "llms" / "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"))
LLAMA31_MODEL_PATH = os.environ.get("LLAMA31_MODEL_PATH", str(MODELS_DIR / "llms" / "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"))
PHI3_MODEL_PATH = os.environ.get("PHI3_MODEL_PATH", str(MODELS_DIR / "llms" / "Phi-3-mini-4k-instruct-q4.gguf"))
RAG_MINIMAL_EMBEDDER_PATH = str(MODELS_DIR / "allminilm")  # from legacy/rag_minimal_config.py
KSHITIJ_LLM_MODEL_NAME = str(MODELS_DIR / "Phi-3.1-mini-128k-instruct-IQ2_M.gguf")
KSHITIJ_EMBEDDING_MODEL_NAME = str(MODELS_DIR / "allminilm")
RAG_CLI_LLM_GGUF = str(MODELS_DIR / "DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf")  # legacy/rag_cli_main.py

CLASS_LABELS = {
    0: "INTENT1_REALTIME",
    1: "INTENT2_ANALYTICAL",
    2: "INTENT3_FAULTINFO",
    3: "INTENT4_DOMAININFO",
    4: "INTENT5_GENERALINFO"
}

def setup_logging():
    log_dir = os.path.dirname(LOG_DIRECTORY)
    os.makedirs(log_dir, exist_ok=True)

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=LOG_DIRECTORY,
        filemode='a'
    )

# SERVER = 'TUF%5CSQLEXPRESS'
SERVER = 'localhost\\SQLEXPRESS'
DATABASE = 'CONTROLS'

# CONNECTION_STRING = f"""
# DRIVER={{SQL Server}};
# SERVER={SERVER};
# DATABASE={DATABASE};
# Trusted_Connection=yes
# """

PYODBC_CONNECTION_STRING = f"mssql+pyodbc://{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

SQLALCHEMY_CONNECTION_STRING = (
    f"mssql+pyodbc://@{SERVER}/{DATABASE}"
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)


# from llama_cpp import Llama

# print(Llama.__doc__)

# from llama_cpp import Llama

# llm = Llama(
#     model_path=DEEPSEEK_MODEL_PATH,
#     n_ctx=2048,
#     n_gpu_layers=1   # test GPU offload
# )
# print("Llama initialized successfully with n_gpu_layers=1")
