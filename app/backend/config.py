import logging
import os

ACC_PY_DOC_DIR = r"C:\Users\Administrator\Documents\chatbot\app\backend\rag_data\acc_py_docs"
DB_SCHEMA_DOC_DIR = r"C:\Users\Administrator\Documents\chatbot\app\backend\rag_data\db_schema_docs"
FAULT_DOC_DIR = r"C:\Users\Administrator\Documents\chatbot\app\backend\rag_data\fault_docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DOMAININFO_COLLECTION = "DOMAININFO_COLLECTION"
DBSCHEMA_COLLECTION = "DBSCHEMA_COLLECTION"
FAULT_INFO_COLLECTION = "FAULT_INFO_COLLECTION"

LOG_DIRECTORY = r'C:\Users\Administrator\Documents\chatbot\app\backend\logs\app.log'

CHROMA_DB_DIR = r"C:\Users\Administrator\Documents\chatbot\app\backend\chroma_db"

EMBEDDER_MODEL_PATH = r"C:\Users\Administrator\Documents\chatbot\models\embedders\allminilm"
INTENT_CLASSIFIER_MODEL = r"C:\Users\Administrator\Documents\chatbot\models\intent_classifier_model"
SQL_GEN_MODEL = r"C:\Users\Administrator\Documents\chatbot\models\embedders\prem-1B-SQL"
TINYLLAMA_MODEL_PATH = r"C:\Users\Administrator\Documents\chatbot\models\llms\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
DEEPSEEK_MODEL_PATH = r"C:\Users\Administrator\Documents\chatbot\models\llms\DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf"
PHI3_MINI_MODEL_PATH = r"C:\Users\Administrator\Documents\chatbot\models\llms\Phi-3.1-mini-128k-instruct-IQ2_M.gguf"

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

SERVER = 'DESKTOP-FG7N2DC\\SQLEXPRESS'
DATABASE = 'flogbook'

PYODBC_CONNECTION_STRING = f"mssql+pyodbc://{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

SQLALCHEMY_CONNECTION_STRING = (
    f"mssql+pyodbc://@{SERVER}/{DATABASE}"
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

FLASK_APP_STATIC_FOLDER = r"C:\Users\Administrator\Documents\chatbot\app\frontend\static"
FLASK_APP_TEMPLATE_FOLDER = r"C:\Users\Administrator\Documents\chatbot\app\frontend\templates"