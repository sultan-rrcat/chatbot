import logging
import os

ACC_PY_DOC_DIR = r"C:\Users\Administrator\Documents\chatbot\app\backend\rag_data\acc_py_docs"
DB_SCHEMA_DOC_DIR = r"C:\Users\Administrator\Documents\chatbot\app\backend\rag_data\db_schema_docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DOMAININFO_COLLECTION = "ACC_PY_DOCS"
DBSCHEMA_COLLECTION = "DB_SCHEMA_DOCS"
FAULT_INFO_COLLECTION = "FAULT_INFO_COLLECTION"

LOG_DIRECTORY = r'C:\Users\Administrator\Documents\chatbot\app\backend\logs\app.log'

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

CONNECTION_STRING = f"""
DRIVER={{SQL Server}};
SERVER={SERVER};
DATABASE={DATABASE};
Trusted_Connection=yes
"""

db_uri = f"mssql+pyodbc://{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"