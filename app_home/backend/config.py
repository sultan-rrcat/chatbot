import logging
import os

ACC_PY_DOC_DIR = r"C:\Users\offic\Documents\chatbot\app_home\backend\rag_data\acc_docs"
DB_SCHEMA_DOC_DIR = r"C:\Users\offic\Documents\chatbot\app_home\backend\rag_data\db_docs"
FAULT_DOC_DIR = r"C:\Users\offic\Documents\chatbot\app_home\backend\rag_data\fault_docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DOMAININFO_COLLECTION = "ACC_PY_DOCS"
DBSCHEMA_COLLECTION = "DB_SCHEMA_DOCS"
FAULT_INFO_COLLECTION = "FAULT_INFO_COLLECTION"

LOG_DIRECTORY = r'C:\Users\offic\Documents\chatbot\app_home\backend\logs\app.log'

EMBEDDER_MODEL_PATH = r"C:\Users\offic\Documents\chatbot\models\embedding\bge-en"
INTENT_CLASSIFIER_MODEL = r"C:\Users\offic\Documents\chatbot\models\intent_classifier\intent_classifier_model"
DEEPSEEK_MODEL_PATH = r"C:\Users\offic\Documents\chatbot\models\llms\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
LLAMA31_MODEL_PATH = r"C:\Users\offic\Documents\chatbot\models\llms\Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
PHI3_MODEL_PATH = r"C:\Users\offic\Documents\chatbot\models\llms\Phi-3-mini-4k-instruct-q4.gguf"

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
