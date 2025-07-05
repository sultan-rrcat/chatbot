import pandas as pd
from sqlalchemy import create_engine
import urllib
import re
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import config

# -----------------------------
# 1️⃣ Connect to SQL Server using SQLAlchemy with Trusted Connection

# SERVER = config.SERVER
# DATABASE = config.DATABASE

# params = urllib.parse.quote_plus(
#     f"DRIVER={{ODBC Driver 17 for SQL Server}};"
#     f"SERVER={SERVER};"
#     f"DATABASE={DATABASE};"
#     f"Trusted_Connection=yes;"
# )
# engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')

engine = create_engine(config.CONNECTION_STRING)

# -----------------------------
# 2️⃣ Fetch data from fault_bookv3

query = 'SELECT * FROM fault_bookv3 where fault_id<4000;'
df = pd.read_sql(query, engine)
# df = pd.read_sql_query(query, engine)
with open('faultbook_data.csv', 'w', encoding='utf-8') as f:
    df.to_csv(f, index=False)
print(df.head(3))

# -----------------------------
# 3️⃣ Data Preprocessing

def clean_duration(duration):
    if isinstance(duration, str):
        duration = duration.strip().lower()
        if duration in ['na', 'n/a', 'not available', '']:
            return '0 min'
        match = re.match(r'(\d+)\s*mins?\s*(\d+)\s*secs?', duration)
        if match:
            mins, secs = match.groups()
            total_min = int(mins) + (int(secs) / 60)
            return f"{total_min:.1f} min"
        return duration
    return '0 min'

df['fault_time'] = pd.to_datetime(df['fault_time'], errors='coerce')
df['log_time'] = pd.to_datetime(df['log_time'], errors='coerce')
df['fault_duration'] = df['fault_duration'].apply(clean_duration)
df.fillna('N/A', inplace=True)

def safe_strip(val):
    try:
        return str(val).strip()
    except Exception:
        return 'N/A'

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(safe_strip)

# -----------------------------
# 4️⃣ FaultbookIngestor Class

class FaultbookIngestor:
    """
    Handles transformation of fault log DataFrame into embeddings
    and persists them into Chroma vector store for retrieval.
    """

    def __init__(self, dataframe: pd.DataFrame, persist_directory: str, collection_name: str):
        self.df = dataframe
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.documents = []

        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDER_MODEL_PATH,
            model_kwargs={"local_files_only": True}
        )

    def row_to_chunk(self, row) -> str:
        return f"""
Fault ID: {row['fault_id']}
Fault Time: {row['fault_time']}
System: {row['system_name']}
Device: {row['device_name']}
Description: {row['fault_description']}
Persons Involved: {row['persons_involved']}
Action Taken: {row['action_taken']}
Faulty System: {row['faulty_system']}
Human Error: {row['human_error']}
FDA Entry: {row['fda_entry']}
Logged By: {row['logged_by']}
Log Time: {row['log_time']}
First Observation: {row['first_observation']}
Beam Affected: {row['beam_affected']}
""".strip()

    def prepare_documents(self):
        documents = []
        for _, row in self.df.iterrows():
            content = self.row_to_chunk(row)
            metadata = {
                "fault_id": row['fault_id'],
                "fault_time": str(row['fault_time']),
                "system_name": row['system_name']
            }
            documents.append(Document(page_content=content, metadata=metadata))
        self.documents = documents

    def ingest_to_chroma(self):
        if not self.documents:
            self.prepare_documents()

        vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        print(f"✅ Chroma vector store created and persisted at: {self.persist_directory}")
        print(f"✅ Collection name used: {self.collection_name}")
        return vectorstore

# -----------------------------
# 5️⃣ Entry Point

if __name__ == "__main__":
    config.setup_logging()

    persist_dir = "chroma_db"
    collection_name = "FAULT_INFO_COLLECTION"

    with open('faultbook_data.csv', 'r', encoding='utf-8') as f:
        df = pd.read_csv(f) 
    # df.fillna('N/A', inplace=True)

    ingestor = FaultbookIngestor(df, persist_directory=persist_dir, collection_name=collection_name)
    vectorstore = ingestor.ingest_to_chroma()
