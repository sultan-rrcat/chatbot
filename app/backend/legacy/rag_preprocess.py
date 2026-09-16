import pandas as pd
from sqlalchemy import create_engine
import urllib
import re
import os

# -----------------------------
# 1️⃣ Connect to SQL Server using SQLAlchemy with Trusted Connection
SERVER = 'DESKTOP-FG7N2DC\\SQLEXPRESS'
DATABASE = 'flogbook'

params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"Trusted_Connection=yes;"
)
engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')

# -----------------------------
# 2️⃣ Fetch data from fault_bookv3 with explicit encoding handling
query = 'SELECT * FROM fault_bookv3'
df = pd.read_sql(query, engine)

# -----------------------------
# 3️⃣ Data Preprocessing
def clean_duration(duration):
    if isinstance(duration, str):
        duration = duration.strip().lower()
        if duration in ['na', 'n/a', 'not available', '']:
            return '0 min'
        match = re.match(r'(\\d+)\\s*mins?\\s*(\\d+)\\s*secs?', duration)
        if match:
            mins, secs = match.groups()
            total_min = int(mins) + (int(secs) / 60)
            return f"{total_min:.1f} min"
        return duration
    return '0 min'

# Normalize date columns
df['fault_time'] = pd.to_datetime(df['fault_time'], errors='coerce')
df['log_time'] = pd.to_datetime(df['log_time'], errors='coerce')

# Clean fault_duration
df['fault_duration'] = df['fault_duration'].apply(clean_duration)

# Replace NULLs with 'N/A'
df.fillna('N/A', inplace=True)

# Strip spaces from string columns with safe coercion to avoid UnicodeDecodeError
def safe_strip(val):
    try:
        return str(val).strip()
    except Exception:
        return 'N/A'

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(safe_strip)

# -----------------------------
# DataFrame `df` is now ready for chunking and embedding.
print("✅ Data fetched directly from SQL Server with trusted connection, preprocessed safely, and ready for RAG pipeline.")
# -----------------------------
# 4️⃣ Save to TXT file
OUTPUT_TXT_PATH = r'C:\Users\Administrator\Desktop\chatbot\backend\data\fault_bookv3_cleaned.txt'
os.makedirs(os.path.dirname(OUTPUT_TXT_PATH), exist_ok=True)

with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f:
    for idx, row in df.iterrows():
        record = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        f.write(record + "\n" + "-"*40 + "\n")

print(f"✅ Data fetched, preprocessed, and saved to {OUTPUT_TXT_PATH} for RAG pipeline.")