import config
import logging 
from transformers import pipeline
from langchain_ollama import ChatOllama
import pyodbc
from llama_cpp import Llama

config.setup_logging()
logger = logging.getLogger(__name__)

class SQLSetup:
    def __init__(self):
        print("object is getting created...")
        # self.sql_llm = ChatOllama(model="phi3")
        # self.sql_llm = pipeline("", model = "")
        self.sql_llm = Llama(model_path=config.SQL_GEN_MODEL,n_ctx=10000, verbose=False)
        print("LLm loaded...")
        pass
    def get_table_schema(self, table_name, connection_string):
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        sql_query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME='{table_name}'
        """
        cursor.execute(sql_query)
        columns_info = cursor.fetchall()
        print(columns_info)

        # Fetch a demo row to give it as a sample data
        # try: 
        #     cursor.execute(f"SELECT TOP 1 * FROM {table_name}")
        #     demo_rows = cursor.fetchall()
        #     print(demo_rows)
        # except Exception as e:
        #     print(e)
        #     demo_rows = None

        schema_lines = [f"Table: {table_name}", "Columns:"]
        for col_index, col_info in enumerate(columns_info):
            col_name, data_type, is_nullable, char_len = col_info
            line = f" -{col_name} ({data_type.upper()})"
            if char_len and char_len > 0:
                line += f" [{char_len}]"
            if is_nullable == "NO":
                line += " NOT NULL"

            # # append demo data
            # if demo_rows:
            #     sample_value = [repr(row[col_index]) for row in demo_rows]
            #     line += f" => Sample: {', '.join(sample_value)}"
            # else:
            #     line += " => Sample: N/A"
            schema_lines.append(line)
            
        cursor.close()
        conn.close()
        return "\n".join(schema_lines)

    def generate_sql_query(self, prompt):
        # sql_query = self.sql_llm.invoke(prompt).content
        sql_query = self.sql_llm(prompt=prompt, max_tokens=256, temperature=0.7, top_p=0.9)
        print(sql_query["choices"][0]["text"].strip())
        # print(sql_query)
        pass
    def execute_sql_query(self, conn_string, sql_query):
        pass
    def generate_response(self, sql_result):
        pass
if __name__ == "__main__":
    obj = SQLSetup()
    table_schema = obj.get_table_schema('fault_bookv3', config.CONNECTION_STRING)

    # with open("table_schema.txt", 'w') as f:
    #     f.write(table_schema)
    # f.close()

    with open("table_schema.txt", 'r') as f:
        table_schema = f.read()
    f.close()


    context = f"""
    You are advanced chatbot designed for generating SQL Queries in SQL Server, 
    from the given table schema \n{table_schema} 
    \nConvert this below prompt to a SQL query for SQL Server. 
    Just need the SQL Query dont generate anything else\n"""

    prompt = f"{context}Prompt: To whome we need to contact if there is a fault in RF systems?"
    # prompt = f"{context}Prompt: How many times Power Supply related fault occured?"

    # BEFORE INCORPORATING MORE CONTEXT USE A FINE TUNED MODEL AND EVALUATE THE PERFORMANCE

    print(prompt)
    obj.generate_sql_query(prompt)
    pass