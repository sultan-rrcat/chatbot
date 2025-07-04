import config
import logging
import pyodbc
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains.llm import LLMChain
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
# from langchain.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_ollama import ChatOllama
# from llama_cpp import Llama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate, PromptTemplate, MessagesPlaceholder
import sqlglot
import re
from langchain_community.llms import LlamaCpp
config.setup_logging()
logger = logging.getLogger(__name__)
logger.info("logging started...")

class NL2SQL:
    def __init__(self):
        print("loading llm...")
        # self.llm = ChatOllama(model="phi3")
        # self.llm = Llama(model_path=config.DEEPSEEK_MODEL_PATH, n_ctx=62000, verbose=False)
        self.llm = LlamaCpp(
                model_path=config.PHI3_MINI_MODEL_PATH,
                n_ctx=8192, # Adjusted to a more standard context size to prevent errors
                temperature=0,
                max_tokens=512,
                verbose=False,
                streaming=True
            )
        # self.llm = Llama(model_path=config.SQL_GEN_MODEL,n_ctx=10000, verbose=False)
        print("llm loaded...")
        pass
    def connect_db(self, conn_string):
        try:
            db = SQLDatabase.from_uri(conn_string)
            # print(db.get_usable_table_names())
            # print(db.table_info)
            return db
        except Exception as e:
            print(f"Error connecting database: {e}")
        pass
    
    def generate_query(self, prompt, db):
        with open(r"C:\Users\Administrator\Desktop\chatbot\backend\app\rag_data\db_schema_docs\table_schema.txt", 'r') as f:
            table_details = f.read()

        examples = [
            {
                "input": "Persons involved for any issues regarding INDUS2 RF sytems?",
                "query": "select persons_involved from fault_bookv3 where lower(system_name) like '%indus-2 rf%' group by persons_involved;"
            },
            {
                "input": "List top 10 faulty devices",
                "query": "SELECT TOP 10 device_name, count(*) as cnt FROM fault_bookv3 group by device_name order by cnt desc;"
            },
            {
                "input": "How many times beam has been affected for faults ",
                "query": "select count(*) as no_of_times from fault_bookv3 where lower(beam_affected) like '%yes%';"
            }
        ]

        # Create a string representation of the examples
        example_texts = []
        for example in examples:
            example_texts.append(f"Human: {example['input']}\nSQLQuery:\nAI: {example['query']}")
        
        example_str = "\n".join(example_texts)


        # A much cleaner and more direct prompt template
        prompt_template = """System: You are a SQL Server expert. Given an input question, create a syntactically correct SQL Server query to run.
    Only return the SQL query and nothing else.

    Table Info:
    {table_info}

    Here are some examples of how to write good queries:
    {examples}

    Human: {input}
    SQLQuery:
    AI:"""

        final_prompt = PromptTemplate(
            input_variables=["table_info", "examples", "input"],
            template=prompt_template
        )
        
        # Use the recommended RunnableSequence for modern LangChain
        generate_query_chain = final_prompt | self.llm

        # Invoke the chain with the necessary variables for your prompt
        result = generate_query_chain.invoke({
            "input": prompt.strip(),
            "table_info": table_details.strip(),
            "examples": example_str
        })

        print("--- Generated SQL ---")
        # The output from a direct LLM call is the content string itself
        print(result)
        print("--------------------")
        return result


    def extract_sql(self, text):
        # Remove markdown blocks
        if "```sql" in text:
            sql = re.findall(r"```sql\s*(.*?)```", text, re.DOTALL)
            if sql:
                return sql[0].strip()
        # Remove "Here is the SQL query:" and capture the SQL
        pattern = r"(SELECT|WITH|INSERT|UPDATE|DELETE).*"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0).strip()
        # If none found, return the original text as fallback
        return text.strip()

    def validate_sql_syntax(self, sql_text):
        clean_sql = self.extract_sql(sql_text['result'] if isinstance(sql_text, dict) else sql_text)
        print(clean_sql)
        try:
            parsed = sqlglot.parse_one(clean_sql, read='tsql')  # Use 'tsql' for SQL Server
            print("SQL syntax is valid.")
            return True, None
        except sqlglot.errors.ParseError as e:
            print(f"SQL syntax error: {e}")
            return False, str(e)

    def execute_query(self, db, sql_query):
        pass

if __name__ == "__main__":
    obj = NL2SQL()
    # The 'db' object is no longer strictly necessary for this generation part
    # but can be kept for other functionalities.
    db = obj.connect_db(config.db_uri) 

    prompt = "Count the number of faults that happened in the morning (before 12 PM)"
    sql_text = obj.generate_query(prompt, db)
    # validation_status = obj.validate_sql_syntax(sql_text)
    # print(validation_status)
    pass