import os
os.environ["CHROMA_TELEMETRY"] = "FALSE"

from langchain_ollama import ChatOllama
from llama_cpp import Llama
from langchain_core.prompts import ChatPromptTemplate
from intent_classifier import IntentClassifier
from rag_setup import RagSetup
from NL2SQL import NL2SQL
from textwrap import dedent
import logging
import config

config.setup_logging()
logger = logging.getLogger(__name__)
logger.info("Logging started...")

# llm = ChatOllama(model="tinyllama")
llm_model = Llama(model_path=config.DEEPSEEK_MODEL_PATH, n_ctx=62000, verbose=False)

def inference(user_prompt):
    output = llm_model(prompt=user_prompt, max_tokens=512, temperature=0.7, top_p=0.9)
    print(f"Prompt tokens: {output['usage']['prompt_tokens']}, Completion tokens: {output['usage']['completion_tokens']}")
    response_text = output["choices"][0]["text"].strip()

    # Ollama inferencing technique
    # response = llm.invoke(user_prompt)
    # print(response.content)
    return response_text

if __name__ == "__main__":
    # user_prompt = "What is Linear Betatron Motion?"
    # user_prompt = "Tell me about INDUS2 Accelerator"
    # user_prompt = "Best book for physics?"

    classifier_obj = IntentClassifier()
    rag_obj = RagSetup()

    while(True):
        user_prompt = input("\nEnter your question (type 'exit' to quit): ")
        if user_prompt == "exit":
            break
        
        result = classifier_obj.classify_query(user_prompt)
        print(result)

        if result in ("INTENT1_REALTIME", "INTENT2_ANALYTICAL", "INTENT3_FAULTINFO"):
            # nl2sql_obj = NL2SQL()
            try:
                context, unique_sources  = rag_obj.retrieve_from_collection(config.DBSCHEMA_COLLECTION, user_prompt)
            except Exception as e:
                logger.error(f"Retrieval Error: {e}")
                print("Could not retrieve context. Answering using general model.")
                response_text = inference(user_prompt)
                print(response_text)
                continue
            
            prompt_template = ChatPromptTemplate.from_template(dedent("""
                You are a SQL server query expert using the following context related to database schema generate a SQL query:

                Context:
                {context}

                Question:
                {question}

                Answer:
            """))

            formatted_prompt = prompt_template.format_messages(context = context, question = user_prompt)
            print("Context: ", context)
            print("Formatted Prompt: ", formatted_prompt)
            # response_text = inference(formatted_prompt[0].content)
            # print(response_text)
            pass

        elif result == "INTENT4_DOMAININFO":
            try:
                context, unique_sources  = rag_obj.retrieve_from_collection(config.DOMAININFO_COLLECTION, user_prompt)
            except Exception as e:
                logger.error(f"Retrieval Error: {e}")
                print("Could not retrieve context. Answering using general model.")
                response_text = inference(user_prompt)
                print(response_text)
                continue

            prompt_template = ChatPromptTemplate.from_template(dedent("""
                You are a helpful assistant using the following context to answer questions:

                Context:
                {context}

                Question:
                {question}

                Answer:
            """))

            
            formatted_prompt = prompt_template.format_messages(context = context, question = user_prompt)

            # response = llm.invoke(formatted_prompt)
            # print(response.content)
            response_text = inference(formatted_prompt[0].content)
            print(response_text)
            for i, src in enumerate(unique_sources, 1):
                print(f"\nSource [{i}] {src}")

        else:
            response_text = inference(user_prompt)
            # response = llm.invoke(user_prompt)
            # print(response.content)
            print(response_text)