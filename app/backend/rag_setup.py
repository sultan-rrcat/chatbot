import os
os.environ["CHROMA_TELEMETRY"] = "FALSE"
import logging
import config
import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

config.setup_logging()
logger = logging.getLogger(__name__)
logger.info("rag setup logging enabled...")

class RagSetup:
    def __init__(self):
        print("object initilization...")
        # self.embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name = config.EMBEDDER_MODEL_PATH,  
            model_kwargs={
                "local_files_only": True
                }
            )
        #"trust_remote_code": True
        # self.embedding_model = SentenceTransformer(config.embedd)
        # self.embedding_model = OllamaEmbeddings(model = "nomic-embed-text")

        self.client_settings = Settings(
            is_persistent = True,
            persist_directory = "chroma_db",
            anonymized_telemetry = False
        )
        # self.collection_name = collection_name
        # self.client = chromadb.PersistentClient(path="chroma_db", settings=self.client_settings)
        print("embedding model and chroma client initialized...")


    def document_loader(self, data_directory):
        documents =[]
        for root, _, files in os.walk(data_directory):
            for file in files:
                file_path = os.path.join(root, file)
                _, file_extension = os.path.splitext(file_path)

                if file_extension == ".txt":
                    try:
                        loader = TextLoader(file_path)
                        documents.extend(loader.load())
                        logger.info(f"Loaded {file_path} with TextLoader")
                    except Exception as e:
                        logger.info(f"Error loading {file_path} with Textloader\n{e}")
                elif file_extension == ".pdf":
                    try:
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load())
                        logger.info(f"Loaded {file_path} with PyPDFLoader")
                    except Exception as e:
                        logger.info(f"Error Loading {file_path} with PyPDFLoader\n{e}")
                elif file_extension == ".json":
                    try:
                        loader = JSONLoader(file_path, jq_schema=".", text_content=False)
                        docs = loader.load()
                        formatted_doc = []
                        for doc in docs:
                            metadata = doc.metadata
                            data = doc.page_content
                            data = json.loads(data)

                            text = (f"System: {data[0].get('system')}\n"
                                    f"Description: {data[0].get('description')}\n"
                                    f"Severity: {data[0].get('severity')}\n"
                                    f"Cause: {data[0].get('cause')}\n"
                                    f"Solution: {data[0].get('solution')}\n"
                                    )
                            
                            doc.page_content = text
                            formatted_doc.append(doc)
                            # formatted_doc.append(doc.page_content)
                        documents.extend(formatted_doc)
                        logger.info(f"Loaded {file_path} with JSONLoader")
                    except Exception as e:
                        logger.info(f"Error Loading {file_path} with JSONLoader\n{e}")
                else:
                    logger.info(f"Skippig unsupported files : {file_path}")

        logger.info(f"Loaded {len(documents)} documents in total.")
        print(f"Loaded {len(documents)} documents in total.")
        return documents
    
    def chunk_text(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size = config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")
            print(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.info(f"Error splitting text: {e}")
            return None

    def embed_and_store(self, chunks, collection_name):
        
        #Cannot use this because chroma client is still holding the same data
        #thats why using chroma API to clear data from collection
        # persist_directory = "chroma_db"
        # if os.path.exists(persist_directory):
        #     shutil.rmtree(persist_directory)
        #     print(f"Old ChromaDB deleted from: {persist_directory}")
        #     logging.info(f"Old ChromaDB deleted from: {persist_directory}")
        
        
        client_collection = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory="chroma_db",
            client_settings=self.client_settings
        )
        
        client_collection.add_documents(chunks)

        print(f"added {len(chunks)} chunks in collection: {collection_name}")
            # db = Chroma.from_documents(chunks, embedding=self.embedding_model, client_settings=client_settings)
            # db.persist() #not required anymore
        # return client_collection

    def retrieve_from_collection(self, collection_name, user_query, k=3, filters=None):

        client_collection = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory="chroma_db",
            client_settings=self.client_settings
        )

        collection_info = client_collection.get()
        if len(collection_info["documents"]) == 0:
            print(f"Collection '{collection_name}' is empty. No retrieval will be performed.")
            logger.info(f"Collection '{collection_name}' is empty. No retrieval will be performed.")

        search_kwargs = {"k":k}
        retriever = client_collection.as_retriever(search_kwargs=search_kwargs)

        results = retriever.invoke(user_query)

        retrieved_docs = [doc.page_content for doc in results]
        retrieved_metadata = [doc.metadata for doc in results]

        '''collection = self.client.get_or_create_collection(collection = collection_name)
        if filters:
            search_kwargs["where"] = filters
        
        results = collection.query(
            query_texts = [user_query],
            n_results = k,
            where = filters if filters else None
        )

        retrieved_docs = results["documents"][0]
        retrieved_metadata = results["metadatas"][0]
        '''
        context = "\n\n".join(retrieved_docs)
        
        sources = []
        for meta in retrieved_metadata:
            if 'source' in meta:
                sources.append(meta['source'])

        # unique_sources = list(set(sources))
        unique_sources_ordered = list(dict.fromkeys(sources))

        return context, unique_sources_ordered

if __name__ == "__main__":
    user_prompt = "What is SRS in Accelerator Physics?"
    # user_prompt = '"question": "identify the number of faults from humans?"'

    obj = RagSetup()
    documents = obj.document_loader(config.ACC_PY_DOC_DIR)
    chunks = obj.chunk_text(documents)
    obj.embed_and_store(chunks, config.DOMAININFO_COLLECTION)

    documents = obj.document_loader(config.DB_SCHEMA_DOC_DIR)
    chunks = obj.chunk_text(documents)
    obj.embed_and_store(chunks, config.DBSCHEMA_COLLECTION)

    retrieved_docs, retrieved_metadata = obj.retrieve_from_collection(config.DOMAININFO_COLLECTION, user_prompt)
    print(retrieved_docs)