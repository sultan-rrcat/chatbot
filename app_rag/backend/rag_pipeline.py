import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import config
from preprocess import df


class FaultbookIngestor:
    """
    Handles the transformation of fault log DataFrame into embeddings
    and persists them into Chroma vector store for retrieval.
    """

    def __init__(self, dataframe: pd.DataFrame, persist_directory: str):
        self.df = dataframe
        self.persist_directory = persist_directory
        self.documents = []

        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDER_MODEL_PATH,
            model_kwargs={"local_files_only": True}
        )

    def row_to_chunk(self, row) -> str:
        """
        Converts a row into a structured chunk string for embeddings.
        """
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
        """
        Transforms the DataFrame into LangChain Document objects with metadata.
        """
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
        """
        Creates and persists the Chroma vector store.
        """
        if not self.documents:
            self.prepare_documents()

        vectorstore = Chroma.from_documents(
            self.documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("✅ Chroma vector store created and persisted at:", self.persist_directory)
        return vectorstore


if __name__ == "__main__":
    import config

    config.setup_logging()
    persist_dir = "new_rag_data/chroma_faultbook_index"

    ingestor = FaultbookIngestor(df, persist_directory=persist_dir)
    vectorstore = ingestor.ingest_to_chroma()
