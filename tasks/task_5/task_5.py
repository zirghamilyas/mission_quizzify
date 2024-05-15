import sys
import os
import streamlit as st
sys.path.append(os.path.abspath('../../'))

import chromadb
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient


# Import Task libraries
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

class ChromaCollectionCreator:
    def __init__(self, processor, embed_model):
        """
        Initializes the ChromaCollectionCreator with a DocumentProcessor instance and embeddings configuration.
        :param processor: An instance of DocumentProcessor that has processed documents.
        :param embeddings_config: An embedding client for embedding documents.
        """
        print("Task 5 initialized")

        self.processor = processor      # This will hold the DocumentProcessor from Task 3
        self.embed_model = embed_model  # This will hold the EmbeddingClient from Task 4
        self.db = None                  # This will hold the Chroma collection
    
    def create_chroma_collection(self):
        a = """
        Creates a Chroma collection from the documents processed by the DocumentProcessor instance.
        
        Steps:
        1. Check if any documents have been processed by the DocumentProcessor instance. If not, display an error message using streamlit's error widget.
        2. Split the processed documents into text chunks suitable for embedding and indexing using the CharacterTextSplitter from Langchain.
        3. Create a Chroma collection in memory with the text chunks and the embeddings model initialized in the class.
        
        Returns:
            None

        
        Task: Create a Chroma collection from the documents processed by the DocumentProcessor instance.
        
        Steps:
        1. Check if any documents have been processed by the DocumentProcessor instance. If not, display an error message using streamlit's error widget.
        
        2. Split the processed documents into text chunks suitable for embedding and indexing. Use the CharacterTextSplitter from Langchain to achieve this. You'll need to define a separator, chunk size, and chunk overlap.
        https://python.langchain.com/docs/modules/data_connection/document_transformers/
        
        3. Create a Chroma collection in memory with the text chunks obtained from step 2 and the embeddings model initialized in the class. Use the Chroma.from_documents method for this purpose.
        https://python.langchain.com/docs/integrations/vectorstores/chroma#use-openai-embeddings
        https://docs.trychroma.com/getting-started
        embeddings = OpenAIEmbeddings()
        new_client = chromadb.EphemeralClient()
        openai_lc_client = Chroma.from_documents(
        docs, embeddings, client=new_client, collection_name="openai_collection"
        )

        Instructions:
        - Begin by verifying that there are processed pages available. If not, inform the user that no documents are found.
        
        - If documents are available, proceed to split these documents into smaller text chunks. This operation prepares the documents for embedding and indexing. Look into using the CharacterTextSplitter with appropriate parameters (e.g., separator, chunk_size, chunk_overlap).
        
        - Next, with the prepared texts, create a new Chroma collection. This step involves using the embeddings model (self.embed_model) along with the texts to initialize the collection.
        
        - Finally, provide feedback to the user regarding the success or failure of the Chroma collection creation.
        
        Note: Ensure to replace placeholders like [Your code here] with actual implementation code as per the instructions above.
        """
        
        # Step 1: Check for processed documents
        if len(self.processor.pages) == 0:
            st.error("No documents found!", icon="🚨")
            return

        # Step 2: Split documents into text chunks
        # Use a TextSplitter from Langchain to split the documents into smaller text chunks
        # https://python.langchain.com/docs/modules/data_connection/document_transformers/character_text_splitter

        text_data = []
        metadata = []
        for page in self.processor.pages:
            text_data.append(page.page_content)
            metadata.append(page.metadata)

         
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            )


        documents = text_splitter.create_documents(text_data, 
                                                   metadatas=metadata)
            
        if documents is not None:
            st.success(f"Successfully split pages to {len(documents)} documents!", icon="✅")

        # Step 3: Create the Chroma Collection
        # https://docs.trychroma.com/
        # Create a Chroma in-memory client using the text chunks and the embeddings model

        new_client = chromadb.EphemeralClient()
        self.db = Chroma.from_documents(
        documents, self.embed_model, client=new_client, collection_name="mission_quizzify_collection"
        )        

        if self.db:
            st.success("Successfully created Chroma Collection!", icon="✅")
        else:
            st.error("Failed to create Chroma Collection!", icon="🚨")

        return self.db

    def query_chroma_collection(self, query) -> Document:
        """
        Queries the created Chroma collection for documents similar to the query.
        :param query: The query string to search for in the Chroma collection.
        
        Returns the first matching document from the collection with similarity score.
        """
        if self.db:
            docs = self.db.similarity_search_with_relevance_scores(query)
            if docs:
                return docs[0]
            else:
                st.error("No matching documents found!", icon="🚨")
        else:
            st.error("Chroma Collection has not been created!", icon="🚨")


if __name__ == "__main__":
 
    with st.form("Load Data to Chroma"):
        st.write("Select PDFs for Ingestion, then click Submit")
        
        processor = DocumentProcessor() # Initialize from Task 3
        processor.ingest_documents()

        path = os.path.abspath('../../')
        key_path = path + '\\radicalai-420811-b300c28fb7ea.json'
        embed_config = {
            "model_name": "textembedding-gecko@003",
            "project": "radicalai-420811",
            "location": "us-central1",
            'key_path': key_path
        }
        embed_client = EmbeddingClient(**embed_config) # Initialize from Task 4

        chroma_creator = ChromaCollectionCreator(processor, embed_client) # Initialize from Task 5
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            chroma_creator.create_chroma_collection()
            print(chroma_creator.query_chroma_collection('Text to be matched'))
            