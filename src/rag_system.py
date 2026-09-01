"""
RAG System components for vector database and retrieval
"""

import os
import logging
from typing import Optional, List
from dataclasses import dataclass
from utils import create_chromadb_client, delete_collection_safe, list_collections_safe, collection_exists

logger = logging.getLogger(__name__)


@dataclass
class AttackResult:
    """Container for attack result data"""
    query: str
    response: str
    poisoned: bool
    retrieval_docs: List[str]
    timestamp: float


class RAGSystem:
    """Handles RAG system setup and operations"""
    
    def __init__(self, config, embeddings, llm, collection_name: str = "rag_demo"):
        self.config = config
        self.embeddings = embeddings
        self.llm = llm
        self.collection_name = collection_name
        self.vectorstore = None
        self.qa_chain = None
        
        # Setup vector database
        self._initialize_vectorstore()
        
        # Create retrieval chain
        self._setup_qa_chain()
    
    def _initialize_vectorstore(self, create_new: bool = False):
        """Initialize or reinitialize the Chroma vectorstore"""
        from langchain_community.vectorstores import Chroma
        
        if create_new:
            try:
                # Use our utility function to create ChromaDB client
                client = create_chromadb_client(self.config.vector_db_path)
                # Use our utility function to safely delete the collection
                delete_collection_safe(client, self.collection_name)
            except Exception as e:
                logger.warning(f"Error during collection deletion: {str(e)}")
        else:
            # Check if the collection exists first (for initial setup)
            try:
                client = create_chromadb_client(self.config.vector_db_path)
                
                if collection_exists(client, self.collection_name):
                    logger.info(f"Collection '{self.collection_name}' already exists, will be reused")
                else:
                    logger.info(f"Collection '{self.collection_name}' will be created")
                    
            except (ConnectionError, FileNotFoundError, ValueError, ImportError) as e:
                logger.warning(f"Error checking collections: {e}")
        
        # Initialize the vector store
        self.vectorstore = Chroma(
            persist_directory=self.config.vector_db_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        
        return self.vectorstore
    
    def _setup_qa_chain(self):
        """Setup the RetrievalQA chain with current configuration"""
        # langchain 1.x moved RetrievalQA out of the `langchain` package into
        # the separate langchain_classic package (already pulled in
        # transitively by langchain-community); `langchain.chains` no longer
        # exists at all.
        from langchain_classic.chains import RetrievalQA
        
        search_kwargs = {}
        if self.config.top_k_retrieval is not None:
            search_kwargs["k"] = self.config.top_k_retrieval
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            ),
            return_source_documents=True
        )
    
    def refresh_chain(self):
        """Public method to reinitialize the QA chain after vector database changes"""
        self._setup_qa_chain()
    
    def setup_vector_database(self, include_poison: bool = False):
        """Populate the vector database with documents"""
        from rag_poisoning_corpus import create_benign_corpus, create_poisoned_document
        
        logger.info(f"Setting up vector database (poison: {include_poison})")
        print(f"📚 Setting up vector database (poison: {include_poison})...")
        
        # Create new vectorstore instance with clean collection
        self._initialize_vectorstore(create_new=True)
        
        # Add benign documents
        benign_docs = create_benign_corpus()
        self.vectorstore.add_documents(benign_docs)
        
        # Optionally add poisoned document
        if include_poison:
            poisoned_doc = create_poisoned_document()
            self.vectorstore.add_documents([poisoned_doc])
            print("☠️  Poisoned document injected!")
        
        # Persist the database
        self.vectorstore.persist()
        print(f"✅ Vector database setup complete ({len(benign_docs) + (1 if include_poison else 0)} documents)")
    
    def query(self, query_text: str) -> dict:
        """Execute a query against the RAG system"""
        return self.qa_chain.invoke({"query": query_text})
