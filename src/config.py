"""
Configuration management for RAG Poisoning Demo
Handles environment variables, logging setup, and API key loading.
"""

import os
import logging
import warnings
from dotenv import load_dotenv

# Suppress warnings early
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass  # LangChain not installed or different version


class Config:
    """Configuration manager for the RAG demo"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self._setup_model_cache_env()
        self._load_api_keys()
        self._setup_logging()
    
    def _setup_model_cache_env(self):
        """Setup model cache environment variables"""
        os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE', './models/embedding')
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.getenv('SENTENCE_TRANSFORMERS_HOME', './models/embedding')
        os.environ['HF_HOME'] = os.getenv('SENTENCE_TRANSFORMERS_HOME', './models/embedding')
        os.environ['HF_DATASETS_CACHE'] = os.getenv('SENTENCE_TRANSFORMERS_HOME', './models/embedding')
        os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Force offline mode
    
    def _load_api_keys(self):
        """Load API keys from .keys file"""
        keys_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.keys')
        if os.path.exists(keys_path):
            with open(keys_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")
        else:
            print("⚠️ No .keys file found. Some API services might not work.")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Suppress ChromaDB telemetry
        logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
        
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        log_file = os.getenv('LOG_FILE', './logs/rag_demo.log')
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Setup logging
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    @property
    def llama_model_path(self):
        return os.getenv('LLAMA_MODEL_PATH', "./models/llm/Phi-3.5-mini-instruct.Q4_K_M.gguf")
    
    @property
    def embedding_model(self):
        return os.getenv('EMBEDDING_MODEL', "sentence-transformers/all-MiniLM-L6-v2")
    
    @property
    def vector_db_path(self):
        return os.getenv('VECTOR_DB_PATH', "./data/chroma_db")
    
    @property
    def ollama_base_url(self):
        return os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    @property
    def ollama_model(self):
        return os.getenv('OLLAMA_MODEL', 'llama2')
    
    @property
    def deepseek_model(self):
        return os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
    
    @property
    def top_k_retrieval(self):
        top_k_str = os.getenv('TOP_K_RETRIEVAL')
        return int(top_k_str) if top_k_str and top_k_str.strip() else None
    
    @property
    def similarity_threshold(self):
        sim_threshold_str = os.getenv('SIMILARITY_THRESHOLD')
        return float(sim_threshold_str) if sim_threshold_str and sim_threshold_str.strip() else None
    
    def print_config(self, provider=None):
        """Print current configuration"""
        print("\n📋 Configuration from Environment:")
        print(f"  • Embedding Model: {self.embedding_model}")
        print(f"  • Vector DB Path: {self.vector_db_path}")
        
        if provider == 'ollama':
            print(f"  • LLM Provider: Ollama")
            print(f"  • Ollama Base URL: {self.ollama_base_url}")
            print(f"  • Ollama Model: {self.ollama_model}")
        elif provider == 'deepseek':
            print(f"  • LLM Provider: DeepSeek")
            print(f"  • DeepSeek Model: {self.deepseek_model}")
            print(f"  • DeepSeek API Key: {'CONFIGURED' if 'DEEPSEEK_API_KEY' in os.environ else 'MISSING'}")
        else:
            print(f"  • LLM Model Path: {self.llama_model_path}")
        
        print(f"  • Top K Retrieval: {'ENABLED: ' + str(self.top_k_retrieval) if self.top_k_retrieval else 'DISABLED (using default ChromaDB retrieval)'}")
        print(f"  • Similarity Threshold: {'ENABLED: ' + str(self.similarity_threshold) if self.similarity_threshold else 'DISABLED (waiting for weaviate support)'}")
        print(f"  • Log Level: {os.getenv('LOG_LEVEL', '(default)')}")
        print(f"  • Log File: {os.getenv('LOG_FILE', '(default)')}\n")
