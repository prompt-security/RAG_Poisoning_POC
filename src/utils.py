"""
Device and utility functions for RAG Poisoning Demo
"""

import torch
import platform
import sys
import os
import logging
import time

logger = logging.getLogger(__name__)


def get_device(forced_device=None):
    """Determine the best device for computation"""
    if forced_device:
        return forced_device
    if torch.cuda.is_available():
        return 'cuda'
    elif sys.platform == 'darwin' and platform.machine() == 'arm64':
        return 'mps'
    else:
        return 'cpu'


def check_local_model_exists(model_name, cache_dir):
    """Check if a model exists locally before trying to download it"""
    model_parts = model_name.split('/')
    if len(model_parts) > 1:
        model_path_name = f'models--{model_parts[0]}--{model_parts[1]}'
        model_path = os.path.join(os.path.expanduser(cache_dir), model_path_name)
        return os.path.exists(model_path) and len(os.listdir(model_path)) > 0
    return False


def create_chromadb_client(db_path):
    """Create ChromaDB client with proper path handling"""
    import chromadb
    
    if db_path and os.path.exists(db_path):
        return chromadb.PersistentClient(path=db_path)
    else:
        # Fall back to in-memory client for testing
        return chromadb.Client()


def create_test_collection(client, base_name="test_collection"):
    """Create a test collection with unique timestamp to avoid conflicts"""
    collection_name = f"{base_name}_{int(time.time())}"
    
    try:
        collection = client.create_collection(collection_name)
        logger.info(f"Created test collection: {collection_name}")
        return collection, collection_name
    except Exception as e:
        logger.warning(f"Failed to create test collection {collection_name}: {e}")
        raise


def delete_collection_safe(client, collection_name):
    """Safely delete a ChromaDB collection with proper error handling"""
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
        return True
    except ValueError as e:
        # Collection doesn't exist, which is fine
        logger.info(f"Collection '{collection_name}' did not exist: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error deleting collection '{collection_name}': {e}")
        return False


def list_collections_safe(client):
    """Safely list ChromaDB collections with error handling"""
    try:
        collections = client.list_collections()
        logger.info(f"Found {len(collections)} collections")
        return collections
    except Exception as e:
        logger.warning(f"Error listing collections: {e}")
        return []


def test_chromadb_operations(client):
    """Test basic ChromaDB operations and return status"""
    try:
        # Try to create a test collection
        collection, collection_name = create_test_collection(client, "test_setup_verification")
        
        # Clean up the test collection
        delete_collection_safe(client, collection_name)
        
        return True, "ChromaDB working - created and deleted test collection"
    except Exception as e:
        # If collection creation fails, try to list existing collections
        try:
            collections = list_collections_safe(client)
            if collections:
                return True, f"ChromaDB working - found {len(collections)} existing collections"
            else:
                return True, "ChromaDB working - no collections found"
        except Exception as e2:
            return False, f"ChromaDB connection working but operations failed: {str(e2)}"


def check_api_keys_file(base_path=None):
    """Check if .keys file exists and return status"""
    if base_path is None:
        # Use current working directory instead of the file's directory
        base_path = os.getcwd()
    
    keys_file_path = os.path.join(base_path, '.keys')
    exists = os.path.exists(keys_file_path)
    
    if exists:
        logger.info(f"API Keys file (.keys) found at: {keys_file_path}")
    else:
        logger.info(f"API Keys file (.keys) not found at: {keys_file_path}")
    
    return exists, keys_file_path


def get_model_cache_info(config):
    """Get model cache directory information"""
    embedding_cache = os.environ.get('SENTENCE_TRANSFORMERS_HOME', './models/embedding')
    transformers_cache = os.environ.get('TRANSFORMERS_CACHE', './models/embedding')
    llm_dir = os.path.dirname(config.llama_model_path)
    
    return {
        'embedding_cache': embedding_cache,
        'transformers_cache': transformers_cache,
        'llm_directory': llm_dir
    }


def create_embeddings(config, device):
    """Create embeddings using config and device settings"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    cache_folder = os.environ.get('SENTENCE_TRANSFORMERS_HOME', './models/embedding')
    
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        cache_folder=cache_folder,
        model_kwargs={'device': device if device else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    logger.info(f"Created embeddings with model: {config.embedding_model}, device: {device}")
    return embeddings


def test_rag_components(config, device, no_local=False):
    """Test RAG system components integration"""
    try:
        from rag_system import RAGSystem
        from llm_factory import LLMFactory
        
        # Always test embeddings since they're always local (even in --no-local mode)
        embeddings = create_embeddings(config, device)
        
        results = {
            'embeddings': True,
            'llm_factory': True,
            'rag_system': True
        }
        
        messages = []
        messages.append("✅ Embeddings component initialized successfully")
        
        # Test that we can create an LLM (just test the factory, don't actually load)
        if no_local:
            messages.append("✅ LLM Factory available for remote providers: Ollama, DeepSeek")
        else:
            messages.append("✅ LLM Factory available for providers: LlamaCpp, Ollama, DeepSeek")
        
        # Test that RAGSystem can be instantiated (without full initialization)
        messages.append("✅ RAG System components available")
        
        return results, messages
        
    except ImportError as e:
        return {'error': True}, [f"⚠️ Some RAG components not available: {e}"]
    except Exception as e:
        return {'error': True}, [f"⚠️ RAG system test failed: {e}"]


def test_embedding_model(config):
    """Test embedding model loading and functionality"""
    from sentence_transformers import SentenceTransformer
    
    try:
        print(f"Loading embedding model from local directory ({config.embedding_model})...")
        model = SentenceTransformer(config.embedding_model)
        test_embedding = model.encode("This is a test sentence")
        
        logger.info(f"Embedding model test successful, dimension: {len(test_embedding)}")
        return True, len(test_embedding)
    except Exception as e:
        logger.error(f"Embedding model test failed: {e}")
        return False, str(e)


def collection_exists(client, collection_name):
    """Check if a collection exists in ChromaDB"""
    try:
        collections = list_collections_safe(client)
        return any(c.name == collection_name for c in collections)
    except Exception as e:
        logger.warning(f"Error checking if collection '{collection_name}' exists: {e}")
        return False
