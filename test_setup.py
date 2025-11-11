#!/usr/bin/env python3
"""Test script to verify the RAG setup using our modular components"""

import sys
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test RAG system setup')
parser.add_argument('--no-local', action='store_true', 
                    help='Skip local model checks (for remote inference)')
args = parser.parse_args()

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from utils import (
    get_device, 
    check_local_model_exists, 
    create_chromadb_client,
    test_chromadb_operations,
    check_api_keys_file,
    get_model_cache_info,
    test_rag_components,
    test_embedding_model
)
import time

# Initialize configuration (this handles environment loading, logging setup, etc.)
print("🔧 Initializing configuration...")
if args.no_local:
    print("🔧 Running in --no-local mode (skipping local model checks)")
config = Config()
print("✅ Configuration loaded successfully")

try:
    # Test core dependencies - moved to utilities where possible
    print("✅ Core dependencies imported successfully")
    
    # Detect device using our utility function
    device = get_device()
    print(f"✅ Device detected: {device}")
    
    # Check if embedding model exists locally using our utility (embeddings are always local)
    model_exists = check_local_model_exists(config.embedding_model, os.environ.get('SENTENCE_TRANSFORMERS_HOME', './models/embedding'))
    if model_exists:
        print(f"✅ Local embedding model found: {config.embedding_model}")
    else:
        print(f"⚠️ Local embedding model not found, may need to download: {config.embedding_model}")
    
    # Test embedding model using our utility function
    success, result = test_embedding_model(config)
    if success:
        print(f"✅ Embedding model working (dimension: {result})")
    else:
        print(f"❌ Embedding model test failed: {result}")
    
    # Check if local LLM exists (skip in --no-local mode since we'll use remote inference)
    if not args.no_local:
        if os.path.exists(config.llama_model_path):
            print(f"✅ Local LLM found at {config.llama_model_path}")
        else:
            print(f"⚠️ Local LLM not found at {config.llama_model_path} (optional for demo)")
    else:
        print("⏭️ Skipping local LLM check (--no-local mode - will use remote inference)")
        print("📡 Ready for remote inference providers (Ollama, DeepSeek, etc.)")
    print(f"Testing ChromaDB with path: {config.vector_db_path}")
    # Create ChromaDB client using our utility function
    client = create_chromadb_client(config.vector_db_path)
    
    # Test ChromaDB operations using our utility function
    success, message = test_chromadb_operations(client)
    if success:
        print(f"✅ {message}")
    else:
        print(f"⚠️  {message}")
    
    # Check if .keys file exists using our utility
    keys_exists, keys_path = check_api_keys_file()
    if keys_exists:
        print("✅ API Keys file (.keys) found")
    else:
        print("⚠️ API Keys file (.keys) not found. Create this file to store your API keys.")
    
    # Verify local model cache directories using our utility
    cache_info = get_model_cache_info(config)
    print(f"✅ Embedding cache: {cache_info['embedding_cache']}")
    print(f"✅ Transformers cache: {cache_info['transformers_cache']}")
    print(f"✅ LLM directory: {cache_info['llm_directory']}")
    
    # Print configuration using our Config object
    print("\n📋 Configuration Summary:")
    config.print_config()  # This will show current configuration
    
    # Test RAG System components integration using our utility
    print("\n🧪 Testing RAG System components...")
    results, messages = test_rag_components(config, device, args.no_local)
    for message in messages:
        print(message)
    
    print("\n🎉 Setup verification successful!")
    if args.no_local:
        print("📡 Ready for RAG demonstration with remote inference!")
        print("💡 Suggested providers: Ollama, DeepSeek")
    else:
        print("🏴‍☠️ Ready for RAG poisoning demonstration!")
    
except Exception as e:
    print(f"❌ Setup verification failed: {e}")
    sys.exit(1)
