#!/usr/bin/env python3
"""
RAG Poisoning Proof of Concept: The Hidden Parrot Attack (Refactored)
Demonstrates how malicious instructions can be embedded in vector databases
to compromise RAG system behavior.
"""

import logging
import argparse
import os
from typing import Optional

# Import our modular components (config.py handles warning suppression)
from config import Config
from utils import get_device, check_local_model_exists, create_embeddings
from llm_factory import LLMFactory
from rag_system import RAGSystem
from attack_demo import AttackDemo

logger = logging.getLogger("rag_demo")


class RAGPoisoningDemo:
    """Main demo class that orchestrates the attack demonstration"""
    
    def __init__(self, device: Optional[str] = None, provider: Optional[str] = None):
        """Initialize the RAG system components"""
        print("🔧 Initializing RAG components...")
        
        # Load configuration
        self.config = Config()
        self.provider = provider
        
        # Determine device
        if provider not in ['ollama', 'deepseek']:
            device = get_device(device)
            print(f"Using device: {device}")
        else:
            device = None
            print(f"Using provider: {provider}")
        
        # Check embedding model availability
        cache_folder = os.getenv('SENTENCE_TRANSFORMERS_HOME', './models/embedding')
        if check_local_model_exists(self.config.embedding_model, cache_folder):
            print(f"✅ Using locally cached embedding model: {self.config.embedding_model}")
        else:
            print(f"⚠️ Local embedding model not found, may need to download: {self.config.embedding_model}")
        
        # Initialize components
        self.embeddings = create_embeddings(self.config, device)
        self.llm = LLMFactory.create_llm(provider, self.config, device)
        self.rag_system = RAGSystem(self.config, self.embeddings, self.llm)
        self.attack_demo = AttackDemo(self.rag_system)
        
        print("✅ RAG system initialized successfully!")
    
    def run_demo(self, show_prompt: bool = False, payload: Optional[str] = None):
        """Run the complete attack demonstration"""
        return self.attack_demo.run_attack_demo(show_prompt=show_prompt, payload=payload)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="RAG Poisoning Demo")
    parser.add_argument('--infer',
                        choices=['cpu', 'cuda', 'darwin', 'ollama', 'openai-compat', 'deepseek'],
                        default=None,
                        help="Force inference device/provider: cpu, cuda, darwin (macOS MPS), "
                             "ollama, openai-compat (llama-server / LM Studio), or deepseek")
    parser.add_argument('--show-prompt', action='store_true',
                        help="Print the assembled prompt (system + retrieved context + question) "
                             "sent to the LLM for each query")
    parser.add_argument('--payload-file', metavar='FILE',
                        help="Use FILE's content as the poisoned document instead of the "
                             "built-in pirate-persona payload")
    args = parser.parse_args()

    payload = None
    if args.payload_file:
        with open(args.payload_file, 'r') as payload_file:
            payload = payload_file.read()

    # Handle different inference options
    if args.infer in ['ollama', 'openai-compat', 'deepseek']:
        device = None 
        provider = args.infer
    else:
        # Map 'darwin' to 'mps' for torch
        device = 'mps' if args.infer == 'darwin' else args.infer
        provider = None
    
    # Initialize and show configuration
    config = Config()
    config.print_config(provider)
    
    try:
        # Initialize and run demo
        logger.info("Initializing RAG Poisoning Demo")
        demo = RAGPoisoningDemo(device=device, provider=provider)
        
        # Run the complete demonstration
        clean_results, poisoned_results = demo.run_demo(show_prompt=args.show_prompt, payload=payload)
        
        print("\n🎯 Demo completed successfully!")
        print("This demonstrates the vulnerability of RAG systems to prompt injection via poisoned embeddings.")
        
    except (FileNotFoundError, ImportError, ValueError, ConnectionError) as e:
        print(f"❌ Error during demo: {str(e)}")
        raise
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        return
    except Exception as e:
        print(f"❌ Unexpected error during demo: {str(e)}")
        logger.exception("Unexpected error in main demo")
        raise


if __name__ == "__main__":
    main()
