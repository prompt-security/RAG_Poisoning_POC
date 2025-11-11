"""
LLM factory for creating different types of language models
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory class for creating different types of LLMs"""
    
    @staticmethod
    def create_llm(provider: Optional[str], config, device: Optional[str] = None):
        """Create an LLM instance based on the provider"""
        
        if provider == 'ollama':
            return LLMFactory._create_ollama_llm(config)
        elif provider == 'deepseek':
            return LLMFactory._create_deepseek_llm(config)
        else:
            return LLMFactory._create_llamacpp_llm(config, device)
    
    @staticmethod
    def _create_ollama_llm(config):
        """Create Ollama LLM instance"""
        print(f"Using Ollama model: {config.ollama_model} at {config.ollama_base_url}")
        
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model="llama3:8b-instruct-q5_0",  # Use smaller model for faster responses
            openai_api_base=f"{config.ollama_base_url}/v1",
            openai_api_key="dummy-key",
            temperature=0.7,
            max_tokens=256  # Limit token count for faster responses
        )
    
    @staticmethod
    def _create_deepseek_llm(config):
        """Create DeepSeek LLM instance"""
        # Ensure API key is set
        deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek models. Please add it to your .keys file.")
        
        print(f"Using DeepSeek model: {config.deepseek_model}")
        
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=config.deepseek_model,
            openai_api_base="https://api.deepseek.com/v1",
            openai_api_key=deepseek_api_key
        )
    
    @staticmethod
    def _create_llamacpp_llm(config, device):
        """Create LlamaCpp LLM instance"""
        print(f"Using LlamaCpp model: {config.llama_model_path}")
        
        from langchain_community.llms import LlamaCpp
        
        llama_kwargs = {
            'model_path': config.llama_model_path,
            'n_ctx': 4096,
            'n_threads': 4,
            'verbose': False
        }
        
        if device == 'cuda':
            llama_kwargs['n_gpu_layers'] = 32  # or adjust as needed
        elif device == 'mps':
            llama_kwargs['n_gpu_layers'] = 1
            llama_kwargs['use_mlock'] = False

        return LlamaCpp(**llama_kwargs)
