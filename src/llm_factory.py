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
        elif provider == 'openai-compat':
            return LLMFactory._create_openai_compat_llm(config)
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
            model=config.ollama_model,
            openai_api_base=f"{config.ollama_base_url}/v1",
            openai_api_key="dummy-key",
            # temperature=0 and a short completion cap keep a live workshop
            # room's results reproducible and each query fast.
            temperature=0,
            max_tokens=128,
            # langchain-openai 1.x always sends the newer max_completion_tokens
            # field; Ollama's OpenAI-compat layer only recognizes the legacy
            # max_tokens (ollama/ollama#7125 is still open), so without this
            # the cap above would be silently ignored for Ollama specifically.
            # extra_body merges straight into the request body alongside it.
            extra_body={"max_tokens": 128}
        )
    
    @staticmethod
    def _create_openai_compat_llm(config):
        """
        Create an LLM against any OpenAI-compatible endpoint.

        Covers llama-server (llama.cpp) and LM Studio, which expose /v1 but are
        not Ollama. llama-server ignores the model field entirely, so the value
        only has to be non-empty for it.

        No credentials: every endpoint this demo supports is unauthenticated, so
        the key is the same literal placeholder the Ollama provider uses. If an
        authenticated endpoint is ever needed, add it here AND in preflight
        deliberately -- a half-configured bearer token is worse than none.
        """
        print(f"Using OpenAI-compatible endpoint: {config.openai_compat_base_url} "
              f"(model: {config.openai_compat_model})")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.openai_compat_model,
            openai_api_base=f"{config.openai_compat_base_url}/v1",
            openai_api_key="dummy-key",
            # temperature=0 and a short completion cap keep a live workshop
            # room's results reproducible and each query fast.
            temperature=0,
            max_tokens=128,
            # Belt-and-suspenders for LM Studio, untested here: langchain-openai
            # 1.x sends max_completion_tokens, and not every OpenAI-compat
            # server has caught up to that field name yet (see the ollama
            # branch above). llama-server already honours max_completion_tokens
            # (verified live), so this is a harmless no-op for it.
            extra_body={"max_tokens": 128}
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
