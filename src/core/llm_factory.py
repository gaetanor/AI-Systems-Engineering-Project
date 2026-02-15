"""
src/core/llm_factory.py

LLM Factory Module.
Abstracts the instantiation of Language Models.
"""

import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.core.config_manager import AppConfig

class LLMFactory:
    """
    Factory class to create LLM instances based on configuration.
    """

    @staticmethod
    def create_llm(config: AppConfig) -> Optional[BaseChatModel]:
        """
        Main entry point. Dispatches the request to the specific provider initializer.
        """
        provider = config.llm_tech_stack.lower()
        model_name = config.llm_model_name

        if provider == "openai":
            return LLMFactory._init_gpt_service(model_name, config)
        
        elif provider == "groq":
            return LLMFactory._init_groq_service(model_name, config)
            
        elif provider == "huggingface":
            return LLMFactory._init_huggingface_service(model_name, config)
        
        else:
            print(f"[LLM Factory] Unsupported provider: {provider}")
            return None

    # --- SPECIFIC INITIALIZERS ---

    @staticmethod
    def _init_gpt_service(model_name: str, config: AppConfig) -> BaseChatModel:
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens
        )

    @staticmethod
    def _init_groq_service(model_name: str, config: AppConfig) -> BaseChatModel:
        return ChatGroq(
            model_name=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens
        )

    @staticmethod
    def _init_huggingface_service(repo_id: str, config: AppConfig) -> Optional[BaseChatModel]:
        if not repo_id:
            return None

        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            print("❌ [LLM Factory] CRITICAL ERROR: API Key not found!")
            return None

        print(f"[LLM Factory] Loading HuggingFace via Router API: {repo_id}")
        
        try:
            return ChatOpenAI(
                model=repo_id,
                openai_api_key=hf_token,
                base_url="https://router.huggingface.co/v1",
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                timeout=120
            )
        
        except Exception as e:
            print(f"[LLM Factory] Error initializing HF model: {e}")
            return None