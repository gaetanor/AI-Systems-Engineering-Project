"""
backend/config.py

Configuration Management Module for Legal RAG System.
Handles global settings for LLM inference, vector storage paths, and retrieval parameters.
Designed to support Multi-jurisdictional (Italy, Estonia, Slovenia) document retrieval.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass
class AppConfig:
    """
    Central configuration state for the application.
    Managed via the Streamlit UI and consumed by backend services.
    """

    # ---------------- Inference Settings (LLM) ----------------
    # Defaulting to Groq for high-speed, free-tier inference
    # Options: "groq", "openai", "huggingface"
    llm_tech_stack: str = "openai"
    
    # Recommended models for Groq: "llama-3.3-70b-versatile" or "mixtral-8x7b-32768"
    model_id: str = "gpt-4o"

    # ---------------- Embedding Settings ----------------
    # Defaulting to a lightweight, high-performance local model to avoid API costs
    # MODIFICARE PER USARE EMBEDDING DI OPENAI (VALORE FISSO)
    embedding_backend: str = "openai"
    embedding_model_path: str = "text-embedding-3-small"

    # ---------------- Data Ingestion Paths ----------------
    # Directories containing the source JSON legal documents
    source_corpus_paths: List[str] = field(default_factory=list)

    # ---------------- Vector Database Configuration ----------------
    # Primary storage location for FAISS indices
    # Used for the Single-Agent architecture
    vector_db_root_path: str = "vector_store"
    
    # Active index path for the current session
    active_db_path: str = "vector_store/merged_legal_index"

    # Registry of specialized indices for the Multi-Agent architecture
    # e.g., ["vector_store/italy_civil_code", "vector_store/estonia_cases"]
    specialized_db_paths: List[str] = field(default_factory=list)

    # ---------------- Retrieval Parameters ----------------
    # Number of chunks to retrieve per query
    retrieval_k: int = 4
    
    # RERANKING CONFIGURATION
    enable_reranking: bool = False
    rerank_initial_fetch: int = 25  # Quanti documenti prendere prima del filtro

    # ---------------- Agent Architecture Mode ----------------
    # Defines the reasoning engine strategy:
    #   - "naive_rag": Direct context injection
    #   - "react_agent": ReAct logic (Thought/Action/Observation) [Task A]
    #   - "supervisor_agent": Hierarchical multi-agent routing [Task B]
    system_mode: str = "react_agent"

    # ---------------- Compatibility Layer ----------------
    # These properties ensure backward compatibility if other files 
    # reference the old variable names (llm_provider, etc.)
    
    def __init__(self):
        
        self.vector_db_root_path = "vector_store"
        
        # NUOVI PARAMETRI PER L'ESTENSIONE
        self.llm_temperature: float = 0.0  # Default deterministico
        self.llm_max_tokens: int = 4096
        self.retrieval_top_k: int = 4      # Quanti chunk recuperare

    @property
    def llm_provider(self):
        return self.llm_tech_stack

    @llm_provider.setter
    def llm_provider(self, value):
        self.llm_tech_stack = value

    @property
    def llm_model_name(self):
        return self.model_id

    @llm_model_name.setter
    def llm_model_name(self, value):
        self.model_id = value

    @property
    def embedding_provider(self):
        return self.embedding_backend

    @property
    def embedding_model_name(self):
        return self.embedding_model_path

    @property
    def vector_store_base_dir(self):
        return self.vector_db_root_path
    
    @property
    def vector_store_dir(self):
        return self.active_db_path

    @vector_store_dir.setter
    def vector_store_dir(self, value):
        self.active_db_path = value

    @property
    def top_k(self):
        return self.retrieval_k
        
    @property
    def agentic_mode(self):
        # Maps new mode names to old keys if necessary for frontend compatibility
        mapping = {
            "naive_rag": "standard_rag",
            "react_agent": "react",
            "supervisor_agent": "hybrid_legal" 
        }
        # Default fallback logic
        if self.system_mode in mapping.values(): return self.system_mode
        return mapping.get(self.system_mode, "standard_rag")

    @agentic_mode.setter
    def agentic_mode(self, value):
        self.system_mode = value

    # Multi-agent flag specifically for the pipeline router
    use_multiagent: bool = False
    
    # JSON folders aliases
    @property
    def json_folders(self):
        return self.source_corpus_paths

    @json_folders.setter
    def json_folders(self, value):
        self.source_corpus_paths = value