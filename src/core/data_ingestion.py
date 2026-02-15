"""
src/core/data_ingestion.py

Unified Data Pipeline Module.
Combines responsibilities for:
1. Ingesting and normalizing raw legal JSON corpora (Italy, Estonia, Slovenia).
2. Instantiating embedding models for vectorization.

This consolidation simplifies the architecture by keeping all "data preparation" logic in one place.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# LangChain & Model Imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Internal Imports
from src.core.config_manager import AppConfig

# =============================================================================
# SECTION 1: DOCUMENT LOADING (CORPUS INGESTION)
# =============================================================================

class LegalContentIngestor:
    """
    Service class to handle the ingestion of legal corpora.
    Normalizes heterogeneous JSON structures into a standard Document format.
    """

    @staticmethod
    def ingest_corpus(source_directories: List[str]) -> List[Document]:
        """
        Main entry point. Recursively walks through provided directories 
        to find and parse legal JSON files.
        """
        dataset_buffer: List[Document] = []

        for directory in source_directories:
            root_path = Path(directory)
            if not root_path.exists():
                print(f"[Ingestor Warning] Source directory not found: {directory}")
                continue

            # Recursive search for all .json files
            for file_path in root_path.rglob("*.json"):
                try:
                    documents = LegalContentIngestor._process_single_file(file_path)
                    dataset_buffer.extend(documents)
                except Exception as e:
                    print(f"[Ingestor Error] Failed to process {file_path.name}: {e}")

        print(f"[Ingestor Info] Total legal documents ingested: {len(dataset_buffer)}")
        return dataset_buffer

    @staticmethod
    def _process_single_file(file_path: Path) -> List[Document]:
        """
        Reads a specific JSON file and converts its content (list or dict) 
        into a list of Documents.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError:
                print(f"[Ingestor Error] Invalid JSON in {file_path}")
                return []

        docs = []
        
        # Handle case where JSON is a list of records
        if isinstance(raw_data, list):
            for entry in raw_data:
                if isinstance(entry, dict):
                    doc = LegalContentIngestor._map_record_to_document(entry, str(file_path))
                    if doc: docs.append(doc)
        
        # Handle case where JSON is a single object
        elif isinstance(raw_data, dict):
            doc = LegalContentIngestor._map_record_to_document(raw_data, str(file_path))
            if doc: docs.append(doc)

        return docs

    @staticmethod
    def _map_record_to_document(record: Dict[str, Any], origin_source: str) -> Optional[Document]:
        """
        Transforms a raw dictionary into a structured Document.
        Injects critical metadata for RAG routing (Country, Legal Domain).
        """
        # 1. Identify the main text content
        text_body = record.get("content") or record.get("text") or record.get("description")
        
        if not text_body:
            return None
        
        if not isinstance(text_body, str):
            text_body = str(text_body)

        # 2. Extract and Normalize Metadata
        existing_meta = record.get("metadata", {})
        if not isinstance(existing_meta, dict):
            existing_meta = {"raw_meta": str(existing_meta)}

        # 3. Metadata Enrichment for Routing (Task B requirement)
        enriched_meta = {
            "source": origin_source,
            "filename": os.path.basename(origin_source),
            **existing_meta
        }

        # --- LOGICA CORRETTA PER RILEVARE IL PAESE DAL PATH ---
        path_lower = origin_source.lower()
        
        # Default
        country = "General"
        
        if "italy" in path_lower or "italia" in path_lower:
            country = "Italy"
        elif "estonia" in path_lower:
            country = "Estonia"
        elif "slovenia" in path_lower:
            country = "Slovenia"
        
        # Sovrascriviamo il paese nei metadati
        enriched_meta["country"] = country
        
        # Logica opzionale per il dominio legale
        if "divorce" in path_lower:
            enriched_meta["legal_domain"] = "Divorce"
        elif "inheritance" in path_lower:
            enriched_meta["legal_domain"] = "Inheritance"

        return Document(page_content=text_body, metadata=enriched_meta)


# =============================================================================
# SECTION 2: VECTORIZATION (EMBEDDINGS FACTORY)
# =============================================================================

class EmbeddingFactory:
    """
    Factory class to generate embedding model instances based on global configuration.
    Supports switching between local CPU-based models (HuggingFace) and API-based models (OpenAI).
    """

    @staticmethod
    def create_embedding_model(configuration: AppConfig) -> Embeddings:
        """
        Main factory method.
        Reads the configuration state and returns the appropriate LangChain Embeddings wrapper.
        """
        backend_type = configuration.embedding_backend.lower()
        model_id = configuration.embedding_model_path

        # Case 1: OpenAI API Embeddings (Cost involved)
        if backend_type == "openai":
            print(f"[Embedding Service] Initializing OpenAI Embeddings: {model_id}")
            return OpenAIEmbeddings(model=model_id)

        # Case 2: Local HuggingFace Embeddings (Free, CPU-optimized)
        # Preferred default for the exam project to ensure reproducibility without costs.
        print(f"[Embedding Service] Initializing Local HuggingFace Model: {model_id}")
        
        return HuggingFaceEmbeddings(
            model_name=model_id,
            # Force CPU execution to prevent "meta tensor" errors on non-CUDA devices
            model_kwargs={'device': 'cpu'},
            # Normalize vectors to ensure cosine similarity is calculated correctly
            encode_kwargs={'normalize_embeddings': True}
        )

    #MODIFICARE PER USARE SEMPRE OPENAI CON RAGAS
    @staticmethod
    def get_ragas_embeddings(configuration: AppConfig) -> Embeddings:
        """
        Specific instantiator for RAGAS evaluation metrics.
        Can be customized if the evaluation requires a stronger model than the retrieval one.
        """
        return EmbeddingFactory.create_embedding_model(configuration)


# =============================================================================
# COMPATIBILITY WRAPPERS (Optional, for smoother migration)
# =============================================================================

#def load_documents_from_folders(folders: List[str]) -> List[Document]:
#    """Wrapper for legacy calls to document loading."""
#    return LegalContentIngestor.ingest_corpus(folders)

#def get_embedding_model(config: AppConfig) -> Embeddings:
#    """Wrapper for legacy calls to embedding creation."""
#    return EmbeddingFactory.create_embedding_model(config)