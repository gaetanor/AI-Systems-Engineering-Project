"""
backend/vector_store.py

Vector Database Management Subsystem.
Handles the lifecycle of FAISS indices: creation, partitioning, loading, and deletion.
Critical for supporting both Single-Agent (unified DB) and Multi-Agent (sharded DBs) architectures.
"""

from __future__ import annotations
import os
import shutil
import gc
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

class VectorArchivist:
    """
    Manager class for Vector Knowledge Bases.
    Maintains an in-memory cache of loaded indices to speed up query time.
    """
    
    # Cache condivisa a livello di classe per evitare ricaricamenti costosi
    _active_indices: Dict[str, FAISS] = {}

    @classmethod
    def create_unified_index(cls, documents: List[Document], embedder: Embeddings, save_path: str) -> None:
        """
        [Task A Strategy]
        Builds a single monolithic vector index containing all documents (Italy + Estonia + Slovenia).
        Used by the Single-Agent ReAct system.
        """
        if not documents:
            print("[VectorArchivist] No documents provided for indexing.")
            return

        print(f"[VectorArchivist] Building unified index at {save_path} with {len(documents)} docs...")
        
        # Ensure directory is clean
        os.makedirs(save_path, exist_ok=True)
        
        # Create and save FAISS index
        vector_db = FAISS.from_documents(documents, embedder)
        vector_db.save_local(save_path)
        
        # Update cache
        cls._active_indices[save_path] = vector_db
        print("[VectorArchivist] Unified index created successfully.")

    @classmethod
    def build_sharded_indices(cls, documents: List[Document], embedder: Embeddings, root_path: str) -> List[str]:
        """
        [Task B Strategy]
        Splits the unified document set into specialized sub-indices based on metadata (Country).
        Generates separate FAISS DBs for Italy, Estonia, and Slovenia to support specialized agents.
        
        Returns:
            List[str]: A list of paths where the specialized indices were saved.
        """
        print("[VectorArchivist] Starting shard generation for Multi-Agent Supervisor...")
        
        # 1. Group documents by Country
        shards: Dict[str, List[Document]] = {}
        for doc in documents:
            # Fallback to 'unknown' if metadata is missing, preventing crashes
            country_key = doc.metadata.get("country", "general_context").lower()
            if country_key not in shards:
                shards[country_key] = []
            shards[country_key].append(doc)
            
        created_paths = []

        # 2. Build independent indices for each group
        for country, docs in shards.items():
            shard_path = os.path.join(root_path, f"{country}_db")
            print(f"  -> Creating shard for '{country}' ({len(docs)} docs) at {shard_path}")
            
            os.makedirs(shard_path, exist_ok=True)
            
            shard_db = FAISS.from_documents(docs, embedder)
            shard_db.save_local(shard_path)
            
            cls._active_indices[shard_path] = shard_db
            created_paths.append(shard_path)
            
        return created_paths

    @classmethod
    def load_index(cls, path: str, embedder: Embeddings) -> Optional[FAISS]:
        """
        Retrieves a vector index. Returns from cache if available, otherwise loads from disk.
        """
        # Hit Cache
        if path in cls._active_indices:
            return cls._active_indices[path]

        # Miss Cache -> Load from Disk
        if not os.path.exists(path):
            print(f"[VectorArchivist] Warning: Index path not found: {path}")
            return None

        try:
            vector_db = FAISS.load_local(
                path, 
                embedder, 
                allow_dangerous_deserialization=True # Required for loading local pickle files
            )
            cls._active_indices[path] = vector_db
            return vector_db
        except Exception as e:
            print(f"[VectorArchivist] Error loading index at {path}: {e}")
            return None

    @classmethod
    def purge_index(cls, path: str) -> None:
        """
        Destructive operation: removes the index from both disk and memory.
        """
        # Remove from Memory
        if path in cls._active_indices:
            del cls._active_indices[path]
            gc.collect() # Force garbage collection to free RAM

        # Remove from Disk
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"[VectorArchivist] Deleted index at {path}")

# Wrapper functions to maintain compatibility with existing calls (if any remain)
def build_vector_store(docs, model, path):
    return VectorArchivist.create_unified_index(docs, model, path)

def load_vector_store(path, model):
    return VectorArchivist.load_index(path, model)

def clear_vector_store_cache(path):
    return VectorArchivist.purge_index(path)