"""
src/core/reranking.py

Reranking Module.
Implements a Cross-Encoder to refine vector retrieval results.
"""

from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class SemanticReranker:
    """
    Wrapper for a Cross-Encoder model to rerank retrieved documents.
    Uses 'ms-marco-MiniLM-L-6-v2' optimized for passage ranking.
    """
    
    _model = None # Singleton-like lazy loading

    @classmethod
    def get_model(cls):
        """Lazy loads the model to avoid overhead if reranking is disabled."""
        if cls._model is None:
            # Questo modello è piccolo (~80MB) e veloce su CPU
            print("🔄 Loading Cross-Encoder Model (Reranker)...")
            cls._model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return cls._model

    @staticmethod
    def rerank_documents(query: str, docs: List[Document], top_k: int = 4) -> List[Document]:
        """
        Reorders the document list based on semantic relevance to the query.
        """
        if not docs:
            return []

        model = SemanticReranker.get_model()
        
        # 1. Preparazione delle coppie [Query, Testo Documento]
        # Il Cross-Encoder vuole input tipo: [('Domanda', 'Doc1'), ('Domanda', 'Doc2')...]
        pairs = [[query, doc.page_content] for doc in docs]
        
        # 2. Predizione dei punteggi (relevance scores)
        scores = model.predict(pairs)
        
        # 3. Associazione Documento -> Score
        docs_with_scores = list(zip(docs, scores))
        
        # 4. Ordinamento decrescente (dal punteggio più alto)
        # x[1] è lo score
        sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        
        # 5. Selezione Top-K
        reranked_docs = [doc for doc, score in sorted_docs[:top_k]]
        
        return reranked_docs