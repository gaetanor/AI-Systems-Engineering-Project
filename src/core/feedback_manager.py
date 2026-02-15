"""
src/core/feedback_manager.py

Feedback Management Module (Data Flywheel).
Handles the storage of user feedback with ABSOLUTE PATHS to prevent I/O errors.
"""

import os
import json
import time
from typing import List, Dict, Any

class FeedbackManager:
    
    # --- FIX CRITICO: CALCOLO PERCORSO ASSOLUTO ---
    # 1. Prendi la cartella dove si trova questo file (src/core)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 2. Risali di due livelli per arrivare alla root del progetto (src/core -> src -> ROOT)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
    # 3. Costruisci il path per logs/feedback.jsonl
    FEEDBACK_FILE = os.path.join(PROJECT_ROOT, "logs", "feedback.jsonl")

    @staticmethod
    def save_feedback(
        user_query: str,
        system_response: str,
        retrieved_docs: List[Any],
        rating: str,  # "positive" or "negative"
        user_comment: str = "",
        model_name: str = "unknown"
    ) -> bool:
        """
        Appends a feedback interaction to the JSONL file.
        """
        print(f"📝 [FeedbackManager] Attempting to save feedback to: {FeedbackManager.FEEDBACK_FILE}")
        
        try:
            # 1. Serializzazione dei documenti
            serialized_docs = []
            
            # Fix per evitare crash se retrieved_docs è None
            if retrieved_docs: 
                for doc in retrieved_docs:
                    # Gestiamo sia oggetti Document che dict
                    if hasattr(doc, 'page_content'):
                        serialized_docs.append({
                            "content": doc.page_content,
                            "source": doc.metadata.get("filename", "unknown"),
                            "country": doc.metadata.get("country", "unknown")
                        })
                    else:
                        serialized_docs.append(doc)

            # 2. Creazione del record dati
            feedback_record = {
                "timestamp": time.time(),
                "timestamp_readable": time.strftime("%Y-%m-%d %H:%M:%S"), # Utile per lettura umana
                "model_used": model_name,
                "interaction": {
                    "user_query": user_query,
                    "system_response": system_response,
                    "retrieved_context": serialized_docs
                },
                "feedback": {
                    "rating": rating, # '👍' or '👎'
                    "correction_comment": user_comment
                }
            }

            # 3. Scrittura in Append (JSONL)
            # Assicuriamoci che la cartella logs esista nel path assoluto
            os.makedirs(os.path.dirname(FeedbackManager.FEEDBACK_FILE), exist_ok=True)
            
            with open(FeedbackManager.FEEDBACK_FILE, "a", encoding="utf-8") as f:
                # flush=True forza la scrittura immediata dal buffer al disco
                print(json.dumps(feedback_record, ensure_ascii=False), file=f, flush=True)
                
            print("✅ [FeedbackManager] Feedback saved successfully.")
            return True

        except Exception as e:
            print(f"❌ [Feedback Manager] CRITICAL ERROR saving feedback: {e}")
            return False