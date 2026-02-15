"""
src/core/session_logger.py

Audit Trail & Session Management.
Handles the persistence of chat sessions to JSON for:
1. Compliance with transparency requirements.
2. Offline evaluation using RAGAS metrics.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class SessionRecorder:
    """
    Manages the 'logs/interactions.json' file.
    Appends new Q&A pairs atomically to prevent data loss.
    """
    
    LOG_DIR = "logs"
    LOG_FILE = "interactions.json"

    @classmethod
    def get_log_path(cls) -> str:
        """Returns full path, creating the directory if missing."""
        if not os.path.exists(cls.LOG_DIR):
            os.makedirs(cls.LOG_DIR)
        return os.path.join(cls.LOG_DIR, cls.LOG_FILE)

    @classmethod
    def log_turn(
        cls, 
        user_query: str, 
        system_response: str, 
        retrieved_docs: List[Any],
        mode: str,
        trace_log: Optional[str] = None
    ) -> None:
        """
        Saves a single turn of conversation to the JSON log.
        """
        file_path = cls.get_log_path()
        
        # 1. Load existing history
        history = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, ValueError):
                history = [] # Reset if corrupted

        # 2. Format the new entry
        # We extract metadata from Document objects to make JSON serializable
        formatted_contexts = []
        for doc in retrieved_docs:
            formatted_contexts.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

        entry = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode, # 'single_agent' or 'multi_supervisor'
            "user_query": user_query,
            "system_response": system_response,
            "retrieved_contexts": formatted_contexts,
            "reasoning_trace": trace_log  # Optional: saved for debugging
        }

        history.append(entry)

        # 3. Write back to disk
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_sessions(cls) -> List[Dict[str, Any]]:
        """
        Reads the log file for the Evaluation Dashboard.
        """
        file_path = cls.get_log_path()
        if not os.path.exists(file_path):
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    @staticmethod
    def clear_logs(file_path: str = "logs/interactions.json") -> bool:
        """
        Cancella fisicamente il file di log JSON.
        """
        try:
            # Gestione percorsi diversi o file nella root
            if not os.path.exists(file_path):
                # Prova a cercare nella root se non è in logs/
                if os.path.exists("interactions.json"):
                    file_path = "interactions.json"
                else:
                    return False # Il file non esiste già
            
            os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error deleting log file: {e}")
            return False