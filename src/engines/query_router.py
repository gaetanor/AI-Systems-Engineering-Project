""""
src/engines/query_router.py

Query Router Module.
Dispatches user queries to the appropriate engine.
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import os
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq 
from langchain_openai import ChatOpenAI

# Internal Modules
from src.core.config_manager import AppConfig
from src.core.guardrails import PIIGuardrail
from src.engines.single_react import SingleAgentEngine
from src.engines.multi_supervisor import SupervisorNetwork

class QueryOrchestrator:
    """
    Main entry point for the chat interface.
    Decides execution strategy based on the 'system_mode' flag.
    """

    @staticmethod
    def _contextualize_query(user_query: str, chat_history: List[BaseMessage], settings: AppConfig) -> str:
        """
        Takes the chat history and the latest user question and reformulates it.
        Useful for fallback modes or dumb agents.
        """
        if not chat_history:
            return user_query

        llm = ChatGroq(
            model=settings.model_id, 
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

        system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        prompt_messages = [
            SystemMessage(content=system_prompt),
            *chat_history,
            HumanMessage(content=user_query)
        ]

        try:
            response = llm.invoke(prompt_messages)
            return response.content
        except Exception as e:
            return user_query

    @staticmethod
    def dispatch_request(
        user_query: str, 
        settings: AppConfig, 
        chat_history: List[BaseMessage] = [], 
        metadata_filter: dict = None,
        consistency_check: bool = False,
        use_expansion: bool = False,
        enable_guardrails: bool = False,
        trace_reasoning: bool = False
    ) -> Tuple[str, List[Document], Optional[str]]:
        
        # --- 🛡️ GUARDRAILS LAYER ---
        guardrail_log = ""
        clean_query = user_query

        if enable_guardrails:
            is_safe, sanitized_text, detected_types = PIIGuardrail.scan_and_redact(user_query)
            if not is_safe:
                clean_query = sanitized_text
                guardrail_log = f"\n🛡️ **Guardrails Active**: Sensitive data redacted ({', '.join(detected_types)}).\n"
        
        mode = settings.system_mode
        final_trace = guardrail_log

        # --- ROUTE 1: Multi-Agent Supervisor (Task B) ---
        if mode == "supervisor_agent":
            answer, docs, trace = SupervisorNetwork.execute(
                query=clean_query, 
                config=settings, 
                chat_history=chat_history, 
                return_trace=trace_reasoning,
                use_expansion=use_expansion
            )
            if trace: final_trace += "\n" + trace
            return answer, docs, final_trace

        # --- ROUTE 2: Single ReAct Agent (Task A) ---
        elif mode == "react_agent" or mode == "standard_rag":
            
            # FIX: Deleghiamo la gestione della history direttamente al SingleAgentEngine
            # Non usiamo più _contextualize_query qui, perché SingleAgentEngine ha la sua Fase 0.
            
            answer, docs, trace = SingleAgentEngine.execute(
                query=clean_query, # Passiamo la query originale (pulita)
                config=settings, 
                chat_history=chat_history, # <--- ORA È ATTIVO (decommentato)
                # kwargs passati a SingleAgentEngine
                metadata_filter=metadata_filter, # Se supportato dal tuo codice, altrimenti verrà ignorato in **kwargs
                consistency_check=consistency_check,
                use_expansion=use_expansion,
                return_trace=trace_reasoning
            )
            if trace: final_trace += "\n" + trace
            return answer, docs, final_trace

        else:
            # Fallback per modalità sconosciute
            return SingleAgentEngine.execute(query=clean_query, config=settings)