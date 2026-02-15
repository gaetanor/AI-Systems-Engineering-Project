"""
src/engines/multi_supervisor.py

Multi-Agent Supervisor Engine
"""
from __future__ import annotations
import os
import json
import re
from typing import List, Tuple, Optional, Any, Dict, TYPE_CHECKING
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from src.core.llm_factory import LLMFactory
from src.core.data_ingestion import EmbeddingFactory
from src.engines.vector_ops import VectorArchivist
from src.core.reranking import SemanticReranker

if TYPE_CHECKING:
    from src.core.config_manager import AppConfig

class SupervisorNetwork:
    """
    Hierarchical Agent System
    """

    AGENT_REGISTRY = {
        "Italy_Agent": {
            "country": "Italy",
            "db_name": "italy_db",
            "description": "Expert in Italian Law. Handles civil, criminal, and administrative documents specifically within the Italian jurisdiction."
        },
        "Estonia_Agent": {
            "country": "Estonia",
            "db_name": "estonia_db",
            "description": "Expert in Estonian Law. Specialized in digital governance, commercial codes, and Estonian specific legal frameworks."
        },
        "Slovenia_Agent": {
            "country": "Slovenia",
            "db_name": "slovenia_db",
            "description": "Expert in Slovenian Law. Covers constitutional, civil rights, and corporate regulations within the Republic of Slovenia."
        }
    }

    @staticmethod
    def execute(
        query: str, 
        config: "AppConfig", 
        chat_history: List[BaseMessage] = None,
        return_trace: bool = False,
        use_expansion: bool = False
    ) -> Tuple[str, List[Document], Optional[str]]:
        
        logs = []
        llm = LLMFactory.create_llm(config)
        
        # --- FASE 0: CONTEXTUALIZATION ---
        final_query = query
        is_contextual = False
        
        if chat_history:
            rewritten_query, is_dependent, reason = SupervisorNetwork._contextualize_query(query, chat_history, llm)
            if is_dependent:
                final_query = rewritten_query
                is_contextual = True
                logs.append(f"🔄 **Contextualizer**: Query dependent. Rewritten to: '{final_query}'")
            else:
                # Se è standalone, ignoriamo la history per il routing e la generazione
                chat_history = [] 
                logs.append(f"🆕 **Contextualizer**: Query is standalone. History dropped.")
        else:
            logs.append("ℹ️ **Contextualizer**: No history present.")

        logs.append(f"**Supervisor**: Processing query: '{final_query}'")
        requires_retrieval = SupervisorNetwork._classify_intent(final_query, llm)
        
        if not requires_retrieval:
            logs.append("**Decision**: Query is generic. No retrieval needed.")
            answer = llm.invoke(f"Answer this generic question based on your internal knowledge: {final_query}").content
            return answer, [], "\n\n".join(logs) if return_trace else None

        logs.append("**Decision**: Legal intent detected. Proceeding to Routing.")
        
        # 1. Expansion (Optional)
        search_query = final_query
        if use_expansion:
            try:
                search_query = SupervisorNetwork._expand_query(final_query, llm)
                logs.append(f"**Supervisor**: Query expanded: '{search_query}'")
            except Exception:
                pass

        # 2. Routing (Based on Contextualized Query)
        selected_agent_keys, reason = SupervisorNetwork._route_query(final_query, config)
        logs.append(f"**Supervisor Routing**: Activating -> {selected_agent_keys}. Reason: {reason}")
        
        all_retrieved_docs: List[Document] = []
        partial_answers: List[str] = []
        
        # 3. Parallel Execution (Simulation)
        for agent_key in selected_agent_keys:
            if agent_key not in SupervisorNetwork.AGENT_REGISTRY:
                continue
                
            agent_info = SupervisorNetwork.AGENT_REGISTRY[agent_key]
            logs.append(f"**Agent Activation**: Starting {agent_key} ({agent_info['country']})...")
            
            agent_response, agent_docs, agent_log = SupervisorNetwork._activate_specialized_agent(
                agent_info=agent_info,
                query=final_query,       # Use contextualized query
                search_query=search_query, 
                config=config
            )
            
            if agent_docs:
                all_retrieved_docs.extend(agent_docs)
                partial_answers.append(f"--- REPORT FROM {agent_key} ---\n{agent_response}")
                logs.append(f"**{agent_key}**: Success. Found {len(agent_docs)} docs. {agent_log}")
            else:
                logs.append(f"**{agent_key}**: No relevant documents found.")

        # 4. Aggregation
        if not partial_answers:
            msg = "I consulted the specialized agents, but none found relevant legal documents to answer your specific query."
            logs.append(f"**Final**: {msg}")
            return msg, [], "\n\n".join(logs) if return_trace else None

        logs.append("**Supervisor**: Aggregating partial answers...")
        
        # Nel prompt finale passiamo la history solo se serve
        final_history_str = ""
        if is_contextual and chat_history:
            final_history_str = "CHAT HISTORY:\n" + "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history]) + "\n\n"
            
        final_answer = SupervisorNetwork._aggregate_partial_answers(
            query=final_query, 
            partial_reports=partial_answers, 
            history_context=final_history_str,
            config=config
        )
        
        trace = "\n\n".join(logs) if return_trace else None
        return final_answer, all_retrieved_docs, trace
    
    @staticmethod
    def _activate_specialized_agent(
        agent_info: Dict, 
        query: str, 
        search_query: str,
        config: "AppConfig"
    ) -> Tuple[str, List[Document], str]:
        """
        Specialized Agent Logic: Pure Semantic Search + Reranking.
        """
        embedding_model = EmbeddingFactory.create_embedding_model(config)
        k_val = int(config.retrieval_top_k)
        path = os.path.join(config.vector_db_root_path, agent_info['db_name'])
        trace_notes = ""
        
        if not os.path.exists(path):
            return "Database not available.", [], "DB missing"

        try:
            vectorstore = VectorArchivist.load_index(path, embedding_model)
            
            # --- CONFIGURAZIONE FETCH ---
            is_reranking = getattr(config, 'enable_reranking', False)
            # Se reranking attivo, fetch ampio (es. 25). Se no, fetch esatto (k_val).
            fetch_k = getattr(config, 'rerank_initial_fetch', 30) if is_reranking else k_val

            # --- SINGLE SEMANTIC SEARCH (No Regex Article Boosting) ---
            # Usiamo search_query (espansa) per il fetch
            # Usiamo score per eventuale debug o logica futura
            candidates_with_score = vectorstore.similarity_search_with_score(search_query, k=fetch_k)
            
            # Elaborazione Candidati
            candidates = []
            for doc, score in candidates_with_score:
                # FIX SERIALIZZAZIONE JSON
                doc.metadata["vector_score"] = float(score)
                candidates.append(doc)

            final_docs = []

            # --- RERANKING LAYER ---
            if is_reranking and candidates:
                trace_notes += "[Reranker Active]"
                try:
                    # Il Reranker usa la query originale (intento utente), non quella espansa
                    reranked = SemanticReranker.rerank_documents(
                        query=query, 
                        docs=candidates, 
                        top_k=k_val # Taglio finale
                    )
                    final_docs = reranked
                except Exception as e:
                    trace_notes += f" [Reranker Failed: {e}]"
                    final_docs = candidates[:k_val]
            else:
                # Standard cut
                final_docs = candidates[:k_val]

            if not final_docs:
                return "No relevant documents found.", [], "No docs"

            # --- LOCAL GENERATION ---
            context_text = "\n\n".join([d.page_content for d in final_docs])
            llm = LLMFactory.create_llm(config)
            
            prompt = (
                f"You are the {agent_info['country']} Legal Specialist.\n"
                "Answer the user query based ONLY on the provided context excerpts.\n"
                "Do not mention other countries. Focus purely on local law.\n"
                "Cite sources if possible.\n\n"
                f"CONTEXT ({agent_info['country']}):\n{context_text}\n\n"
                f"QUERY: {query}\n\n"
                "YOUR SPECIALIZED ANSWER:"
            )
            
            partial_answer = llm.invoke(prompt).content
            
            # Tagging finale per la UI
            for d in final_docs:
                d.metadata['source_agent'] = agent_info['country']
                # Normalizza country per la UI (alcuni DB hanno 'source_country', altri no)
                d.metadata['source_country'] = agent_info['country']
                
            return partial_answer, final_docs, trace_notes

        except Exception as e:
            return f"Error processing request: {e}", [], str(e)

    # --- HELPERS ---

    @staticmethod
    def _contextualize_query(query: str, history: List[BaseMessage], llm: Any) -> Tuple[str, bool, str]:
        if not history:
            return query, False, "No history"
        
        history_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in history[-2:]])
        prompt = (
            "Analyze if the NEW USER QUERY depends on the CHAT HISTORY.\n"
            "- If YES (e.g. 'and in Estonia?'): Rewrite it to be standalone.\n"
            "- If NO: Return the query as is.\n\n"
            f"HISTORY:\n{history_text}\n\n"
            f"QUERY: {query}\n\n"
            "Output JSON: {'is_dependent': bool, 'rewritten_query': str, 'reason': str}"
        )
        try:
            res = llm.invoke(prompt).content.strip()
            if "```" in res: res = res.split("```json")[-1].split("```")[0] if "json" in res else res.split("```")[1]
            data = json.loads(res)
            return data["rewritten_query"], data["is_dependent"], data["reason"]
        except:
            return query, False, "Error parsing"

    @staticmethod
    def _classify_intent(query: str, llm: Any) -> bool:
        return "YES" in llm.invoke(f"Is this legal query? (YES/NO): {query}").content.upper()

    @staticmethod
    def _route_query(query: str, config: "AppConfig") -> Tuple[List[str], str]:
        llm = LLMFactory.create_llm(config)
        agents_desc_str = json.dumps(SupervisorNetwork.AGENT_REGISTRY, indent=2)
        prompt = (
            "Act as a Supervisor Router.\n"
            f"AGENTS:\n{agents_desc_str}\n\n"
            "Decide which agents to activate based on the query jurisdiction.\n"
            "If generic/comparative, activate ALL.\n"
            f"QUERY: {query}\n"
            "Output JSON: {\"agents\": [\"Italy_Agent\", ...], \"reason\": \"...\"}"
        )
        try:
            res = llm.invoke(prompt).content
            if "```" in res: res = res.split("```json")[-1].split("```")[0]
            data = json.loads(res)
            agents = data.get("agents", [])
            # Fallback se la lista è vuota
            if not agents: return list(SupervisorNetwork.AGENT_REGISTRY.keys()), "Fallback All"
            return agents, data.get("reason", "Routing")
        except:
            return list(SupervisorNetwork.AGENT_REGISTRY.keys()), "Error Routing"

    @staticmethod
    def _aggregate_partial_answers(query: str, partial_reports: List[str], history_context: str, config: "AppConfig") -> str:
        llm = LLMFactory.create_llm(config)
        combined_reports = "\n\n".join(partial_reports)
        
        # Costruiamo un prompt più strutturato con "Style Guidelines"
        prompt = (
            "You are a Senior Legal Supervisor synthesizing expert reports.\n\n"
            
            "### STYLE GUIDELINES:\n"
            "1. **Direct Start**: Start the answer IMMEDIATELY. Do NOT use prefixes like 'Final Response:', 'Conclusion:', 'Based on the reports', or 'Here is the summary'.\n"
            "2. **Professional Tone**: Write as a direct answer to the user, not as a summary of what the agents said (avoid 'The Italian agent states...').\n\n"
            
            "### CONTENT LOGIC:\n"
            "1. **Single Jurisdiction Intent**: If the User Query targets a specific country (e.g., 'Inheritance in Italy'), focus ONLY on that country. Do NOT mention differences with other countries unless explicitly asked.\n"
            "2. **Comparative Intent**: If the User Query asks for a comparison or is generic (e.g., 'Divorce in Europe', 'Compare Italy and Estonia'), you MUST explicitly highlight the differences and similarities between the jurisdictions found in the reports.\n\n"
            
            f"{history_context}"
            f"USER QUERY: {query}\n\n"
            f"AGENT REPORTS:\n{combined_reports}\n\n"
            "ANSWER:" # Lasciamo solo ANSWER: come trigger finale, l'LLM completerà il resto
        )
        
        # Pulizia finale di sicurezza (nel caso l'LLM disubbidisse e mettesse ancora il prefisso)
        response = llm.invoke(prompt).content.strip()
        
        return response

    @staticmethod
    def _expand_query(original_query: str, llm_engine: Any) -> str:
        return llm_engine.invoke(f"Refine for vector search: {original_query}").content.strip()