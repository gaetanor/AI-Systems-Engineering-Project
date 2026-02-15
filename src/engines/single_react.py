"""
src/engines/single_react.py

Single Agent Engine
"""
from __future__ import annotations
import os
import json
import re
from collections import Counter
from typing import List, Tuple, Optional, Any, Dict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.core.config_manager import AppConfig
from src.core.llm_factory import LLMFactory
from src.core.data_ingestion import EmbeddingFactory
from src.engines.vector_ops import VectorArchivist
from src.core.reranking import SemanticReranker 

class SingleAgentEngine:
    
    SUPPORTED_COUNTRIES = ["Italy", "Estonia", "Slovenia"]

    @staticmethod
    def execute(
        query: str, 
        config: AppConfig, 
        chat_history: List[BaseMessage] = None, # <--- Parametro History
        return_trace: bool = False,
        use_expansion: bool = False,
        consistency_check: bool = False,
        **kwargs
    ) -> Tuple[str, List[Document], Optional[str]]:
        
        logs = []
        llm = LLMFactory.create_llm(config)
        
        # --- FASE 0: CONTEXTUALIZATION ---
        # Decidiamo se usare la memoria o buttarla via.
        final_query = query
        is_contextual = False
        
        if chat_history:
            # Chiediamo all'LLM: "Serve la storia o è una domanda nuova?"
            rewritten_query, is_dependent, reason = SingleAgentEngine._contextualize_query(query, chat_history, llm)
            
            if is_dependent:
                final_query = rewritten_query
                is_contextual = True
                logs.append(f"🔄 **Contextualizer**: Query dependent on history. Rewritten to: '{final_query}'")
            else:
                # SE NON DIPENDE, SVUOTIAMO LA STORIA! 
                # Questo impedisce al modello di "allucinare" collegamenti inesistenti.
                chat_history = [] 
                logs.append(f"🆕 **Contextualizer**: Query is standalone ('{reason}'). **Dropping History** to prevent pollution.")
        else:
            logs.append("ℹ️ **Contextualizer**: No history present. Treating as fresh query.")

        # --- FASE 1: THOUGHT ---
        # Usiamo final_query (che potrebbe essere stata riscritta)
        logs.append(f"**Thought**: Analyzing query: '{final_query}'")
        requires_retrieval = SingleAgentEngine._classify_intent(final_query, llm)
        
        if not requires_retrieval:
            logs.append("**Decision**: Query is generic. No retrieval needed.")
            # Se la history è stata svuotata, l'LLM risponderà solo basandosi sulla query attuale
            answer = llm.invoke(f"Answer this generic question based on your internal knowledge: {final_query}").content
            return answer, [], "\n\n".join(logs) if return_trace else None

        logs.append("**Decision**: Retrieval Required.")

        # --- FASE 2: PREPARATION ---
        
        # 2a. Query Expansion (Sulla query eventualmente riscritta)
        search_query = final_query
        if use_expansion:
            expanded = SingleAgentEngine._expand_query(final_query, llm)
            search_query = expanded
            logs.append(f"**Action**: Query expanded to: '{expanded}'")

        # 2b. Criteria Extraction & Deduplication
        raw_criteria, reason = SingleAgentEngine._extract_search_criteria(final_query, llm)
        
        unique_criteria = []
        seen = set()
        for c in raw_criteria:
            key = (c.get("country", "All"), c.get("law", "All"))
            if key not in seen:
                seen.add(key)
                unique_criteria.append(c)
        
        final_criteria_list = SingleAgentEngine._expand_wildcard_criteria(unique_criteria)
        logs.append(f"**Action**: Extracted Criteria -> {json.dumps(final_criteria_list, indent=2)}")

        # --- FASE 3: TOURNAMENT RETRIEVAL ---
        
        path = os.path.join(config.vector_db_root_path, "merged_legal_index")
        if not os.path.exists(path):
             return "System Error: Index not found.", [], None

        embedding_model = EmbeddingFactory.create_embedding_model(config)
        vectorstore = VectorArchivist.load_index(path, embedding_model)
        
        is_reranking = getattr(config, 'enable_reranking', False)
        target_k = int(config.retrieval_top_k)
        
        if is_reranking:
            k_per_criteria = 30 
        else:
            k_per_criteria = target_k 
            
        logs.append(f"**Retrieval Strategy**: Tournament Mode. Fetching k={k_per_criteria} per criterion.")

        all_candidates_with_score = []
        doc_ids = set()

        for criteria in final_criteria_list:
            target_country = criteria.get("country", "All")
            search_kwargs = {"k": k_per_criteria}
            if target_country != "All":
                search_kwargs["filter"] = {"country": target_country}
                
            try:
                local_query = SingleAgentEngine._sanitize_query_for_country(
                    search_query, target_country, SingleAgentEngine.SUPPORTED_COUNTRIES
                )
                
                hits = vectorstore.similarity_search_with_score(local_query, **search_kwargs)
                
                for doc, score in hits:
                    h = hash(doc.page_content)
                    if h not in doc_ids:
                        doc_ids.add(h)
                        doc.metadata["vector_score"] = float(score) 
                        all_candidates_with_score.append((doc, float(score)))
                        
            except Exception as e:
                logs.append(f"⚠️ Error fetching for {target_country}: {e}")

        # --- FASE 3.5: GLOBAL SORTING ---
        all_candidates_with_score.sort(key=lambda x: x[1])
        sorted_docs = [item[0] for item in all_candidates_with_score]
        
        if not is_reranking:
            all_retrieved_docs = sorted_docs[:target_k]
            logs.append(f"**Tournament Result**: Top {len(all_retrieved_docs)} selected globally.")
        else:
            all_retrieved_docs = sorted_docs
            logs.append(f"**Observation**: {len(all_retrieved_docs)} candidates entering Reranker.")

        if not all_retrieved_docs:
             return "No relevant documents found.", [], "\n".join(logs)

        # --- FASE 4: RERANKING ---
        if is_reranking and all_retrieved_docs:
            logs.append(f"⚖️ **Reranker**: Re-scoring top candidates...")
            try:
                reranked_docs = SemanticReranker.rerank_documents(
                    query=final_query, # Usiamo la query contestualizzata
                    docs=all_retrieved_docs, 
                    top_k=target_k
                )
                all_retrieved_docs = reranked_docs
                logs.append(f"   ✅ Optimized to top {len(all_retrieved_docs)} relevant docs.")
            except Exception as e:
                logs.append(f"   ⚠️ Reranker failed ({e}), using raw similarity results.")
                all_retrieved_docs = all_retrieved_docs[:target_k]
        
        all_retrieved_docs = all_retrieved_docs[:target_k]

        # --- FASE 5: ANSWER ---
        obs = SingleAgentEngine._generate_observation(all_retrieved_docs)
        logs.append(f"**Final Context**: {obs}")

        context_text = ""
        for i, d in enumerate(all_retrieved_docs):
            c_disp = d.metadata.get("country", "General")
            context_text += f"[Doc {i+1}] ({c_disp}): {d.page_content}\n\n"
        
        # Nel prompt finale passiamo la history SOLO SE is_contextual è True
        # Altrimenti passiamo una lista vuota per evitare bias
        final_history_str = ""
        if is_contextual and chat_history:
            final_history_str = "CHAT HISTORY:\n" + "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history]) + "\n\n"
        
        final_prompt = (
            "You are a legal assistant for European Civil Law.\n"
            "Answer using ONLY the provided context.\n"
            f"{final_history_str}"
            f"QUERY: {final_query}\n"
            f"OBSERVATION: {obs}\n"
            f"CONTEXT:\n{context_text}\n"
            "ANSWER:"
        )
        
        answer = llm.invoke(final_prompt).content
        
        if consistency_check:
            is_valid, warning = SingleAgentEngine._perform_consistency_check(answer, all_retrieved_docs)
            if not is_valid: answer += f"\n\n⚠️ {warning}"

        return answer, all_retrieved_docs, "\n\n".join(logs) if return_trace else None

    # --- HELPERS ---
    
    @staticmethod
    def _contextualize_query(query: str, history: List[BaseMessage], llm: Any) -> Tuple[str, bool, str]:
        """
        Analizza se la query dipende dalla storia.
        Returns: (Rewritten Query, Is_Dependent, Reason)
        """
        if not history:
            return query, False, "No history"
            
        # Convertiamo messaggi in stringa semplice per il prompt
        history_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in history[-2:]]) # Guardiamo solo gli ultimi 2 scambi
        
        prompt = (
            "You are a query analyzer. Determine if the NEW USER QUERY depends on the CHAT HISTORY to be understood.\n"
            "- If YES (e.g. 'and in Italy?', 'what about him?'): Rewrite the query to be standalone and complete.\n"
            "- If NO (e.g. new topic, complete sentence): Return the query exactly as is.\n\n"
            f"CHAT HISTORY:\n{history_text}\n\n"
            f"NEW USER QUERY: {query}\n\n"
            "Output Format: JSON with keys 'is_dependent' (bool), 'rewritten_query' (str), 'reason' (short str)."
        )
        
        try:
            res = llm.invoke(prompt).content.strip()
            # Pulizia JSON
            if "```" in res: res = res.split("```json")[-1].split("```")[0] if "json" in res else res.split("```")[1]
            data = json.loads(res)
            return data["rewritten_query"], data["is_dependent"], data["reason"]
        except:
            # Fallback conservativo: assumiamo sia standalone per evitare danni
            return query, False, "Error parsing context"

    @staticmethod
    def _classify_intent(query: str, llm: Any) -> bool:
        return "YES" in llm.invoke(f"Is this legal query? (YES/NO): {query}").content.upper()

    @staticmethod
    def _expand_wildcard_criteria(criteria_list: List[Dict]) -> List[Dict]:
        expanded_list = []
        for c in criteria_list:
            country = c.get("country", "All")
            law = c.get("law", "All")
            if country == "All":
                for sc in SingleAgentEngine.SUPPORTED_COUNTRIES:
                    expanded_list.append({"country": sc, "law": law})
            else:
                expanded_list.append(c)
        return expanded_list

    @staticmethod
    def _extract_search_criteria(query: str, llm: Any) -> Tuple[List[Dict[str, str]], str]:
        """
        Extracts metadata filtering criteria from the query.
        FIX: Added explicit instructions for 'Law' category extraction.
        """
        prompt = (
            "You are a Legal Search Parser. Extract metadata for vector retrieval.\n"
            "Dimensions:\n"
            "1. Country: Italy, Estonia, Slovenia (or 'All').\n"
            "2. Law: Inheritance, Divorce (or 'All' if unspecified).\n\n"  # <--- QUESTA MANCAVA
            "INSTRUCTIONS:\n"
            "1. Map adjectives (Italian -> Italy).\n"
            "2. If multiple countries, create multiple objects.\n"
            "3. Return strictly JSON.\n\n"
            "EXAMPLE:\n"
            "Query: 'Divorce laws in Italy and Estonia'\n"
            "JSON: [{\"country\": \"Italy\", \"law\": \"Divorce\"}, {\"country\": \"Estonia\", \"law\": \"Divorce\"}]\n\n"
            f"User Query: {query}\n"
            "Output JSON:"
        )
        
        try:
            txt = llm.invoke(prompt).content.strip()
            
            # Pulizia JSON (gestisce blocchi markdown ```json ... ```)
            if "```" in txt:
                txt = txt.split("```json")[-1].split("```")[0] if "json" in txt else txt.split("```")[1]
            
            data = json.loads(txt.strip())
            
            # Normalizzazione Struttura (da dict a list)
            criteria_list = []
            if isinstance(data, dict):
                criteria_list = data.get("criteria", [data])
            elif isinstance(data, list):
                criteria_list = data
            
            normalized_list = []
            
            # Normalizzazione Valori
            for item in criteria_list:
                c_raw = str(item.get("country", "All")).upper().strip()
                l_raw = str(item.get("law", "All")).upper().strip()
                
                # Mapping Country
                c_final = "All"
                if "ITAL" in c_raw: c_final = "Italy"
                elif "ESTO" in c_raw: c_final = "Estonia"
                elif "SLOV" in c_raw: c_final = "Slovenia"
                
                # Mapping Law (IMPORTANTE)
                l_final = "All"
                if "INHERIT" in l_raw or "SUCC" in l_raw: l_final = "Inheritance"
                elif "DIVOR" in l_raw or "FAMILY" in l_raw: l_final = "Divorce"
                
                normalized_list.append({"country": c_final, "law": l_final})
            
            # Fallback se la lista è vuota
            if not normalized_list:
                return [{"country": "All", "law": "All"}], "Empty List Fallback"
                
            return normalized_list, "Parsed Successfully"

        except Exception as e:
            # Fallback in caso di errore JSON
            return [{"country": "All", "law": "All"}], f"Error parsing: {str(e)}"

    @staticmethod
    def _sanitize_query_for_country(query: str, target: str, all_countries: List[str]) -> str:
        if target == "All": return query
        clean_q = query
        for country in all_countries:
            if country.lower() != target.lower():
                clean_q = re.sub(f"(?i)\\b{country}\\b", "", clean_q)
                adj_root = country[:-1] if country.endswith("a") else country
                clean_q = re.sub(f"(?i)\\b{adj_root}[a-z]*\\b", "", clean_q)
        return re.sub(r'\s+', ' ', clean_q).strip()

    @staticmethod
    def _expand_query(q: str, llm: Any) -> str:
        return llm.invoke(f"Refine for vector search: {q}").content.strip()

    @staticmethod
    def _generate_observation(docs: List[Document]) -> str:
        countries = [d.metadata.get("country", "Unknown") for d in docs]
        return f"Docs found: {len(docs)}. Distribution: {dict(Counter(countries))}"

    @staticmethod
    def _perform_consistency_check(answer: str, docs: List[Document]) -> Tuple[bool, str]:
        if docs and "[Doc" not in answer: return False, "Warning: Answer does not cite sources."
        return True, ""