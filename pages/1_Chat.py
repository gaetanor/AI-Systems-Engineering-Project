"""
pages/1_Chat.py

Interactive Exam Simulation Interface.
Allows the user to query the legal corpus using either the Single-Agent or Multi-Agent architecture.
"""

import streamlit as st
import time
import numpy as np
from typing import List, Dict, Any

# Internal Modules
from src.engines.query_router import QueryOrchestrator
from src.core.session_logger import SessionRecorder
from src.core.config_manager import AppConfig
from src.core.guardrails import PIIGuardrail
from src.core.feedback_manager import FeedbackManager
from langchain_core.messages import HumanMessage, AIMessage

# Page Config
st.set_page_config(page_title="LexAI Chat", page_icon="💬", layout="wide")

# ==============================================================================
# CLASS: CHAT SESSION MANAGER (Rendering & History)
# ==============================================================================
class ChatSessionManager:
    """
    Manages the chat state and rendering logic for the interface.
    """
    
    @staticmethod
    def initialize_history():
        """Initializes the session state for messages if not present."""
        if "messages" not in st.session_state:
            st.session_state.messages = []

    @staticmethod
    def display_chat_history():
        """Renders the entire chat history from session state."""
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                if "sources" in msg and msg["sources"]:
                    source_list = msg["sources"]
                    unique_agents = set()
                    for doc in source_list:
                        meta = doc.get("metadata", {}) if isinstance(doc, dict) else doc.metadata
                        country = meta.get("source_country") or meta.get("country")
                        if country: unique_agents.add(country)
                    
                    header_text = f"📚 Referenced Legal Sources ({len(source_list)} docs from {len(unique_agents)} Agents)"
                    
                    with st.expander(header_text):
                        for doc in source_list:
                            ChatSessionManager._render_source_preview(doc)

                if "trace" in msg and msg["trace"]:
                    with st.expander("🕵️ Agent Reasoning Trace"):
                        st.code(msg["trace"], language="text")
                
                if "meta_info" in msg:
                    st.caption(msg["meta_info"])

    @staticmethod
    def _render_source_preview(doc_data: Dict[str, Any]):
        """Helper to render a single source document cleanly with Country Flags."""
        if isinstance(doc_data, dict):
            meta = doc_data.get("metadata", {})
            content = doc_data.get("page_content", "")
        else:
            meta = doc_data.metadata
            content = doc_data.page_content

        country = meta.get("source_country") or meta.get("country") or "General"
        
        flags = {
            "Italy": "🇮🇹",
            "Estonia": "🇪🇪", 
            "Slovenia": "🇸🇮",
            "General": "🇪🇺"
        }
        flag_icon = flags.get(country, "📄")
        source_name = meta.get("filename") or meta.get("source") or "Unknown Source"
        
        st.markdown(f"**{flag_icon} [{country} Agent]**") 
        st.caption(f"📜 File: `{source_name}`")
        
        preview_text = content[:350].replace("\n", " ")
        st.text(f"{preview_text}...")
        st.divider()

    @staticmethod
    def sanitize_docs_for_serialization(docs: List[Any]) -> List[Any]:
        """
        Scans documents and converts numpy types (float32) to python native types.
        Prevents 'TypeError: Object of type float32 is not JSON serializable' crashes.
        """
        if not docs: return []
        
        for doc in docs:
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                clean_meta = {}
                for k, v in doc.metadata.items():
                    if hasattr(v, "item"):
                        clean_meta[k] = v.item() # Converte in float/int standard
                    elif isinstance(v, (np.floating, np.integer)):
                         clean_meta[k] = v.item()
                    else:
                        clean_meta[k] = v
                doc.metadata = clean_meta
        return docs


# ==============================================================================
# MAIN APPLICATION LOGIC
# ==============================================================================
def main():
    st.title("💬 How can I help you today?")
    
    if "app_config" not in st.session_state:
        st.session_state.app_config = AppConfig()

    # --------------------------------------------------------------------------
    # 1. SIDEBAR CONFIGURATION
    # --------------------------------------------------------------------------
    metadata_filter = None 

    with st.sidebar:
        st.header("Architecture Control")
        
        mode_selection = st.radio(
            "Select Routing Strategy:",
            ["Single ReAct Agent", "Multi-Agent Supervisor"],
            index=0
        )
        
        if "Single" in mode_selection:
            st.session_state.app_config.system_mode = "react_agent"
            st.info("Active: **Unified index search.**")
        else:
            st.session_state.app_config.system_mode = "supervisor_agent"
            st.warning("Active: **Multi-Jurisdiction Auto-Routing.**")

        st.divider()
        st.markdown("### 🧠 Advanced Strategies")

        use_reranker = st.checkbox("Semantic Reranking", value=False, help="Apply a secondary semantic relevance model to reorder retrieved documents based on the query.")
        st.session_state.app_config.enable_reranking = use_reranker
        
        use_expansion = st.checkbox("Query Expansion", value=False, help="Expand the query with related legal concepts for broader retrieval.")

        consistency_active = st.checkbox("Consistency Check", value=False, help="After generating an answer, perform a secondary check to verify that the response is consistent with the retrieved documents. Useful for critical legal queries.")
        
        st.markdown("### 🛡️ Security Layer")
        guardrails_active = st.checkbox("PII Guardrails", value=True, help="Enable automatic detection and redaction of personally identifiable information in user queries to ensure privacy and compliance.")
            
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            #success = SessionRecorder.clear_logs()
            st.session_state.messages = []
            st.toast("History & Logs deleted successfully!", icon="✅")
            time.sleep(0.5)
            st.rerun()

    # --------------------------------------------------------------------------
    # 2. CHAT HISTORY DISPLAY
    # --------------------------------------------------------------------------
    ChatSessionManager.initialize_history()
    ChatSessionManager.display_chat_history()

    # --------------------------------------------------------------------------
    # 3. USER INPUT HANDLING
    # --------------------------------------------------------------------------
    if prompt := st.chat_input("Ask about Italian, Estonian, or Slovenian civil law..."):
        
        pii_trace_header = ""
        original_prompt_was_unsafe = False
        
        if guardrails_active:
            is_safe, sanitized, detected = PIIGuardrail.scan_and_redact(prompt)
            if not is_safe:
                prompt = sanitized 
                original_prompt_was_unsafe = True
                pii_trace_header = f"🛡️ **Guardrails Active**: Input Sanitized immediately. Redacted types: {detected}.\n\n"
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # APPEND CURRENT MESSAGE TO STATE
        st.session_state.messages.append({"role": "user", "content": prompt})

        # --- B. BACKEND PROCESSING ---
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            k_display = st.session_state.app_config.retrieval_top_k
            
            loading_msg = f"Reasoning (Top-K={k_display})"
            if use_expansion: loading_msg += " + Expanding Query 🧠"
            if use_reranker: loading_msg += " + Reranking ⚖️"
            if original_prompt_was_unsafe: loading_msg += " + PII Removed 🛡️"
            
            with st.spinner(loading_msg + "..."):
                start_time = time.time()

                past_history = st.session_state.messages[:-1]
                
                # Prendi solo gli ultimi 4 scambi per non confondere l'LLM
                recent_past_messages = past_history[-4:]

                langchain_history = []
                for msg in recent_past_messages:
                    if msg["role"] == "user":
                        langchain_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        # IMPORTANTE: Se il contenuto è None (capita con errori), metti stringa vuota
                        content = msg.get("content") or ""
                        langchain_history.append(AIMessage(content=content))
                
                # --- DEBUG to check history  ---
                with st.expander("🔍 HISTORY SENT TO BACKEND", expanded=True):
                    if not langchain_history:
                        st.error("⚠️ HISTORY IS EMPTY! (Backend will verify as NO_HISTORY)")
                    else:
                        for m in langchain_history:
                            st.write(f"**{m.type.upper()}**: {m.content}")
                # -------------------------------------

                response_text, source_docs, reasoning_log = QueryOrchestrator.dispatch_request(
                    user_query=prompt,
                    settings=st.session_state.app_config,
                    chat_history=langchain_history, # Passing only the PAST
                    metadata_filter=metadata_filter,
                    consistency_check=consistency_active,
                    use_expansion=use_expansion,
                    enable_guardrails=guardrails_active, 
                    trace_reasoning=True
                )
                
                # Sanitize for JSON serialization
                source_docs = ChatSessionManager.sanitize_docs_for_serialization(source_docs)
                
                full_reasoning_log = pii_trace_header + (reasoning_log or "")
                latency = time.time() - start_time
                
                # --- C. DISPLAY RESULT ---
                message_placeholder.markdown(response_text)
                
                status_badges = []
                status_badges.append(f"⏱️ {latency:.2f}s")
                status_badges.append(f"🤖 {st.session_state.app_config.system_mode}")
                
                if metadata_filter: status_badges.append(f"🎯 Filter: {metadata_filter['country']}")
                if use_expansion: status_badges.append("🧠 **Expanded**")
                if use_reranker: status_badges.append("⚖️ **Reranked**")
                
                if original_prompt_was_unsafe:
                    status_badges.append("🛡️ **PII BLOCKED**")
                elif guardrails_active:
                    status_badges.append("🛡️ Safe")

                if consistency_active:
                    if "Legal Consistency Warning" in response_text:
                        status_badges.append("⚠️ **Check Failed**")
                    else:
                        status_badges.append("✅ **Verified**")

                meta_info_str = " | ".join(status_badges)
                st.caption(meta_info_str)

                if source_docs:
                    unique_agents = set()
                    for d in source_docs:
                        country = d.metadata.get("source_country") or d.metadata.get("country")
                        if country: unique_agents.add(country)
                    
                    header_text = f"📚 Referenced Legal Sources ({len(source_docs)} docs from {len(unique_agents)} Agents)"
                    
                    with st.expander(header_text, expanded=False):
                        for doc in source_docs:
                            ChatSessionManager._render_source_preview({
                                "page_content": doc.page_content, 
                                "metadata": doc.metadata
                            })

                if full_reasoning_log:
                    with st.expander("🕵️ Agent Reasoning Trace", expanded=False):
                        st.code(full_reasoning_log, language="text")

        # --- D. SAVE TO SESSION STATE & LOG ---
        serializable_sources = [
            {"page_content": d.page_content, "metadata": d.metadata} 
            for d in source_docs
        ]
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "sources": serializable_sources,
            "trace": full_reasoning_log,
            "meta_info": meta_info_str 
        })
        
        SessionRecorder.log_turn(
            user_query=prompt, 
            system_response=response_text,
            retrieved_docs=source_docs,
            mode=st.session_state.app_config.system_mode,
            trace_log=full_reasoning_log
        )
        
        st.rerun()

    # --------------------------------------------------------------------------
    # 4. HUMAN-IN-THE-LOOP FEEDBACK UI
    # --------------------------------------------------------------------------
    
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_msg = st.session_state.messages[-1]
        
        last_response_text = last_msg.get("content", "")
        last_source_docs = last_msg.get("sources", [])
        
        last_user_query = "Unknown"
        if len(st.session_state.messages) >= 2:
            last_user_query = st.session_state.messages[-2]["content"]

        fb_col, _ = st.columns([1.2, 8.8])
        
        with fb_col:
            feedback_key = f"fb_len_{len(st.session_state.messages)}"
            
            with st.popover("🗳️ Feedback"):
                st.markdown("**Data Flywheel Collection**")
                
                with st.form(key=f"form_{feedback_key}"):
                    rating = st.radio("Quality:", ["👍 Positive", "👎 Negative"], horizontal=True)
                    comment = st.text_area("Correction:", placeholder="Optional comments...")
                    
                    submit_btn = st.form_submit_button("Submit")
                    
                    if submit_btn:
                        rating_val = "positive" if "👍" in rating else "negative"
                        
                        success = FeedbackManager.save_feedback(
                            user_query=last_user_query,
                            system_response=last_response_text,
                            retrieved_docs=last_source_docs,
                            rating=rating_val,
                            user_comment=comment,
                            model_name=st.session_state.app_config.model_id
                        )
                        
                        if success:
                            st.success("Saved! ✅")
                        else:
                            st.error("Error ❌")

if __name__ == "__main__":
    main()