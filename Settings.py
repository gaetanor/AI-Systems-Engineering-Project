import os

# Questo deve essere eseguito prima di qualsiasi altro import di librerie ML
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from dotenv import load_dotenv

# Import Internal Modules from our new 'src' structure
from src.core.config_manager import AppConfig
from src.core.data_ingestion import LegalContentIngestor, EmbeddingFactory
from src.engines.vector_ops import VectorArchivist

# Load environment variables explicitly at start
load_dotenv()

# 1. Page Configuration
st.set_page_config(
    page_title="LexAI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Session State Initialization
if "app_config" not in st.session_state:
    st.session_state.app_config = AppConfig()

# Helper function to trigger DB build
def run_indexing_pipeline():
    cfg = st.session_state.app_config
    
    with st.status("🏗️ Orchestrating Knowledge Base...", expanded=True) as status:
        # Step 1: Ingestion
        st.write("📂 Loading Legal JSON Corpus (Italy, Estonia, Slovenia)...")
        raw_docs = LegalContentIngestor.ingest_corpus(cfg.source_corpus_paths)
        if not raw_docs:
            status.update(label="❌ Data Ingestion Failed", state="error")
            st.error("No documents found! Check the 'data/' folder path.")
            return

        st.write(f"✅ Ingested {len(raw_docs)} documents.")
        
        # Step 2: Embedding Model
        st.write("🧠 Initializing Embedding Model...")
        embedder = EmbeddingFactory.create_embedding_model(cfg)
        
        # Step 3: Vector Store Creation (Task A + Task B Support)
        st.write("💾 Building Vector Indices...")
        
        # A. Unified Index (for Single Agent)
        VectorArchivist.create_unified_index(
            raw_docs, embedder, cfg.vector_db_root_path + "/merged_legal_index"
        )
        
        # B. Sharded Indices (for Multi-Agent Supervisor)
        shards = VectorArchivist.build_sharded_indices(
            raw_docs, embedder, cfg.vector_db_root_path
        )
        
        status.update(label="✅ Knowledge Base Ready!", state="complete")
        st.success(f"System initialized with {len(shards) + 1} vector indices.") #len(shards) + 1 perché shards contiene i 3 indici del multiagent, +1 per l'indice del single agent

# 3. Main UI Layout
st.title("⚖️ LexAI: Legal Document Analysis System")
st.markdown("### *Comparative Legal Analysis: Italy, Estonia, Slovenia*")

col1, col2 = st.columns([1, 2])

with col1:
    st.info(
        """
        **Exam Project Overview**
        
        This system implements two retrieval architectures:
        1. **Single ReAct Agent**: Direct retrieval from a unified corpus.
        2. **Multi-Agent Supervisor**: Routing across national sub-indices.
        
        **Instructions:**
        1. Verify API Key Status below.
        2. Click 'Build Knowledge Base' to process the JSON data.
        3. Navigate to **Chat Interface** to start the exam simulation.
        """
    )

with col2:
    # --- Configuration Section ---
    st.subheader("⚙️ System Configuration")
    
    # LLM Selection
    provider = st.selectbox(
        "LLM Provider", 
        ["groq", "openai", "huggingface"], 
        index=0,
        help="Select 'Groq' for free/fast inference (Llama3)."
    )
    
    # --- SELEZIONE MODELLO (Logica Specifica per HuggingFace) ---
    if provider == "huggingface":
        hf_model = st.text_input(
            "HuggingFace Model ID", 
            value="meta-llama/Llama-3.2-3B-Instruct",
            help="Repo ID from huggingface.co"
        )
        st.session_state.app_config.llm_model_name = hf_model #llm_model_name MODIFICA IL VALORE DI model_id IN config_manager.py
    elif provider == "openai":
        st.session_state.app_config.llm_model_name = "gpt-4o" #MODIFICARE QUANDO USIAMO OPENAI PER INFERENZA
    elif provider == "groq":
        st.session_state.app_config.llm_model_name = "llama-3.3-70b-versatile" #llama-3.1-8b-instant"

    st.markdown("#### 🎛️ Model Parameters")
    st.session_state.app_config.llm_temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.0, 0.1,
        help="0.0 = Deterministic, 1.0 = Creative."
    )
    
    st.session_state.app_config.retrieval_top_k = st.slider(
        "Retrieval Top-K", 1, 10, 4, 1,
        help="Number of legal documents to retrieve as context. If using Multi-Agent Supervisor, this is PER COUNTRY."
    )
    
    # --- AUTOMATIC KEY DETECTION LOGIC (AGGIORNATA) ---
    # Qui gestiamo l'eccezione per il nome della chiave HuggingFace
    if provider == "huggingface":
        env_key_name = "HUGGINGFACEHUB_API_TOKEN"
    else:
        env_key_name = f"{provider.upper()}_API_KEY"
    
    detected_key = os.getenv(env_key_name)
    
    if detected_key:
        st.success(f"✅ FOUND {env_key_name}")
        # Imposta automaticamente la chiave senza mostrarla
        os.environ[env_key_name] = detected_key
        api_key = detected_key
    else:
        st.warning(f"⚠️ {env_key_name} NOT FOUND!")
        api_key = st.text_input(
            f"Enter {provider.upper()} API Key manually:", 
            type="password"
        )
        if api_key:
            os.environ[env_key_name] = api_key
    
    # Update Config Object
    st.session_state.app_config.llm_tech_stack = provider

    # --- MODIFICA: Data Source Hardcoded e Nascosto ---
    st.session_state.app_config.source_corpus_paths = ["data/"]
    st.caption("📂 Data Corpus: Loaded from local repository")
    # --------------------------------------------------

    st.markdown("---")
    
    # --- Action Section ---
    if st.button("🚀 Build Knowledge Base", type="primary"):
        if not api_key:
             st.error(f"❌ Critical Error: Missing API Key for {provider}.")
        else:
            run_indexing_pipeline()

# Footer
st.markdown("---")
st.caption("AI Systems Engineering Exam Project | Università di Napoli Federico II | Academic Year 2025/2026")