"""
pages/2_Evaluation_Dashboard.py

Performance Analytics Module.
Implements the RAGAS framework to audit the quality of the Legal RAG System.
Features:
- Metric calculation (Faithfulness, Precision, Recall, etc.).
- Smart Selection Logic (Auto-select all if none checked).
- Visual reporting of Global System Performance.
"""

import streamlit as st
import pandas as pd
import altair as alt
from datasets import Dataset

# RAGAS Metrics & Config
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)

# Internal Modules
from dotenv import load_dotenv
import os

# Carica subito le variabili d'ambiente
load_dotenv()

from src.core.session_logger import SessionRecorder
from src.core.data_ingestion import EmbeddingFactory
from src.core.llm_factory import LLMFactory
from src.core.config_manager import AppConfig

st.set_page_config(page_title="LexAI RAGAS Evaluation", page_icon="📈", layout="wide")

class RagasEvaluatorUI:
    """
    Manages the Evaluation Dashboard UI and RAGAS execution logic.
    """
    
    @staticmethod
    def load_metrics_definitions():
        with st.expander("ℹ️ Metrics Definitions"):
            st.markdown("""
            - **Context Precision** (0-1): Ratio of relevant chunks in the retrieved context. (Noise filter)
            - **Context Recall** (0-1): Is the answer to the ground truth present in the retrieved context? (Retrieval power)
            - **Faithfulness** (0-1): Is the generated answer derived purely from context? (Hallucination check)
            - **Answer Relevancy** (0-1): How pertinent is the answer to the user's query?
            - **Answer Correctness** (0-1): Semantic similarity to the 'Golden Answer'.
            """)

    @staticmethod
    def run_evaluation(dataset: Dataset, config: AppConfig):
        """
        Executes the RAGAS pipeline.
        Auto-selects GPT-4o (Parallel) if available, otherwise fallbacks to Home LLM (Sequential).
        """
        
        # 1. SETUP DEL GIUDICE (LLM)
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if openai_key:
            # 🟢 CASO OTTIMALE: OpenAI GPT-4o
            try:
                from langchain_openai import ChatOpenAI
                # Usiamo GPT-4o per una valutazione rigorosa
                eval_llm = ChatOpenAI(model="gpt-4o", api_key=openai_key)
                
                workers = 1 
                timeout = 1000
                st.toast("Using OpenAI GPT-4o as Judge ⚖️", icon="✅")
                
            except Exception as e:
                st.error(f"Error init OpenAI: {e}")
                return None
        else:
            # 🔴 FALLBACK: Groq (o altro modello locale)
            st.warning("⚠️ OpenAI Key not found. Using the Chat Model as Judge. Complex metrics might fail or return NaN.")
            eval_llm = LLMFactory.create_llm(config)
            
            # Groq ha limiti severi: usiamo 1 worker sequenziale
            workers = 1
            timeout = 180

        # Embeddings (Sempre quelli locali HuggingFace per coerenza)
        try:
            eval_embeddings = EmbeddingFactory.create_embedding_model(config)
        except Exception as e:
            st.error(f"Embedding Init Failed: {e}")
            return None

        # 2. CONFIGURAZIONE RUN
        my_run_config = RunConfig(max_workers=workers, timeout=timeout, max_retries=2)

        with st.spinner(f"🔍 Auditing System Performance (Workers: {workers})..."):
            try:
                results = evaluate(
                    dataset=dataset,
                    metrics=[
                        context_precision,
                        context_recall,
                        faithfulness,
                        answer_relevancy,
                        answer_correctness
                    ],
                    llm=eval_llm,
                    embeddings=eval_embeddings,
                    run_config=my_run_config
                )
            except Exception as e:
                st.error(f"RAGAS Execution Error: {e}")
                st.info("Tip: If using Groq, 'Answer Relevancy' often fails due to rate limits.")
                return None
                
        return results

def main():
    st.title("📈 System Evaluation Dashboard")
    st.caption("Quantitative Assessment using RAGAS Framework")

    RagasEvaluatorUI.load_metrics_definitions()
    
    # 1. Load Data
    raw_logs = SessionRecorder.load_sessions()
    
    tab1, tab2 = st.tabs(["📝 Session Review", "🔎 Upload Test Set"])
    
    with tab1:
        if not raw_logs:
            st.warning("No session logs found. Please use the Chat Interface first to generate data.")
            st.stop()
            
        st.write(f"Loaded **{len(raw_logs)}** interaction logs.")
        
        # Preparazione Dataframe con colonna "Include" (Checkbox)
        simple_df = pd.DataFrame({
            "Include": [False] * len(raw_logs), # Default: nulla selezionato
            "question": [l["user_query"] for l in raw_logs],
            "answer": [l["system_response"] for l in raw_logs],
            "contexts": [[c["page_content"][:50] + "..." for c in l["retrieved_contexts"]] for l in raw_logs],
            "ground_truth": [""] * len(raw_logs) 
        })
        
        st.markdown("### 1. Select & Annotate")
        st.info("💡 **Tip:** If you don't check any boxes, the system will evaluate **ALL** queries that have a Golden Answer.")
        
        edited_df = st.data_editor(
            simple_df,
            column_config={
                "Include": st.column_config.CheckboxColumn("Select", width="small"),
                "ground_truth": st.column_config.TextColumn("Golden Answer (Required)", width="large"),
                "contexts": st.column_config.ListColumn("Context Preview"),
                "question": st.column_config.TextColumn("User Query", disabled=True),
                "answer": st.column_config.TextColumn("System Answer", disabled=True)
            },
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True
        )
        
        # 2. Run Evaluation Button
        st.markdown("### 2. Run Audit")
        
        if st.button("🚀 Calculate RAGAS Metrics", type="primary"):
            
            # --- LOGICA SMART SELECTION ---
            # 1. Controlliamo cosa ha spuntato l'utente
            selected_rows = edited_df[edited_df["Include"] == True]
            
            # 2. Se non ha spuntato nulla, prendiamo tutto il dataframe
            if selected_rows.empty:
                st.info("ℹ️ No specific queries checked. Switching to **Batch Mode** (Evaluating ALL valid entries).")
                indices_to_process = edited_df.index.tolist()
                is_batch_mode = True
            else:
                st.success(f"✅ Evaluating **{len(selected_rows)}** manually selected queries.")
                indices_to_process = selected_rows.index.tolist()
                is_batch_mode = False

            # --- COSTRUZIONE DATASET ---
            final_data = {
                "question": [], "answer": [], "contexts": [], "ground_truth": []
            }
            
            valid_count = 0
            
            for idx in indices_to_process:
                # Recuperiamo la Ground Truth inserita dall'utente
                gt = edited_df.loc[idx, "ground_truth"]
                
                # VALIDAZIONE: La Ground Truth è obbligatoria
                if not gt or str(gt).strip() == "":
                    # Se l'utente l'aveva selezionata manualmente, lo avvisiamo
                    if not is_batch_mode:
                        st.warning(f"⚠️ Query #{idx} skipped: Missing Golden Answer.")
                    continue
                
                # Aggiungiamo i dati ai vettori finali
                final_data["question"].append(raw_logs[idx]["user_query"])
                final_data["answer"].append(raw_logs[idx]["system_response"])
                # Importante: Recuperiamo i contesti COMPLETI dal log originale, non quelli tagliati della tabella
                full_ctx = [c["page_content"] for c in raw_logs[idx]["retrieved_contexts"]]
                final_data["contexts"].append(full_ctx)
                final_data["ground_truth"].append(gt)
                valid_count += 1
            
            # --- CONTROLLO FINALE ---
            if valid_count == 0:
                st.error("❌ No valid data found! Please provide a 'Golden Answer' for at least one query.")
                st.stop()
            
            if is_batch_mode:
                st.write(f"Processing **{valid_count}** valid queries...")

            # Conversione in Dataset HuggingFace
            eval_dataset = Dataset.from_dict(final_data)
            
            # Init Config
            if "app_config" not in st.session_state:
                st.session_state.app_config = AppConfig()
                
            # ESECUZIONE REALE
            results = RagasEvaluatorUI.run_evaluation(eval_dataset, st.session_state.app_config)
            
            if results:
                st.balloons()
                st.success("Evaluation Complete!")
                
                # --- VISUALIZZAZIONE ---
                res_df = results.to_pandas()
                
                # A. Score Cards (Media Globale)
                st.markdown("#### 📊 Global Performance Scores (Average)")
                cols = st.columns(5)
                metrics_list = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
                
                global_scores = {}
                
                for i, metric in enumerate(metrics_list):
                    if metric in res_df.columns:
                        avg_val = res_df[metric].mean()
                        # Salviamo per il grafico (sostituendo NaN con 0 per visualizzazione)
                        global_scores[metric] = avg_val if pd.notna(avg_val) else 0.0
                        
                        # Display nella card (N/A se fallito)
                        disp_val = f"{avg_val:.3f}" if pd.notna(avg_val) else "N/A"
                        cols[i].metric(label=metric.replace("_", " ").title(), value=disp_val)
                
                # B. Global Chart (Altair - 5 Barre Affiancate)
                st.markdown("#### 🏆 Global Metric Comparison")
                
                if global_scores:
                    # Prepariamo il DF per Altair: Metric | Score
                    chart_df = pd.DataFrame(list(global_scores.items()), columns=['Metric', 'Score'])
                    # Clean Labels
                    chart_df['Metric'] = chart_df['Metric'].str.replace('_', ' ').str.title()
                    
                    chart = alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X('Metric:N', title=None, axis=alt.Axis(labelAngle=0)), # Nomi in orizzontale
                        y=alt.Y('Score:Q', title='Average Score (0-1)', scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color('Metric:N', legend=None), # Colore automatico per categoria
                        tooltip=['Metric', alt.Tooltip('Score', format='.3f')]
                    ).properties(
                        height=350,
                        title="Average Performance across all queries"
                    ).configure_axis(
                        labelFontSize=12,
                        titleFontSize=14
                    )
                    
                    st.altair_chart(chart, use_container_width=True)

                # C. Tabella Dettagliata (Per debug/analisi specifica)
                with st.expander("🔍 View Granular Data (Analysis per Query)"):
                    st.dataframe(res_df, use_container_width=True)

    with tab2:
        st.info("Upload a JSON file containing the test queries.")
        uploaded_file = st.file_uploader("Upload test_set.json", type="json")
        
        if uploaded_file:
            st.json(pd.read_json(uploaded_file).head())
            st.button("Run Batch Evaluation on Test Set")

if __name__ == "__main__":
    main()