# combined_app.py
"""
Single-file Streamlit UI + optional embedded backend wrapper
- Uses EnhancedQueryProcessingEngine from main.py when possible (embedded).
- Falls back to calling the backend HTTP API (/hackrx/run) if embedding fails.
Keep main.py and app.py untouched.
"""

import streamlit as st
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import requests
import asyncio
import logging

# --- Try to import classes from your main.py to reuse your backend logic ---
# Importing main (classes only) is safe — main.py's uvicorn.run(...) is under if __name__ == '__main__'
try:
    import main as main_module  # type: ignore
    from main import EnhancedQueryProcessingEngine, QueryRequest, Config  # type: ignore
    IMPORT_MAIN_OK = True
except Exception as e:
    IMPORT_MAIN_OK = False
    main_module = None
    EnhancedQueryProcessingEngine = None
    QueryRequest = None
    Config = None
    logging.warning(f"Could not import main.py backend classes: {e}")

# -------------------------
# UI helper functions (kept compact and consistent with your app.py)
# -------------------------
st.set_page_config(page_title="🧠 AI Document Query (combined)", layout="wide")

def load_css():
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
        .main-header { font-size:2.5rem; font-weight:700;
            background: linear-gradient(90deg,#667eea 0%,#764ba2 100%);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; text-align:center; }
        .sub-header { text-align:center; color:#666; margin-bottom:1rem; }
        .answer-card { border-radius:8px; padding:1rem; margin:0.75rem 0; border:1px solid #ddd; }
        .confidence-score { float:right; font-weight:700; }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_header():
    st.markdown('<div class="main-header">🧠🔥 Combined AI Document Query</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Single-file UI + optional embedded backend (uses your main.py pipeline)</div>', unsafe_allow_html=True)

def render_confidence_badge(confidence: float):
    return f"{confidence:.0%}"

def render_answer_card(answer_item: Dict, index: int):
    confidence = answer_item.get('confidence', 0.0)
    with st.expander(f"Q{index+1}: {answer_item.get('question','')[:80]}", expanded=True):
        st.markdown(f"<div class='answer-card'><div class='confidence-score'>{render_confidence_badge(confidence)}</div>", unsafe_allow_html=True)
        st.markdown(f"**Answer:**\n\n{answer_item.get('answer','No answer')}", unsafe_allow_html=True)
        st.markdown("**Source Clauses:**")
        for s in answer_item.get('source_clauses', []):
            st.write("-", s)
        st.markdown("**Justification:**")
        st.code(answer_item.get('justification', ''), language="text")
        st.markdown("</div>", unsafe_allow_html=True)

def render_processing_metrics(response_data: Dict):
    processing_time = response_data.get('processing_time', 0)
    answers = response_data.get('answers', [])
    avg_conf = sum(a.get('confidence',0) for a in answers)/len(answers) if answers else 0
    cols = st.columns(3)
    cols[0].metric("Processing time", f"{processing_time:.2f}s")
    cols[1].metric("Questions", len(answers))
    cols[2].metric("Avg confidence", f"{avg_conf:.0%}")

# -------------------------
# Engine wrapper
# -------------------------
class EmbeddedEngineWrapper:
    """Wraps your EnhancedQueryProcessingEngine if available and handles async calls."""
    def __init__(self):
        self.engine = None
        self.ready = False
        self.init_error = None
        if IMPORT_MAIN_OK and EnhancedQueryProcessingEngine is not None:
            try:
                # instantiate engine (this will instantiate GroqLLM inside and require env vars)
                self.engine = EnhancedQueryProcessingEngine()
                self.ready = True
            except Exception as e:
                self.init_error = e
                self.ready = False
        else:
            self.init_error = RuntimeError("main.py import failed or missing classes")
            self.ready = False

    def process(self, documents_url: str, questions: List[str]):
        """Run engine.process_query (async) and return dict response similar to your API."""
        if not self.ready or not self.engine:
            raise RuntimeError(f"Embedded engine not ready: {self.init_error}")

        # Build a QueryRequest pydantic instance if available
        if QueryRequest is None:
            raise RuntimeError("QueryRequest model not available from main.py")

        req = QueryRequest(documents=documents_url, questions=questions)

        # run the async process synchronously
        try:
            result = asyncio.run(self.engine.process_query(req))
            # result is a pydantic model (QueryResponse) — convert to dict
            return result.dict()
        except Exception as e:
            # bubble up error
            raise

# -------------------------
# Main Streamlit UI
# -------------------------
def main():
    load_css()
    render_header()

    st.sidebar.header("Backend Mode")
    embed_backend = st.sidebar.checkbox("Embed backend (run pipeline inside this file)", value=True)
    backend_url = st.sidebar.text_input("Backend URL (when not embedding)", value="http://localhost:8000")
    token = st.sidebar.text_input("Bearer token for API", value=(Config.HACKRX_TOKEN if Config else ""), type="password")

    # If embedding requested, attempt to initialize
    embedded_wrapper = None
    if embed_backend:
        with st.sidebar.expander("Embedded backend status", expanded=True):
            if not IMPORT_MAIN_OK:
                st.error("Cannot import main.py — embedding disabled.")
                st.info("main.py import failed; combined_app will fall back to calling remote API.")
            else:
                st.write("Attempting to initialize EnhancedQueryProcessingEngine...")
                try:
                    # Instantiate only once per run
                    if "embedded_engine_inited" not in st.session_state:
                        st.session_state.embedded_engine_inited = False

                    if not st.session_state.embedded_engine_inited:
                        # initialize
                        embedded_wrapper = EmbeddedEngineWrapper()
                        st.session_state.embedded_wrapper_obj = embedded_wrapper
                        st.session_state.embedded_engine_inited = True
                    else:
                        embedded_wrapper = st.session_state.get("embedded_wrapper_obj", None)

                    if embedded_wrapper and embedded_wrapper.ready:
                        st.success("Embedded backend ready.")
                    else:
                        st.warning(f"Embedded backend not ready: {embedded_wrapper.init_error if embedded_wrapper else 'unknown'}")
                        st.info("If you want full functionality, set GROQ_API_KEY and ensure dependencies on Streamlit Cloud.")
                except Exception as e:
                    st.error(f"Embedded init error: {e}")
                    embedded_wrapper = None
    else:
        st.sidebar.info("UI will call remote backend at provided URL.")

    st.markdown("---")
    st.header("📄 Document Analysis (combined)")

    col1, col2 = st.columns([3, 1])
    with col1:
        document_url = st.text_input("Document URL", placeholder="https://example.com/document.pdf")
    with col2:
        if st.button("Use sample PDF"):
            document_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
            st.experimental_rerun()

    st.subheader("Questions")
    if 'questions' not in st.session_state:
        st.session_state['questions'] = [""]

    questions = []
    for i in range(len(st.session_state['questions'])):
        q = st.text_input(f"Question {i+1}", value=st.session_state['questions'][i], key=f"q{i}")
        if q and q.strip():
            questions.append(q.strip())

    cols = st.columns(2)
    if cols[0].button("➕ Add Question"):
        st.session_state['questions'].append("")
        st.experimental_rerun()
    if cols[1].button("➖ Remove Last Question") and len(st.session_state['questions']) > 1:
        st.session_state['questions'].pop()
        st.experimental_rerun()

    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Document", disabled=(not document_url or not questions))
    if analyze_btn:
        if not document_url:
            st.error("Provide a document URL")
        elif not questions:
            st.error("At least one question required")
        else:
            # show processing steps
            with st.spinner("Processing..."):
                progress = st.progress(0)
                steps = [
                    "Downloading document...",
                    "Parsing / chunking...",
                    "Building knowledge graph...",
                    "Setting up retrieval...",
                    "Calling LLM / generating answers...",
                    "Finalizing..."
                ]
                for i, s in enumerate(steps):
                    st.info(s)
                    progress.progress(int((i+1)/len(steps)*100))
                    time.sleep(0.25)

                # Decide which backend to call
                if embed_backend and IMPORT_MAIN_OK:
                    embedded_wrapper = st.session_state.get("embedded_wrapper_obj", None)
                    if embedded_wrapper and embedded_wrapper.ready:
                        try:
                            st.info("Running embedded pipeline (this can take a while)...")
                            response_data = embedded_wrapper.process(document_url, questions)
                            success = True
                        except Exception as e:
                            st.error(f"Embedded processing failed: {e}")
                            success = False
                            response_data = {"error": str(e)}
                    else:
                        st.warning("Embedded backend unavailable; falling back to API mode.")
                        embed_backend = False  # fallback to API call below

                if not embed_backend:
                    # HTTP call to backend
                    try:
                        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
                        payload = {"documents": document_url, "questions": questions}
                        st.info(f"Calling remote backend: {backend_url}/hackrx/run")
                        r = requests.post(f"{backend_url.rstrip('/')}/hackrx/run", headers=headers, json=payload, timeout=180)
                        if r.status_code == 200:
                            response_data = r.json()
                            success = True
                        else:
                            success = False
                            response_data = {"error": f"HTTP {r.status_code}: {r.text}"}
                    except Exception as e:
                        success = False
                        response_data = {"error": str(e)}

                # render results
                if success:
                    st.success("Analysis completed!")
                    render_processing_metrics(response_data)
                    st.markdown("---")
                    st.header("Results")
                    answers = response_data.get('answers', [])
                    if answers:
                        for i, ans in enumerate(answers):
                            render_answer_card(ans, i)
                    else:
                        st.info("No answers returned.")
                    st.markdown("---")
                    st.subheader("Export")
                    st.download_button("Download JSON", data=json.dumps(response_data, indent=2),
                                       file_name=f"query_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
                                       mime="application/json")
                else:
                    st.error("Processing failed")
                    st.write(response_data.get('error', 'Unknown error'))

if __name__ == "__main__":
    main()
