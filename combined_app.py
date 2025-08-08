# combined_app.py
"""
Single-file Streamlit app that includes the backend engine (from main.py) inline.
Keeps main.py and app.py untouched; this file can be deployed as single entrypoint on Streamlit Cloud.

Usage:
    streamlit run combined_app.py
"""

import streamlit as st
import time
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

# --- Try to import the real engine and Pydantic models from main.py ---
# main.py contains EnhancedQueryProcessingEngine, Config, and Pydantic models.
# If import fails or GROQ key not set, we fallback to a mock engine.
try:
    import main as core   # your main.py file (kept untouched)
    HAS_MAIN = True
except Exception as e:
    core = None
    HAS_MAIN = False

# --- Fallback Mock classes (used when real engine can't initialize) ---
class MockAnswer:
    def __init__(self, question: str):
        self.question = question
        self.answer = f"(mock) I couldn't access the LLM. Echo: {question}"
        self.confidence = 0.42
        self.source_clauses = []
        self.justification = "Mock justification - no LLM available"
        self.conflicts_detected = []

class MockEngine:
    """Simple async mock engine that imitates EnhancedQueryProcessingEngine.process_query"""
    async def process_query(self, request):
        # Build a simple mocked response compatible with app UI
        answers = []
        for q in request.questions:
            ans = MockAnswer(q)
            answers.append({
                "question": ans.question,
                "answer": ans.answer,
                "confidence": ans.confidence,
                "source_clauses": ans.source_clauses,
                "justification": ans.justification,
                "conflicts_detected": ans.conflicts_detected
            })
        return {
            "answers": answers,
            "processing_time": 0.01,
            "audit_trail": ["mock: engine not available, returned stub results"]
        }

# --- Initialize engine: try to use the real one, otherwise fallback ---
ENGINE = None
ENGINE_INIT_ERROR = None

def init_engine_once():
    global ENGINE, ENGINE_INIT_ERROR
    if ENGINE is not None or ENGINE_INIT_ERROR is not None:
        return

    if HAS_MAIN:
        try:
            # If Config.GROQ_API_KEY looks unset or placeholder, avoid initializing Groq client.
            if hasattr(core, "Config") and getattr(core.Config, "GROQ_API_KEY", "").startswith("your_"):
                ENGINE_INIT_ERROR = "GROQ_API_KEY not set in environment - falling back to MockEngine"
                ENGINE = MockEngine()
            else:
                # instantiate the real engine (this will try to initialize GroqLLM etc.)
                ENGINE = core.EnhancedQueryProcessingEngine()
        except Exception as e:
            ENGINE_INIT_ERROR = f"Failed to init EnhancedQueryProcessingEngine: {e}. Falling back to MockEngine"
            ENGINE = MockEngine()
    else:
        ENGINE_INIT_ERROR = "main.py import failed - using MockEngine"
        ENGINE = MockEngine()

# --- Small reusable helpers for UI ---
def run_async(coro):
    """Run an async coroutine from sync context (safe for Streamlit)"""
    return asyncio.get_event_loop().run_until_complete(coro) if asyncio.get_event_loop().is_running() else asyncio.run(coro)

def call_engine(document_url: str, questions: List[str]) -> Dict[str, Any]:
    """Wrapper that creates the request structure and calls engine.process_query"""
    init_engine_once()
    if ENGINE is None:
        raise RuntimeError("Engine not initialized")

    # If main.QueryRequest model exists, use it to validate; otherwise make a simple object
    if HAS_MAIN and hasattr(core, "QueryRequest"):
        request = core.QueryRequest(documents=document_url, questions=questions)
    else:
        # minimal fallback
        class R: pass
        request = R()
        request.documents = document_url
        request.questions = questions

    # engine.process_query is async
    try:
        result = run_async(ENGINE.process_query(request))
        # If result is a Pydantic object, convert to dict. If not, assume dict-like.
        if hasattr(result, "dict"):
            return result.dict()
        elif isinstance(result, dict):
            return result
        else:
            # try to convert attributes
            return json.loads(json.dumps(result, default=lambda o: o.__dict__))
    except Exception as e:
        return {"answers": [], "processing_time": 0.0, "audit_trail": [f"Engine call failed: {e}"], "error": str(e)}

# --- Streamlit UI (compact, but aligned to app.py flow) ---
st.set_page_config(page_title="Combined AI Document Query", layout="wide")
st.title("🧠 Combined — UI + Backend (single file)")

init_engine_once()
if ENGINE_INIT_ERROR:
    st.warning(f"Engine notice: {ENGINE_INIT_ERROR}")

with st.sidebar:
    st.header("Settings")
    st.write("This combined app calls the local engine directly (no HTTP).")
    st.markdown("---")
    if HAS_MAIN:
        st.write(f"Using main.py present in repo (import OK).")
        if hasattr(core, "Config"):
            st.write(f"Model: {core.Config.LLM_MODEL_NAME}")
            st.write(f"Embedding: {core.Config.EMBEDDING_MODEL}")
    else:
        st.write("main.py not importable — using MockEngine.")

st.header("📄 Document Analysis (combined)")

col1, col2 = st.columns([3,1])
with col1:
    document_url = st.text_input("Document URL (public)", placeholder="https://example.com/doc.pdf")
with col2:
    sample = st.button("Use sample PDF")
    if sample:
        document_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

st.subheader("❓ Questions")
if "questions" not in st.session_state:
    st.session_state.questions = [""]

# dynamic questions UI
for i in range(len(st.session_state.questions)):
    st.session_state.questions[i] = st.text_input(f"Question {i+1}", value=st.session_state.questions[i], key=f"q{i}")

add_col, rem_col = st.columns(2)
with add_col:
    if st.button("➕ Add Question"):
        st.session_state.questions.append("")
        st.experimental_rerun()
with rem_col:
    if len(st.session_state.questions) > 1 and st.button("➖ Remove Last"):
        st.session_state.questions.pop()
        st.experimental_rerun()

questions = [q.strip() for q in st.session_state.questions if q.strip()]

st.markdown("---")
analyze_btn = st.button("🚀 Analyze (local engine)")

if analyze_btn:
    if not document_url:
        st.error("Provide a document URL.")
    elif not questions:
        st.error("Add at least one question.")
    else:
        with st.spinner("Processing..."):
            start = datetime.now()
            response = call_engine(document_url, questions)
            duration = (datetime.now() - start).total_seconds()

        # Display processing summary
        st.success(f"Completed in {duration:.2f}s")
        if response.get("error"):
            st.error(f"Engine error: {response['error']}")

        # Show metrics
        st.metric("Processing Time (engine)", f"{response.get('processing_time', duration):.2f}s")
        answers = response.get("answers", [])
        st.metric("Questions Processed", len(answers))

        # Answers as expandable cards
        for idx, ans in enumerate(answers):
            conf = ans.get("confidence", 0.0)
            label = f"Q{idx+1}: {ans.get('question','')[:80]}"
            with st.expander(label, expanded=True):
                st.write("**Answer:**")
                st.write(ans.get("answer", ""))
                st.write("**Confidence:**", f"{conf:.2f}")
                st.write("**Source Clauses:**", ans.get("source_clauses", []))
                st.write("**Justification:**")
                st.code(ans.get("justification", ""))
                if ans.get("conflicts_detected"):
                    st.warning("Conflicts detected: " + ", ".join(ans.get("conflicts_detected", [])))

        # Audit trail
        at = response.get("audit_trail", [])
        if at:
            with st.expander("Audit trail"):
                for step in at:
                    st.write("-", step)

        # Download JSON
        st.download_button("Download Results (JSON)", json.dumps(response, indent=2), file_name=f"results_{int(time.time())}.json", mime="application/json")

# small footer
st.markdown("---")
st.caption("This combined app calls the engine in-process instead of via HTTP. Keep main.py and app.py unchanged.")
