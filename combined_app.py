# combined_app.py
"""
Single-file Streamlit app — improved version with lazy import of main.py.
Place this file in the same repo as main.py and app.py.
Run: streamlit run combined_app.py
"""

import streamlit as st
import asyncio
import importlib.util
import sys
import traceback
import json
import time
from datetime import datetime
from typing import List, Dict, Any

st.set_page_config(page_title="Combined AI Document Query (lazy import)", layout="wide")
st.title("🧠 Combined — UI + Backend (lazy import)")

# ------------------------
# Mock fallback engine
# ------------------------
class MockAnswer:
    def __init__(self, question: str):
        self.question = question
        self.answer = f"(mock) No backend available. Echo: {question}"
        self.confidence = 0.3
        self.source_clauses = []
        self.justification = "This is a mock justification because the real engine could not be loaded."

class MockEngine:
    async def process_query(self, request):
        answers = []
        qs = getattr(request, "questions", []) or list(getattr(request, "questions", []))
        for q in qs:
            a = MockAnswer(q)
            answers.append({
                "question": a.question,
                "answer": a.answer,
                "confidence": a.confidence,
                "source_clauses": a.source_clauses,
                "justification": a.justification,
                "conflicts_detected": []
            })
        return {"answers": answers, "processing_time": 0.0, "audit_trail": ["mock fallback used"]}

# ------------------------
# Utilities
# ------------------------
def run_async(coro):
    """Run an async coroutine from sync context (Streamlit)."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # If running (rare in Streamlit), create task and wait
        return asyncio.run(coro)
    else:
        return asyncio.run(coro)

def dynamic_import_main(main_path: str = "main.py"):
    """
    Try to import user's main.py dynamically.
    Returns (module_or_None, error_message_or_None, missing_modules_list)
    """
    try:
        spec = importlib.util.spec_from_file_location("user_main_module", main_path)
        module = importlib.util.module_from_spec(spec)
        # allow relative imports inside user's main.py to work
        sys.modules["user_main_module"] = module
        spec.loader.exec_module(module)
        return module, None, []
    except Exception as e:
        tb = traceback.format_exc()
        # try to extract missing module names from traceback
        missing = []
        for line in tb.splitlines():
            if "ModuleNotFoundError: No module named" in line:
                # line example: ModuleNotFoundError: No module named 'langchain_community'
                parts = line.split("No module named")
                if len(parts) > 1:
                    name = parts[1].strip().strip(":").strip().strip("'\"")
                    missing.append(name)
        return None, tb, missing

# ------------------------
# Engine manager (lazily initialized)
# ------------------------
ENGINE = None
ENGINE_INIT_NOTICE = None
MAIN_MODULE = None

def init_engine_if_needed(main_path="main.py"):
    """
    Called when user presses Analyze. Attempts to import main.py and instantiate the engine.
    Returns (engine_instance, notice_message)
    """
    global ENGINE, ENGINE_INIT_NOTICE, MAIN_MODULE

    if ENGINE is not None:
        return ENGINE, ENGINE_INIT_NOTICE

    MAIN_MODULE, err_tb, missing = dynamic_import_main(main_path)
    if MAIN_MODULE:
        # Inspect the module for expected classes
        if hasattr(MAIN_MODULE, "EnhancedQueryProcessingEngine"):
            try:
                # If Config exists and GROQ_API_KEY looks missing, warn but still try to init; engine init may fail.
                config_info = None
                if hasattr(MAIN_MODULE, "Config"):
                    cfg = getattr(MAIN_MODULE, "Config")
                    groq_key = getattr(cfg, "GROQ_API_KEY", None)
                    config_info = {"GROQ_API_KEY": groq_key, "LLM_MODEL_NAME": getattr(cfg, "LLM_MODEL_NAME", None), "EMBEDDING_MODEL": getattr(cfg, "EMBEDDING_MODEL", None)}
                    if isinstance(groq_key, str) and (groq_key.strip() == "" or groq_key.lower().startswith("your_") or "changeme" in groq_key.lower()):
                        ENGINE_INIT_NOTICE = "Config.GROQ_API_KEY looks unset or placeholder — Groq LLM may not initialize without proper secret set."
                # instantiate engine (this may raise if heavy deps missing)
                try:
                    ENGINE = MAIN_MODULE.EnhancedQueryProcessingEngine()
                    if config_info:
                        ENGINE_INIT_NOTICE = (ENGINE_INIT_NOTICE or "") + f" Imported main.py and constructed EnhancedQueryProcessingEngine. Config preview: {config_info}"
                    else:
                        ENGINE_INIT_NOTICE = "Imported main.py and constructed EnhancedQueryProcessingEngine."
                    return ENGINE, ENGINE_INIT_NOTICE
                except Exception as e:
                    tb2 = traceback.format_exc()
                    ENGINE_INIT_NOTICE = f"Failed to instantiate EnhancedQueryProcessingEngine: {e}\nSee traceback below."
                    # fall back to MockEngine but keep the traceback in the notice
                    ENGINE = MockEngine()
                    return ENGINE, ENGINE_INIT_NOTICE + "\n" + tb2
            except Exception as e:
                ENGINE = MockEngine()
                ENGINE_INIT_NOTICE = f"Error while initializing real engine: {e}"
                return ENGINE, ENGINE_INIT_NOTICE
        else:
            ENGINE = MockEngine()
            ENGINE_INIT_NOTICE = "Imported main.py but EnhancedQueryProcessingEngine not found — using MockEngine."
            return ENGINE, ENGINE_INIT_NOTICE
    else:
        # import failed
        notice = "Failed to import main.py. Using MockEngine.\n\nTraceback:\n" + (err_tb or "No traceback")
        if missing:
            notice += f"\n\nDetected missing modules: {missing}\nInstall them (e.g., pip install {' '.join(missing)}) and/or add to requirements.txt."
        ENGINE = MockEngine()
        ENGINE_INIT_NOTICE = notice
        return ENGINE, ENGINE_INIT_NOTICE

# ------------------------
# UI state for questions
# ------------------------
if "questions" not in st.session_state:
    st.session_state.questions = [""]

with st.sidebar:
    st.header("Info & Troubleshooting")
    st.markdown("This combined app will attempt to import `main.py` **only** when you click Analyze.")
    st.markdown("If import fails, the app stays up and uses a mock engine so you can test the UI.")
    st.markdown("If you want the full backend to run, ensure the required packages are in `requirements.txt` and set secrets (e.g., GROQ_API_KEY).")
    st.markdown("---")
    st.write("Quick checklist:")
    st.write("- `langchain_community`, `langchain` (if used), `groq`, `sentence-transformers`, `faiss-cpu`, `torch` may be required")
    st.write("- Set `GROQ_API_KEY` in Streamlit Secrets if using Groq")
    st.write("- Test locally: `streamlit run combined_app.py` after installing requirements")

st.header("📄 Document Analysis (lazy import)")

col1, col2 = st.columns([3,1])
with col1:
    document_url = st.text_input("Document URL (public)", placeholder="https://example.com/doc.pdf")
with col2:
    if st.button("Use sample PDF"):
        document_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

st.subheader("❓ Questions")
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
analyze = st.button("🚀 Analyze (lazy init)")

if analyze:
    if not document_url:
        st.error("Provide a public document URL.")
    elif not questions:
        st.error("Add at least one question.")
    else:
        with st.spinner("Attempting to import main.py and initialize engine..."):
            engine, notice = init_engine_if_needed("main.py")

        if notice:
            # Show a collapsible notice with details (traceback, missing modules)
            with st.expander("Engine initialization info / warnings"):
                st.text(notice)

        # Build a minimal request object compatible with main.QueryRequest if available
        if MAIN_MODULE and hasattr(MAIN_MODULE, "QueryRequest"):
            try:
                request = MAIN_MODULE.QueryRequest(documents=document_url, questions=questions)
            except Exception:
                # If Pydantic model signature is different, fallback to simple object
                class Req: pass
                request = Req()
                request.documents = document_url
                request.questions = questions
        else:
            class Req: pass
            request = Req()
            request.documents = document_url
            request.questions = questions

        # Call engine.process_query (async)
        start = datetime.now()
        try:
            result = run_async(engine.process_query(request))
        except Exception as e:
            # unexpected runtime error from engine
            tb = traceback.format_exc()
            st.error(f"Engine runtime error: {e}")
            st.code(tb)
            result = {"answers": [], "processing_time": 0.0, "audit_trail": [f"Engine runtime error: {e}"]}

        duration = (datetime.now() - start).total_seconds()
        st.success(f"Done (UI elapsed {duration:.2f}s)")

        # Normalize result to dict if necessary
        if hasattr(result, "dict"):
            result = result.dict()

        st.metric("Engine processing_time (reported)", f"{result.get('processing_time', duration):.2f}s")
        st.metric("Questions", len(result.get("answers", [])))

        for idx, ans in enumerate(result.get("answers", [])):
            label = f"Q{idx+1}: {ans.get('question','')[:80]}"
            with st.expander(label, expanded=True):
                st.write("**Answer:**")
                st.write(ans.get("answer", ""))
                st.write("**Confidence:**", ans.get("confidence", 0.0))
                st.write("**Source clauses:**")
                st.write(ans.get("source_clauses", []))
                st.write("**Justification:**")
                st.code(ans.get("justification", ""))

        at = result.get("audit_trail", [])
        if at:
            with st.expander("Audit trail"):
                for s in at:
                    st.write("-", s)

        st.download_button("Download JSON", json.dumps(result, indent=2), file_name=f"results_{int(time.time())}.json", mime="application/json")

st.markdown("---")
st.caption("This app lazy-imports main.py on demand and uses a MockEngine if import or init fails.")
