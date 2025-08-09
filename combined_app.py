"""
🧠🔥 PROFESSIONAL STREAMLIT UI FOR ENHANCED DOCUMENT QUERY SYSTEM
High-quality frontend matching backend excellence

UPDATED: Adds optional local-backend orchestration (start/stop main.py)
- Keeps original UI & API behavior unchanged
- Spawns main.py as a subprocess (unmodified) only when requested
- Prefers local backend when available
"""

import os
import sys
import time
import json
import socket
import threading
import subprocess
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# ---------------------------
# New: Utilities for managing local backend (runs main.py) — added conservatively
# ---------------------------

def _is_port_open(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False

def _wait_for_http(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """Polls the given URL until it returns a successful response or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3.0)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False

class LocalBackendController:
    """
    Controller to spawn and manage a local backend process running your unmodified main.py.
    - Launch: runs `python -u main.py` in provided working directory.
    - Health: checks /health endpoint on provided host:port.
    - Stop: terminates spawned process (only if it was spawned by this controller).
    """

    def __init__(self, python_executable: str = sys.executable, main_path: str = "main.py",
                 host: str = "127.0.0.1", port: int = 8000, env: Optional[dict] = None):
        self.python_executable = python_executable
        self.main_path = main_path
        self.host = host
        self.port = port
        self.env = env or os.environ.copy()
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    def is_running(self) -> bool:
        # If we spawned it and process is alive and port open -> running
        if self._proc and self._proc.poll() is None:
            return _is_port_open(self.host, self.port)
        # Maybe main.py was started externally; check port
        return _is_port_open(self.host, self.port)

    def start(self, timeout: float = 30.0) -> Tuple[bool, str]:
        """
        Start main.py as a subprocess if it's not already running.
        Returns (success, message).
        """
        with self._lock:
            if self.is_running():
                return True, f"Backend already running at {self.base_url}"

            main_fullpath = os.path.abspath(self.main_path)
            if not os.path.exists(main_fullpath):
                return False, f"main.py not found at {main_fullpath}"

            cmd = [self.python_executable, "-u", main_fullpath]
            try:
                # Spawn process with captured stdout/stderr for diagnostics
                self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                              env=self.env, cwd=os.path.dirname(main_fullpath) or None)
            except Exception as e:
                return False, f"Failed to spawn backend process: {e}"

            # Wait for health endpoint to become available
            ok = _wait_for_http(self.health_url, timeout=timeout, interval=0.5)
            if not ok:
                stderr_preview = ""
                try:
                    if self._proc and self._proc.stderr:
                        stderr_preview = self._proc.stderr.read().decode(errors="ignore")[:2000]
                except Exception:
                    pass
                return False, f"Backend did not become healthy within {timeout}s. stderr (prefix): {stderr_preview}"
            return True, f"Backend started and healthy at {self.base_url}"

    def stop(self) -> Tuple[bool, str]:
        with self._lock:
            if self._proc:
                try:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                    self._proc = None
                    return True, "Local backend process terminated."
                except Exception as e:
                    return False, f"Error terminating process: {e}"
            else:
                # Nothing to stop that we spawned
                return False, "No spawned backend process to stop (the backend may have been started externally)."

# ---------------------------
# Original app.py content follows — preserved and augmented
# ---------------------------

# Page configuration
st.set_page_config(
    page_title="🧠 AI Document Query System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Enhanced LLM Document Query System v3.0"
    }
)

# Custom CSS for professional styling
def load_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom headers */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-connected {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    /* Answer cards - adaptive for dark and light themes */
    .answer-card {
        background: var(--background-color, #1e1e1e);
        color: var(--text-color, #ffffff) !important;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
    }

    
    .confidence-high {
        border-left: 4px solid #28a745;
    }
    
    .confidence-medium {
        border-left: 4px solid #ffc107;
    }
    
    .confidence-low {
        border-left: 4px solid #dc3545;
    }
    
    .confidence-score {
        position: absolute;
        top: 10px;
        right: 15px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    /* Progress styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg,
#ffffff 100%);
    }
    
    /* Feature boxes */
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Audit trail */
    .audit-step {
        background: #f1f3f4;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        border-left: 3px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

class DocumentQueryUI:
    def __init__(self):
        # default base url - kept as original
        self.api_base_url = "http://localhost:8000"  # Change to your deployed URL if desired
        self.hackrx_token = "90bf5fcfc3d0de340e50ac29a5cf53eb6da42e7a24af15f2111186878d510d6c"
        
    def check_system_health(self):
        """Check if the backend system is healthy"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException:
            return False, None
    
    def get_system_info(self):
        """Get system configuration information"""
        try:
            response = requests.get(f"{self.api_base_url}/system-info", timeout=5)
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException:
            return None
    
    def query_documents(self, document_url: str, questions: List[str]):
        """Send query to the backend system"""
        headers = {
            "Authorization": f"Bearer {self.hackrx_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "documents": document_url,
            "questions": questions
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/hackrx/run",
                headers=headers,
                json=payload,
                timeout=120  # 2 minutes timeout for processing
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return False, {"error": f"Request failed: {str(e)}"}

def render_header():
    """Render the main header section"""
    st.markdown('<div class="main-header">🧠🔥 AI Document Query System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enhanced hybrid retrieval with knowledge graphs • Powered by Groq API</div>', unsafe_allow_html=True)

def render_sidebar_status(ui_handler):
    """Render system status in sidebar"""
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Check system health
        is_healthy, health_data = ui_handler.check_system_health()
        
        if is_healthy:
            st.success("✅ System Online")
            
            if health_data:
                groq_status = health_data.get('groq_api_status', 'unknown')
                if groq_status == 'connected':
                    st.markdown('<div class="status-connected">🤖 Groq API: Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-error">🤖 Groq API: Disconnected</div>', unsafe_allow_html=True)
                
                st.info(f"🧠 Model: {health_data.get('model', 'Unknown')}")
                
                # Component status
                components = health_data.get('components', {})
                st.subheader("📊 Components")
                for component, status in components.items():
                    if status in ['ready', 'connected']:
                        st.success(f"✅ {component.replace('_', ' ').title()}")
                    else:
                        st.error(f"❌ {component.replace('_', ' ').title()}")
        else:
            st.error("❌ System Offline")
            st.warning("Please check if the backend server is running")
        
        # System info
        system_info = ui_handler.get_system_info()
        if system_info:
            st.subheader("⚙️ Configuration")
            st.json({
                "Model": system_info.get('current_model', 'Unknown'),
                "Max Tokens": system_info.get('configuration', {}).get('max_tokens', 'Unknown'),
                "Retrieval K": system_info.get('configuration', {}).get('top_k_retrieval', 'Unknown')
            })

def render_features_showcase():
    """Render system features showcase"""
    st.subheader("🚀 System Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='feature-box'>🔍 Hybrid Retrieval</div>", unsafe_allow_html=True)
        st.write("Combines semantic search and keyword matching for precision.")
    with col2:
        st.markdown("<div class='feature-box'>🕸️ Knowledge Graph</div>", unsafe_allow_html=True)
        st.write("Relates clauses and detects conflicts/dependencies.")
    with col3:
        st.markdown("<div class='feature-box'>🤖 Groq LLM</div>", unsafe_allow_html=True)
        st.write("High-quality reasoning and structured outputs.")

def render_help_section():
    """Render help and documentation section"""
    st.header("📚 Help & Documentation")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Quick Start", "📋 API Guide", "🔧 Troubleshooting", "💡 Tips"])
    
    with tab1:
        st.markdown("""
        ## 🚀 Quick Start Guide
        
        ### 1. Prepare Your Document
        - Ensure your document is accessible via a public URL
        - Supported formats: PDF, DOCX, EML
        - Document should contain structured text (insurance policies, contracts, etc.)
        
        ### 2. Ask Your Questions
        - Be specific and clear in your questions
        - Use question templates for common scenarios
        - You can ask up to 10 questions per document
        
        ### 3. Review Results
        - Check confidence scores for answer reliability
        - Review source clauses for verification
        - Look for conflicts between different clauses
        
        ### 4. Export & Share
        - Download results in JSON or CSV format
        - Share specific answers using the copy feature
        """)
    
    with tab2:
        st.markdown("""
        ## 📋 API Integration Guide
        
        ### Backend Endpoints
        ```
        GET  /                 - System information
        GET  /health          - Health check
        GET  /system-info     - Configuration details
        POST /hackrx/run      - Main query endpoint
        ```
        
        ### Request Format
        ```json
        {
          "documents": "https://example.com/document.pdf",
          "questions": [
            "What is covered under this policy?",
            "What are the exclusions?"
          ]
        }
        ```
        
        ### Response Format
        ```json
        {
          "answers": [
            {
              "question": "What is covered?",
              "answer": "The policy covers.",
              "confidence": 0.85,
              "source_clauses": ["Clause 1", "Clause 2"],
              "justification": "Analysis based on.",
              "conflicts_detected": []
            }
          ],
          "processing_time": 12.34,
          "audit_trail": ["Step 1", "Step 2"]
        }
        ```
        """)
    
    with tab3:
        st.markdown("""
        **Common issues & fixes**
        - Backend not running: start your server or use the local-start controls in the sidebar
        - Document URL unreachable: ensure file is public
        - Groq issues: verify GROQ_API_KEY environment variable
        """)
    with tab4:
        st.markdown("Tip: Break large documents into smaller ones for faster processing.")

def render_answer_card(answer_item: Dict, index: int):
    """Render a single answer card"""
    confidence = answer_item.get('confidence', 0.0)
    classes = "answer-card "
    if confidence >= 0.8:
        classes += "confidence-high"
    elif confidence >= 0.5:
        classes += "confidence-medium"
    else:
        classes += "confidence-low"
    
    st.markdown(f"<div class='{classes}'>", unsafe_allow_html=True)
    st.markdown(f"**Q{index+1}:** {answer_item.get('question','')}")
    st.markdown(f"**Answer:** {answer_item.get('answer','')}")
    st.markdown(f"<div class='confidence-score'>{confidence:.0%}</div>", unsafe_allow_html=True)
    if answer_item.get('source_clauses'):
        st.markdown("**Sources:**")
        for src in answer_item.get('source_clauses', []):
            st.markdown(f"- {src}")
    if answer_item.get('justification'):
        st.markdown("**Justification:**")
        st.code(answer_item.get('justification', ''), language="text")
    if answer_item.get('conflicts_detected'):
        st.warning(f"Conflicts detected: {', '.join(answer_item.get('conflicts_detected', []))}")
    st.markdown("</div>", unsafe_allow_html=True)

def render_processing_metrics(response_data: Dict):
    """Render processing performance metrics"""
    processing_time = response_data.get('processing_time', 0)
    answers = response_data.get('answers', [])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
    
    with col2:
        st.metric("❓ Questions Processed", len(answers))
    
    with col3:
        avg_confidence = sum(ans.get('confidence', 0) for ans in answers) / len(answers) if answers else 0
        st.metric("📊 Avg Confidence", f"{avg_confidence:.0%}")
    
    with col4:
        total_conflicts = sum(len(ans.get('conflicts_detected', [])) for ans in answers)
        st.metric("⚠️ Total Conflicts", total_conflicts)

def render_audit_trail(audit_trail: List[str]):
    """Render the processing audit trail"""
    st.subheader("🔍 Processing Audit Trail")
    
    with st.expander("View detailed processing steps", expanded=False):
        for step in audit_trail:
            st.markdown(f'<div class="audit-step">{step}</div>', unsafe_allow_html=True)

# ---------------------------
# Main application function — with added local-backend sidebar controls
# ---------------------------

def main():
    """Main application function"""
    load_css()
    st.title("📄 Document Analysis — Frontend (with optional local-backend)")

    # Sidebar: backend controls (ADDED)
    st.sidebar.header("Backend Options")
    api_url_input = st.sidebar.text_input("Backend base URL", value="http://localhost:8000")
    prefer_local = st.sidebar.checkbox("Prefer local backend if available", value=True)
    st.sidebar.markdown("**Local backend (optional)** — spawns your unmodified main.py")
    col_a, col_b = st.sidebar.columns([2,1])
    python_exec = col_a.text_input("Python executable for local backend", value=sys.executable)
    local_port = col_b.number_input("Local backend port", value=8000, min_value=1024, max_value=65535, step=1)
    main_py_path = st.sidebar.text_input("Path to main.py", value="main.py")
    auto_start_local = st.sidebar.checkbox("Auto-start local backend if not running", value=False)
    start_local_btn = st.sidebar.button("Start local backend")
    stop_local_btn = st.sidebar.button("Stop local backend")
    health_check_btn = st.sidebar.button("Check local backend health")

    # Create or update controller in session state
    if "local_backend_controller" not in st.session_state:
        st.session_state["local_backend_controller"] = LocalBackendController(
            python_executable=python_exec,
            main_path=main_py_path,
            host="127.0.0.1",
            port=int(local_port),
            env=os.environ.copy()
        )
    else:
        # update controller params in case user changed them
        controller: LocalBackendController = st.session_state["local_backend_controller"]
        controller.python_executable = python_exec
        controller.main_path = main_py_path
        controller.port = int(local_port)

    controller: LocalBackendController = st.session_state["local_backend_controller"]

    # Button actions
    if start_local_btn:
        st.sidebar.info("Starting local backend (this runs your unmodified main.py)...")
        ok, msg = controller.start(timeout=30.0)
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)

    if stop_local_btn:
        ok, msg = controller.stop()
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.warning(msg)

    if health_check_btn:
        if controller.is_running():
            st.sidebar.success(f"Backend reachable at {controller.base_url}")
        else:
            st.sidebar.error(f"Backend not reachable at {controller.base_url}")

    # Auto-start behavior
    if auto_start_local and prefer_local and not controller.is_running():
        st.sidebar.info("Auto-starting local backend...")
        ok, msg = controller.start(timeout=20.0)
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.warning(msg)

    # Decide final backend URL to use
    def choose_backend_url() -> str:
        if prefer_local and controller.is_running():
            return controller.base_url
        return api_url_input.rstrip("/")

    # Initialize UI handler (keeps original behavior)
    ui_handler = DocumentQueryUI()
    # Update UI handler's api_base_url to chosen backend
    ui_handler.api_base_url = choose_backend_url()

    # Header + status
    render_header()
    render_sidebar_status(ui_handler)
    render_features_showcase()

    st.markdown("---")

    # Main query interface
    st.header("📄 Document Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        document_url = st.text_input(
            "📎 Document URL",
            placeholder="https://example.com/document.pdf",
            help="Enter the URL of the PDF, DOCX, or EML document to analyze"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        use_sample = st.button("📋 Use Sample Document", type="secondary")
        if use_sample:
            document_url = "https://example.com/sample-insurance-policy.pdf"
            st.experimental_rerun()

    # Questions input
    st.subheader("❓ Questions to Ask")
    with st.expander("🎯 Quick Question Templates"):
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("💰 Coverage Questions"):
                st.session_state.questions = [
                    "What medical conditions are covered under this policy?",
                    "What is the maximum coverage amount?",
                    "Are pre-existing conditions covered?"
                ]
        with c2:
            if st.button("❌ Exclusion Questions"):
                st.session_state.questions = [
                    "What treatments or conditions are excluded?",
                    "Are there any age-related exclusions?",
                    "What activities void the coverage?"
                ]
        with c3:
            if st.button("⏰ Timing Questions"):
                st.session_state.questions = [
                    "What is the waiting period for coverage?",
                    "When does the policy become effective?",
                    "Are there any time limits for claims?"
                ]

    # Dynamic question inputs
    if 'questions' not in st.session_state:
        st.session_state.questions = [""]

    questions = []
    for i in range(len(st.session_state.questions)):
        question = st.text_input(
            f"Question {i+1}",
            value=st.session_state.questions[i],
            key=f"question_{i}"
        )
        if question.strip():
            questions.append(question.strip())

    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("➕ Add Question"):
            st.session_state.questions.append("")
            st.experimental_rerun()
    with col_b:
        if len(st.session_state.questions) > 1 and st.button("➖ Remove Last Question"):
            st.session_state.questions.pop()
            st.experimental_rerun()

    # Process button
    st.markdown("---")
    colx, coly, colz = st.columns([1, 2, 1])
    with coly:
        process_button = st.button(
            "🚀 Analyze Document",
            type="primary",
            use_container_width=True,
            disabled=not document_url or not questions
        )

    # Execution: call backend using ui_handler.query_documents (unchanged)
    if process_button:
        if not document_url:
            st.error("❌ Please provide a document URL")
        elif not questions:
            st.error("❌ Please provide at least one question")
        else:
            # Update ui_handler.api_base_url before calling
            ui_handler.api_base_url = choose_backend_url()
            st.info(f"🔗 Using backend: {ui_handler.api_base_url}")
            with st.spinner("🔄 Processing document and questions. This may take some time..."):
                progress_bar = st.progress(0)
                try:
                    ok, response_data = ui_handler.query_documents(document_url, questions)
                    progress_bar.progress(100)
                except Exception as e:
                    ok = False
                    response_data = {"error": str(e)}

                if ok:
                    st.success("✅ Analysis completed")
                    render_processing_metrics(response_data)
                    st.markdown("---")
                    st.header("Results")
                    answers = response_data.get('answers', [])
                    if answers:
                        for i, answer_item in enumerate(answers):
                            render_answer_card(answer_item, i)
                    else:
                        st.warning("No answers were generated")
                    
                    # Audit trail
                    audit_trail = response_data.get('audit_trail', [])
                    if audit_trail:
                        render_audit_trail(audit_trail)
                    
                    # Download results
                    st.markdown("---")
                    st.subheader("💾 Export Results")
                    
                    col1d, col2d = st.columns(2)
                    
                    with col1d:
                        # JSON export
                        json_data = json.dumps(response_data, indent=2)
                        st.download_button(
                            label="📄 Download JSON",
                            data=json_data,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with col2d:
                        # CSV export for answers
                        if answers:
                            df_data = []
                            for i, ans in enumerate(answers, 1):
                                df_data.append({
                                    'Question_ID': i,
                                    'Question': ans.get('question', ''),
                                    'Answer': ans.get('answer', ''),
                                    'Confidence': ans.get('confidence', 0),
                                    'Source_Clauses': '; '.join(ans.get('source_clauses', [])),
                                    'Conflicts': '; '.join(ans.get('conflicts_detected', [])),
                                    'Justification': ans.get('justification', '')
                                })
                            
                            df = pd.DataFrame(df_data)
                            csv_data = df.to_csv(index=False)
                            
                            st.download_button(
                                label="📊 Download CSV",
                                data=csv_data,
                                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("❌ Analysis failed")
                    error_msg = response_data.get('error', 'Unknown error occurred')
                    st.error(f"Error details: {error_msg}")
                    
                    # Troubleshooting tips
                    with st.expander("🔧 Troubleshooting Tips"):
                        st.markdown("""
                        **Common issues:**
                        1. **Document URL not accessible**: Ensure the URL is publicly accessible
                        2. **Server not running**: Check if the backend service is running on the correct port
                        3. **Groq API key**: Verify that GROQ_API_KEY environment variable is set
                        4. **Network issues**: Check your internet connection
                        5. **Document format**: Ensure document is PDF, DOCX, or EML format
                        
                        **Quick fixes:**
                        - Try a different document URL
                        - Restart the backend server (use the sidebar controls if you want the UI to start it)
                        - Check system status in the sidebar
                        - Reduce the number of questions
                        """)
    
    st.markdown("---")
    st.markdown("📌 Tip: If you want this UI to spawn your backend automatically, enable the local backend controls in the sidebar and press **Start local backend**. The UI will then prefer the local backend when available.")

if __name__ == "__main__":
    main()
