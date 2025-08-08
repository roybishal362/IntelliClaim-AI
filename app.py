import subprocess
import time
import os

# Start backend API from main.py
backend_process = subprocess.Popen(
    ["python", "main.py"],
    env=os.environ.copy()
)

# Give the backend a moment to start
time.sleep(3)

"""
🧠🔥 PROFESSIONAL STREAMLIT UI FOR ENHANCED DOCUMENT QUERY SYSTEM
High-quality frontend matching backend excellence
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import base64
from io import BytesIO

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
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
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
        self.api_base_url = "http://localhost:8000"  # Change to your deployed URL
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
        st.markdown("""
        <div class="feature-box">
            <h4>🧠 Groq API Integration</h4>
            <p>Best open-source LLMs with lightning-fast inference</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4>🔍 Hybrid Retrieval</h4>
            <p>Semantic + Keyword search for comprehensive coverage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h4>🕸️ Knowledge Graph</h4>
            <p>Conflict detection and relationship mapping</p>
        </div>
        """, unsafe_allow_html=True)

def render_confidence_gauge(confidence: float):
    """Render confidence score as a gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250, font={'size': 12})
    return fig

def render_answer_card(answer_item: Dict, index: int):
    """Render a single answer card"""
    confidence = answer_item.get('confidence', 0.0)
    
    # Determine confidence class
    if confidence >= 0.8:
        confidence_class = "confidence-high"
        confidence_color = "#28a745"
    elif confidence >= 0.5:
        confidence_class = "confidence-medium" 
        confidence_color = "#ffc107"
    else:
        confidence_class = "confidence-low"
        confidence_color = "#dc3545"
    
    # Create expandable answer card
    with st.expander(f"❓ Question {index + 1}: {answer_item.get('question', '')[:100]}...", expanded=True):
        
        # Confidence score at the top
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Question:** {answer_item.get('question', '')}")
        
        with col2:
            fig = render_confidence_gauge(confidence)
            st.plotly_chart(fig, use_container_width=True)
        
        # Main answer
        st.markdown("### 📝 Answer")
        st.markdown(f"""
        <div class="answer-card {confidence_class}">
            <div class="confidence-score" style="color: {confidence_color}">
                {confidence:.0%} Confidence
            </div>
            <div style="margin-right: 60px;">
                {answer_item.get('answer', 'No answer provided')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional details in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Sources", "🔍 Justification", "⚠️ Conflicts", "📋 Metadata"])
        
        with tab1:
            source_clauses = answer_item.get('source_clauses', [])
            if source_clauses:
                st.markdown("**Source Clauses:**")
                for i, clause in enumerate(source_clauses, 1):
                    st.markdown(f"{i}. {clause}")
            else:
                st.info("No source clauses identified")
        
        with tab2:
            justification = answer_item.get('justification', '')
            if justification:
                st.markdown("**Analysis Process:**")
                st.code(justification, language="text")
            else:
                st.info("No justification provided")
        
        with tab3:
            conflicts = answer_item.get('conflicts_detected', [])
            if conflicts:
                st.warning(f"⚠️ {len(conflicts)} potential conflicts detected:")
                for conflict in conflicts:
                    st.markdown(f"• {conflict}")
            else:
                st.success("✅ No conflicts detected")
        
        with tab4:
            st.json({
                "confidence_score": confidence,
                "source_count": len(source_clauses),
                "conflicts_count": len(conflicts),
                "question_length": len(answer_item.get('question', '')),
                "answer_length": len(answer_item.get('answer', ''))
            })

def render_audit_trail(audit_trail: List[str]):
    """Render the processing audit trail"""
    st.subheader("🔍 Processing Audit Trail")
    
    with st.expander("View detailed processing steps", expanded=False):
        for step in audit_trail:
            st.markdown(f'<div class="audit-step">{step}</div>', unsafe_allow_html=True)

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

def main():
    """Main application function"""
    load_css()
    ui_handler = DocumentQueryUI()
    
    # Header
    render_header()
    
    # Sidebar status
    render_sidebar_status(ui_handler)
    
    # Features showcase
    render_features_showcase()
    
    st.markdown("---")
    
    # Main query interface
    st.header("📄 Document Analysis")
    
    # Input section
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
            st.rerun()
    
    # Questions input
    st.subheader("❓ Questions to Ask")
    
    # Pre-defined question templates
    with st.expander("🎯 Quick Question Templates"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💰 Coverage Questions"):
                st.session_state.questions = [
                    "What medical conditions are covered under this policy?",
                    "What is the maximum coverage amount?",
                    "Are pre-existing conditions covered?"
                ]
        
        with col2:
            if st.button("❌ Exclusion Questions"):
                st.session_state.questions = [
                    "What treatments or conditions are excluded?",
                    "Are there any age-related exclusions?",
                    "What activities void the coverage?"
                ]
        
        with col3:
            if st.button("⏰ Timing Questions"):
                st.session_state.questions = [
                    "What is the waiting period for coverage?",
                    "When does the policy become effective?",
                    "Are there any time limits for claims?"
                ]
    
    # Dynamic questions input
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
    
    # Add/Remove question buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Add Question"):
            st.session_state.questions.append("")
            st.rerun()
    
    with col2:
        if len(st.session_state.questions) > 1 and st.button("➖ Remove Last Question"):
            st.session_state.questions.pop()
            st.rerun()
    
    # Process button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Analyze Document",
            type="primary",
            use_container_width=True,
            disabled=not document_url or not questions
        )
    
    # Processing and results
    if process_button:
        if not document_url:
            st.error("❌ Please provide a document URL")
        elif not questions:
            st.error("❌ Please provide at least one question")
        else:
            # Processing indicator
            with st.spinner("🔄 Processing document and questions..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate processing steps
                steps = [
                    "📥 Downloading document...",
                    "📄 Parsing and extracting content...", 
                    "🕸️ Building knowledge graph...",
                    "🔍 Setting up hybrid retrieval...",
                    "🧠 Processing questions with LLM...",
                    "✅ Finalizing results..."
                ]
                
                for i, step in enumerate(steps):
                    status_text.text(step)
                    progress_bar.progress((i + 1) / len(steps))
                    time.sleep(0.5)
                
                # Actual API call
                status_text.text("🤖 Querying AI system...")
                success, response_data = ui_handler.query_documents(document_url, questions)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
            
            if success:
                st.success("🎉 Document analysis completed successfully!")
                
                # Processing metrics
                render_processing_metrics(response_data)
                
                st.markdown("---")
                
                # Display answers
                st.header("📋 Analysis Results")
                
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
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # JSON export
                    json_data = json.dumps(response_data, indent=2)
                    st.download_button(
                        label="📄 Download JSON",
                        data=json_data,
                        file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # CSV export for answers
                    if answers:
                        df_data = []
                        for i, answer in enumerate(answers, 1):
                            df_data.append({
                                'Question_ID': i,
                                'Question': answer.get('question', ''),
                                'Answer': answer.get('answer', ''),
                                'Confidence': answer.get('confidence', 0),
                                'Source_Clauses': '; '.join(answer.get('source_clauses', [])),
                                'Conflicts': '; '.join(answer.get('conflicts_detected', [])),
                                'Justification': answer.get('justification', '')
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
                    - Restart the backend server
                    - Check system status in the sidebar
                    - Reduce the number of questions
                    """)

def render_analytics_dashboard():
    """Render analytics dashboard for query history"""
    st.header("📊 Analytics Dashboard")
    
    # Mock data for demonstration
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if st.session_state.query_history:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.query_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution
            fig_conf = px.histogram(
                df, x='avg_confidence', 
                title="Confidence Score Distribution",
                nbins=20
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        
        with col2:
            # Processing time trends
            fig_time = px.line(
                df, y='processing_time', 
                title="Processing Time Trends",
                markers=True
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Recent queries table
        st.subheader("🕐 Recent Queries")
        st.dataframe(df[['timestamp', 'questions_count', 'avg_confidence', 'processing_time']], use_container_width=True)
    
    else:
        st.info("📈 No query history available yet. Run some document analyses to see analytics!")

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
              "answer": "The policy covers...",
              "confidence": 0.85,
              "source_clauses": ["Clause 1", "Clause 2"],
              "justification": "Analysis based on...",
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
        ## 🔧 Troubleshooting Guide
        
        ### Common Issues
        
        **🔴 System Offline**
        - Check if backend server is running
        - Verify the API URL in settings
        - Ensure port 8000 is accessible
        
        **🔴 Groq API Issues**
        - Verify GROQ_API_KEY environment variable
        - Check Groq API quota and limits
        - Try switching to a different model
        
        **🔴 Document Processing Errors**
        - Ensure document URL is publicly accessible
        - Check document format (PDF, DOCX, EML only)
        - Verify document contains readable text
        
        **🔴 Low Confidence Scores**
        - Rephrase questions more specifically
        - Check if document contains relevant information
        - Try breaking complex questions into simpler ones
        
        ### Performance Optimization
        - Use fewer questions for faster processing
        - Choose appropriate document chunk sizes
        - Monitor system resources during processing
        """)
    
    with tab4:
        st.markdown("""
        ## 💡 Pro Tips for Better Results
        
        ### 🎯 Question Crafting
        - **Be Specific**: Instead of "What's covered?", ask "What medical procedures are covered for outpatient care?"
        - **Include Context**: "For a 35-year-old, what is the waiting period for dental coverage?"
        - **Ask About Edge Cases**: "What happens if I need treatment while traveling abroad?"
        
        ### 📄 Document Preparation
        - Use high-quality, text-searchable documents
        - Ensure proper document structure with clear sections
        - Avoid heavily redacted or image-only documents
        
        ### 🔍 Interpreting Results
        - **High Confidence (80%+)**: Answer is very reliable
        - **Medium Confidence (50-80%)**: Answer is good but verify with source clauses
        - **Low Confidence (<50%)**: Answer may be uncertain, review carefully
        
        ### 🕸️ Understanding Conflicts
        - Red conflicts indicate contradictory clauses
        - Review all conflicting clauses before making decisions
        - Consider asking clarifying questions about conflicts
        
        ### 📊 Using Analytics
        - Track confidence trends over time
        - Identify types of questions that work best
        - Monitor processing performance
        """)

def sidebar_navigation():
    """Render sidebar navigation"""
    with st.sidebar:
        st.markdown("---")
        st.header("🧭 Navigation")
        
        nav_option = st.selectbox(
            "Choose Section",
            ["🏠 Main Query", "📊 Analytics", "📚 Help"],
            index=0
        )
        
        st.markdown("---")
        
        # Quick settings
        st.header("⚙️ Settings")
        
        # API URL setting
        api_url = st.text_input(
            "🔗 Backend URL",
            value="http://localhost:8000",
            help="Change this to your deployed backend URL"
        )
        
        if api_url != "http://localhost:8000":
            st.session_state.custom_api_url = api_url
        
        # Display current model info
        st.markdown("---")
        st.header("🤖 Model Info")
        st.info("""
        **Current Setup:**
        - Provider: Groq API
        - Model: llama-3.1-70b-versatile
        - Embedding: all-MiniLM-L6-v2
        - Features: Hybrid Retrieval + KG
        """)
        
        return nav_option

# Update the main function to include navigation
def main():
    """Main application function with navigation"""
    load_css()
    ui_handler = DocumentQueryUI()
    
    # Sidebar navigation
    nav_option = sidebar_navigation()
    
    if nav_option == "🏠 Main Query":
        # Header
        render_header()
        
        # Features showcase
        render_features_showcase()
        
        st.markdown("---")
        
        # Main query interface (existing code)
        st.header("📄 Document Analysis")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            document_url = st.text_input(
                "📎 Document URL",
                placeholder="https://example.com/document.pdf",
                help="Enter the URL of the PDF, DOCX, or EML document to analyze"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            use_sample = st.button("📋 Use Sample Document", type="secondary")
            if use_sample:
                document_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
                st.rerun()
        
        # Questions input
        st.subheader("❓ Questions to Ask")
        
        # Pre-defined question templates
        with st.expander("🎯 Quick Question Templates"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💰 Coverage Questions"):
                    st.session_state.questions = [
                        "What medical conditions are covered under this policy?",
                        "What is the maximum coverage amount?",
                        "Are pre-existing conditions covered?"
                    ]
            
            with col2:
                if st.button("❌ Exclusion Questions"):
                    st.session_state.questions = [
                        "What treatments or conditions are excluded?",
                        "Are there any age-related exclusions?",
                        "What activities void the coverage?"
                    ]
            
            with col3:
                if st.button("⏰ Timing Questions"):
                    st.session_state.questions = [
                        "What is the waiting period for coverage?",
                        "When does the policy become effective?",
                        "Are there any time limits for claims?"
                    ]
        
        # Dynamic questions input
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
        
        # Add/Remove question buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ Add Question"):
                st.session_state.questions.append("")
                st.rerun()
        
        with col2:
            if len(st.session_state.questions) > 1 and st.button("➖ Remove Last Question"):
                st.session_state.questions.pop()
                st.rerun()
        
        # Process button
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_button = st.button(
                "🚀 Analyze Document",
                type="primary",
                use_container_width=True,
                disabled=not document_url or not questions
            )
        
        # Processing and results (existing code continues...)
        if process_button:
            if not document_url:
                st.error("❌ Please provide a document URL")
            elif not questions:
                st.error("❌ Please provide at least one question")
            else:
                # Processing indicator
                with st.spinner("🔄 Processing document and questions..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate processing steps
                    steps = [
                        "📥 Downloading document...",
                        "📄 Parsing and extracting content...", 
                        "🕸️ Building knowledge graph...",
                        "🔍 Setting up hybrid retrieval...",
                        "🧠 Processing questions with LLM...",
                        "✅ Finalizing results..."
                    ]
                    
                    for i, step in enumerate(steps):
                        status_text.text(step)
                        progress_bar.progress((i + 1) / len(steps))
                        time.sleep(0.5)
                    
                    # Actual API call
                    status_text.text("🤖 Querying AI system...")
                    success, response_data = ui_handler.query_documents(document_url, questions)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                
                if success:
                    st.success("🎉 Document analysis completed successfully!")
                    
                    # Store in history
                    if 'query_history' not in st.session_state:
                        st.session_state.query_history = []
                    
                    answers = response_data.get('answers', [])
                    avg_confidence = sum(ans.get('confidence', 0) for ans in answers) / len(answers) if answers else 0
                    
                    st.session_state.query_history.append({
                        'timestamp': datetime.now(),
                        'questions_count': len(questions),
                        'avg_confidence': avg_confidence,
                        'processing_time': response_data.get('processing_time', 0)
                    })
                    
                    # Processing metrics
                    render_processing_metrics(response_data)
                    
                    st.markdown("---")
                    
                    # Display answers
                    st.header("📋 Analysis Results")
                    
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
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # JSON export
                        json_data = json.dumps(response_data, indent=2)
                        st.download_button(
                            label="📄 Download JSON",
                            data=json_data,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        # CSV export for answers
                        if answers:
                            df_data = []
                            for i, answer in enumerate(answers, 1):
                                df_data.append({
                                    'Question_ID': i,
                                    'Question': answer.get('question', ''),
                                    'Answer': answer.get('answer', ''),
                                    'Confidence': answer.get('confidence', 0),
                                    'Source_Clauses': '; '.join(answer.get('source_clauses', [])),
                                    'Conflicts': '; '.join(answer.get('conflicts_detected', [])),
                                    'Justification': answer.get('justification', '')
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
                        - Restart the backend server
                        - Check system status in the sidebar
                        - Reduce the number of questions
                        """)
    
    elif nav_option == "📊 Analytics":
        render_analytics_dashboard()
    
    elif nav_option == "📚 Help":
        render_help_section()

if __name__ == "__main__":
    main()
