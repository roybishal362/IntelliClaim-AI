"""
🧠🔥 UNIFIED AI DOCUMENT QUERY SYSTEM
FastAPI Backend + Streamlit Frontend in ONE FILE for easy deployment
"""

import os
import json
import logging
import asyncio
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tempfile
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import networkx as nx
import re

# Check if running in Streamlit mode
import sys
IS_STREAMLIT = 'streamlit' in sys.modules or 'streamlit' in str(sys.argv)

if not IS_STREAMLIT:
    # FastAPI imports (only when not in Streamlit)
    from fastapi import FastAPI, HTTPException, Depends, Header
    from pydantic import BaseModel, Field
    import uvicorn

# Common imports for both modes
from langchain.document_loaders import (
    UnstructuredPDFLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredEmailLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Groq API
from groq import Groq

# Streamlit (only import when in Streamlit mode)
if IS_STREAMLIT:
    import streamlit as st

# Whoosh for keyword search
try:
    from whoosh.index import create_in
    from whoosh.fields import Schema, TEXT, ID
    from whoosh.qparser import QueryParser
    WHOOSH_AVAILABLE = True
except ImportError:
    WHOOSH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# 🔧 CONFIGURATION
# ========================

class Config:
    # Groq API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
    
    # Best open-source models on Groq
    LLM_MODEL_OPTIONS = [
        "llama-3.1-70b-versatile",      # Best overall
        "llama-3.1-8b-instant",        # Fast
        "mixtral-8x7b-32768",          # Excellent reasoning
        "gemma2-9b-it",                # Good balance
    ]
    
    LLM_MODEL_NAME = LLM_MODEL_OPTIONS[0]
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Processing
    MAX_TOKENS = 4096
    TEMPERATURE = 0.1
    TOP_P = 0.9
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K_RETRIEVAL = 6
    
    # Hybrid weights
    SEMANTIC_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3
    
    # API
    HACKRX_TOKEN = "90bf5fcfc3d0de340e50ac29a5cf53eb6da42e7a24af15f2111186878d510d6c"
    MAX_QUESTIONS = 10

# ========================
# 📊 DATA MODELS
# ========================

@dataclass
class ClauseNode:
    clause_id: str
    content: str
    clause_type: str
    age_limit: Optional[int] = None
    waiting_period: Optional[int] = None
    amount_limit: Optional[float] = None
    dependencies: List[str] = None
    conflicts: List[str] = None

if not IS_STREAMLIT:
    # FastAPI models (only when not in Streamlit)
    class QueryRequest(BaseModel):
        documents: str = Field(..., description="URL to document")
        questions: List[str] = Field(..., description="List of questions")

    class AnswerItem(BaseModel):
        question: str
        answer: str
        confidence: float
        source_clauses: List[str]
        justification: str
        conflicts_detected: List[str] = []

    class QueryResponse(BaseModel):
        answers: List[AnswerItem]
        processing_time: float
        audit_trail: List[str]

# ========================
# 🤖 GROQ LLM MANAGER
# ========================

class GroqLLM:
    def __init__(self):
        if not Config.GROQ_API_KEY or Config.GROQ_API_KEY == "your_groq_api_key_here":
            if IS_STREAMLIT:
                st.error("❌ Please set GROQ_API_KEY in your environment or Streamlit secrets")
                st.stop()
            else:
                raise ValueError("Please set GROQ_API_KEY environment variable")
        
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model_name = Config.LLM_MODEL_NAME
        logger.info(f"🚀 Initialized Groq LLM: {self.model_name}")
    
    async def generate_response(self, prompt: str, max_tokens: int = 1024) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert legal and insurance document analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"Error processing request: {str(e)}"

# ========================
# 🧾 DOCUMENT PROCESSOR
# ========================

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "]
        )
    
    async def download_document(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine file type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                suffix = '.pdf'
            elif 'word' in content_type or url.lower().endswith(('.docx', '.doc')):
                suffix = '.docx'
            elif 'email' in content_type or url.lower().endswith('.eml'):
                suffix = '.eml'
            else:
                suffix = '.pdf'
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(response.content)
                return tmp_file.name
        except Exception as e:
            logger.error(f"Download error: {e}")
            raise Exception(f"Failed to download document: {e}")
    
    async def parse_document(self, file_path: str) -> Tuple[List[Document], List[ClauseNode]]:
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                loader = UnstructuredPDFLoader(file_path)
            elif file_ext in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_ext == '.eml':
                loader = UnstructuredEmailLoader(file_path)
            else:
                loader = UnstructuredPDFLoader(file_path)
            
            documents = loader.load()
            
            # Add metadata and split
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source_file': Path(file_path).name,
                    'chunk_id': i,
                    'processed_at': datetime.now().isoformat()
                })
            
            chunks = self.text_splitter.split_documents(documents)
            clause_nodes = []
            
            for i, chunk in enumerate(chunks):
                clause_id = f"clause_{i+1}"
                chunk.metadata['clause_id'] = clause_id
                chunk.metadata['clause_no'] = i + 1
                
                # Create clause node with structure extraction
                clause_node = self._extract_clause_structure(chunk.page_content, clause_id)
                clause_nodes.append(clause_node)
            
            logger.info(f"📄 Parsed into {len(chunks)} chunks")
            return chunks, clause_nodes
            
        except Exception as e:
            logger.error(f"Parse error: {e}")
            raise Exception(f"Failed to parse document: {e}")
    
    def _extract_clause_structure(self, content: str, clause_id: str) -> ClauseNode:
        content_lower = content.lower()
        
        # Determine clause type
        clause_type = "general"
        if any(word in content_lower for word in ['cover', 'benefit', 'include']):
            clause_type = "coverage"
        elif any(word in content_lower for word in ['exclude', 'not cover', 'except']):
            clause_type = "exclusion"
        elif any(word in content_lower for word in ['condition', 'requirement']):
            clause_type = "condition"
        
        # Extract age limits
        age_limit = None
        age_match = re.search(r'age\s+(\d+)|(\d+)\s+years?\s+old', content_lower)
        if age_match:
            age_limit = int(age_match.group(1) or age_match.group(2))
        
        # Extract waiting periods  
        waiting_period = None
        wait_match = re.search(r'(\d+)\s+days?\s+wait|waiting\s+period\s+(\d+)', content_lower)
        if wait_match:
            waiting_period = int(wait_match.group(1) or wait_match.group(2))
        
        return ClauseNode(
            clause_id=clause_id,
            content=content,
            clause_type=clause_type,
            age_limit=age_limit,
            waiting_period=waiting_period,
            dependencies=[],
            conflicts=[]
        )

# ========================
# 🔍 HYBRID RETRIEVAL
# ========================

class HybridRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None
        self.documents = None
    
    async def setup(self, documents: List[Document]):
        self.documents = documents
        # Create FAISS vectorstore
        self.vectorstore = await asyncio.get_event_loop().run_in_executor(
            None, FAISS.from_documents, documents, self.embeddings
        )
    
    def retrieve(self, query: str, top_k: int = Config.TOP_K_RETRIEVAL) -> List[Dict]:
        try:
            # Semantic search
            semantic_results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            
            retrieved = []
            for doc, score in semantic_results:
                # Normalize semantic score
                semantic_score = max(0.0, 1.0 - (score / 2.0))
                
                # Simple keyword scoring
                keyword_score = self._keyword_score(query, doc.page_content)
                
                # Combined score
                final_score = (Config.SEMANTIC_WEIGHT * semantic_score + 
                              Config.KEYWORD_WEIGHT * keyword_score)
                
                retrieved.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score,
                    'final_score': final_score,
                    'clause_id': doc.metadata.get('clause_id', ''),
                    'source_info': f"Clause {doc.metadata.get('clause_no', 'N/A')}"
                })
            
            # Sort by final score
            retrieved.sort(key=lambda x: x['final_score'], reverse=True)
            return retrieved
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def _keyword_score(self, query: str, content: str) -> float:
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)

# ========================
# 🎯 UNIFIED PROCESSING ENGINE
# ========================

class UnifiedQueryEngine:
    def __init__(self):
        self.llm = None
        self.doc_processor = DocumentProcessor()
        self.retriever = HybridRetriever()
        self.kg = nx.DiGraph()
        
        # Initialize LLM
        try:
            self.llm = GroqLLM()
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            if IS_STREAMLIT:
                st.error(f"❌ Failed to initialize LLM: {e}")
    
    async def process_query(self, document_url: str, questions: List[str]) -> Dict:
        start_time = datetime.now()
        audit_trail = []
        
        try:
            audit_trail.append("🚀 Starting document analysis...")
            
            # Download and parse
            audit_trail.append("📥 Downloading document...")
            file_path = await self.doc_processor.download_document(document_url)
            
            audit_trail.append("📄 Parsing document...")
            documents, clause_nodes = await self.doc_processor.parse_document(file_path)
            
            # Setup retrieval
            audit_trail.append("🔍 Setting up retrieval system...")
            await self.retriever.setup(documents)
            
            # Build simple knowledge graph
            audit_trail.append("🕸️ Building knowledge graph...")
            self._build_knowledge_graph(clause_nodes)
            
            answers = []
            
            # Process each question
            for i, question in enumerate(questions, 1):
                audit_trail.append(f"❓ Processing question {i}...")
                
                # Retrieve relevant clauses
                retrieved = self.retriever.retrieve(question)
                
                # Detect conflicts
                conflicts = self._detect_conflicts(retrieved)
                
                # Generate answer
                answer = await self._generate_answer(question, retrieved, conflicts)
                answers.append(answer)
                
                audit_trail.append(f"✅ Question {i} completed (confidence: {answer['confidence']:.0%})")
            
            # Cleanup
            try:
                os.unlink(file_path)
            except:
                pass
            
            processing_time = (datetime.now() - start_time).total_seconds()
            audit_trail.append(f"🏁 Completed in {processing_time:.2f}s")
            
            return {
                'answers': answers,
                'processing_time': processing_time,
                'audit_trail': audit_trail
            }
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(error_msg)
            audit_trail.append(f"❌ Error: {error_msg}")
            
            return {
                'answers': [],
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'audit_trail': audit_trail,
                'error': error_msg
            }
    
    def _build_knowledge_graph(self, clause_nodes: List[ClauseNode]):
        """Build simple knowledge graph"""
        for node in clause_nodes:
            self.kg.add_node(node.clause_id, **node.__dict__)
        
        # Detect simple relationships
        for i, node1 in enumerate(clause_nodes):
            for j, node2 in enumerate(clause_nodes):
                if i >= j:
                    continue
                
                # Conflict detection (exclusion vs coverage)
                if (node1.clause_type == "coverage" and node2.clause_type == "exclusion") or \
                   (node1.clause_type == "exclusion" and node2.clause_type == "coverage"):
                    words1 = set(node1.content.lower().split())
                    words2 = set(node2.content.lower().split())
                    if len(words1.intersection(words2)) > 5:
                        self.kg.add_edge(node1.clause_id, node2.clause_id, relation="conflicts")
    
    def _detect_conflicts(self, retrieved: List[Dict]) -> List[str]:
        """Detect conflicts using knowledge graph"""
        conflicts = []
        for item in retrieved:
            clause_id = item['clause_id']
            if clause_id in self.kg:
                for neighbor in self.kg.neighbors(clause_id):
                    if self.kg[clause_id][neighbor].get('relation') == 'conflicts':
                        conflicts.append(f"Conflicts with {neighbor}")
        return list(set(conflicts))
    
    async def _generate_answer(self, question: str, retrieved: List[Dict], conflicts: List[str]) -> Dict:
        """Generate structured answer"""
        if not retrieved:
            return {
                'question': question,
                'answer': "I couldn't find relevant information in the document.",
                'confidence': 0.0,
                'source_clauses': [],
                'justification': "No relevant clauses found",
                'conflicts_detected': []
            }
        
        # Prepare context
        context = ""
        source_clauses = []
        
        for i, item in enumerate(retrieved[:4], 1):
            context += f"\n[Clause {i}] {item['content'][:400]}...\n"
            source_clauses.append(f"Clause {item['metadata'].get('clause_no', i)} (Score: {item['final_score']:.2f})")
        
        # Create prompt
        prompt = f"""Analyze this insurance/legal document question using the provided clauses.

Question: {question}

Relevant Clauses:
{context}

Provide a comprehensive answer that:
1. Directly answers the question
2. Includes specific details (amounts, periods, conditions)
3. Mentions any limitations or exclusions
4. Is clear and factual

Answer:"""
        
        answer_text = await self.llm.generate_response(prompt, max_tokens=600)
        
        # Calculate confidence
        avg_score = sum(item['final_score'] for item in retrieved[:3]) / min(3, len(retrieved))
        confidence = min(0.95, max(0.1, avg_score))
        
        return {
            'question': question,
            'answer': answer_text,
            'confidence': round(confidence, 2),
            'source_clauses': source_clauses,
            'justification': f"Based on {len(retrieved)} retrieved clauses with avg score {avg_score:.2f}",
            'conflicts_detected': conflicts
        }

# ========================
# 🚀 FASTAPI BACKEND (Only when not in Streamlit)
# ========================

if not IS_STREAMLIT:
    app = FastAPI(title="🧠 AI Document Query API", version="3.0.0")
    query_engine = None
    
    @app.on_event("startup")
    async def startup():
        global query_engine
        query_engine = UnifiedQueryEngine()
    
    def verify_token(authorization: str = Header(None)):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization[7:]
        if token != Config.HACKRX_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid token")
        return token
    
    @app.get("/")
    async def root():
        return {
            "message": "🧠🔥 AI Document Query System",
            "version": "3.0.0",
            "status": "ready",
            "model": Config.LLM_MODEL_NAME
        }
    
    @app.get("/health")
    async def health():
        groq_status = "connected" if query_engine and query_engine.llm else "disconnected"
        return {
            "status": "healthy",
            "groq_api_status": groq_status,
            "model": Config.LLM_MODEL_NAME,
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/hackrx/run", response_model=QueryResponse)
    async def hackrx_run(request: QueryRequest, token: str = Depends(verify_token)):
        if not query_engine:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        if len(request.questions) > Config.MAX_QUESTIONS:
            raise HTTPException(status_code=400, detail=f"Max {Config.MAX_QUESTIONS} questions")
        
        try:
            result = await query_engine.process_query(request.documents, request.questions)
            if 'error' in result:
                raise HTTPException(status_code=500, detail=result['error'])
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# ========================
# 💻 STREAMLIT FRONTEND (Only when in Streamlit)
# ========================

if IS_STREAMLIT:
    
    # Page config
    st.set_page_config(
        page_title="🧠 AI Document Query System",
        page_icon="🔥",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .answer-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .confidence-high { border-left-color: #28a745; }
    .confidence-medium { border-left-color: #ffc107; }
    .confidence-low { border-left-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'query_engine' not in st.session_state:
        with st.spinner("🚀 Initializing AI system..."):
            st.session_state.query_engine = UnifiedQueryEngine()
    
    if 'questions' not in st.session_state:
        st.session_state.questions = [""]
    
    # Header
    st.markdown('<div class="main-header">🧠🔥 AI Document Query System</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Enhanced hybrid retrieval with knowledge graphs • Powered by Groq API</div>', unsafe_allow_html=True)
    
    # Features showcase
    st.subheader("🚀 System Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-box"><h4>🧠 Groq API</h4><p>Best open-source LLMs</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-box"><h4>🔍 Hybrid Search</h4><p>Semantic + Keyword</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-box"><h4>🕸️ Knowledge Graph</h4><p>Conflict Detection</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main interface
    st.header("📄 Document Analysis")
    
    # Document input
    col1, col2 = st.columns([3, 1])
    with col1:
        document_url = st.text_input(
            "📎 Document URL",
            placeholder="https://example.com/document.pdf",
            help="PDF, DOCX, or EML document URL"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📋 Use Sample", type="secondary"):
            document_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
            st.rerun()
    
    # Questions section
    st.subheader("❓ Your Questions")
    
    # Quick templates
    with st.expander("🎯 Quick Templates"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💰 Coverage"):
                st.session_state.questions = [
                    "What is covered under this policy?",
                    "What is the maximum coverage amount?",
                    "Are pre-existing conditions covered?"
                ]
                st.rerun()
        with col2:
            if st.button("❌ Exclusions"):
                st.session_state.questions = [
                    "What is excluded from coverage?",
                    "Are there age restrictions?",
                    "What activities void coverage?"
                ]
                st.rerun()
        with col3:
            if st.button("⏰ Timing"):
                st.session_state.questions = [
                    "What is the waiting period?",
                    "When does coverage start?",
                    "What are the claim time limits?"
                ]
                st.rerun()
    
    # Dynamic question input
    questions = []
    for i in range(len(st.session_state.questions)):
        question = st.text_input(
            f"Question {i+1}",
            value=st.session_state.questions[i],
            key=f"q_{i}"
        )
        if question.strip():
            questions.append(question.strip())
    
    # Add/Remove buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add Question"):
            st.session_state.questions.append("")
            st.rerun()
    with col2:
        if len(st.session_state.questions) > 1 and st.button("➖ Remove Last"):
            st.session_state.questions.pop()
            st.rerun()
    
    # Process button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Analyze Document", type="primary", use_container_width=True):
            
            if not document_url or not questions:
                st.error("❌ Please provide document URL and at least one question")
            
            elif not st.session_state.query_engine.llm:
                st.error("❌ LLM not initialized. Please check your Groq API key.")
            
            else:
                # Processing with progress
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    steps = [
                        "📥 Downloading document...",
                        "📄 Parsing content...",
                        "🔍 Setting up retrieval...",
                        "🧠 Processing questions...",
                        "✅ Finalizing results..."
                    ]
                    
                    async def run_analysis():
                        for i, step in enumerate(steps[:-1]):
                            status_text.text(step)
                            progress_bar.progress((i + 1) / len(steps))
                            await asyncio.sleep(0.1)
                        
                        # Actual processing
                        status_text.text("🤖 Running AI analysis...")
                        result = await st.session_state.query_engine.process_query(document_url, questions)
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Analysis complete!")
                        return result
                    
                    # Run async processing
                    try:
                        result = asyncio.run(run_analysis())
                        
                        # Clear progress indicators
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display results
                        if 'error' in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            st.success("🎉 Analysis completed!")
                            
                            # Metrics
                            answers = result.get('answers', [])
                            processing_time = result.get('processing_time', 0)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("⏱️ Time", f"{processing_time:.1f}s")
                            with col2:
                                st.metric("❓ Questions", len(answers))
                            with col3:
                                avg_conf = sum(a['confidence'] for a in answers) / len(answers) if answers else 0
                                st.metric("📊 Avg Confidence", f"{avg_conf:.0%}")
                            
                            st.markdown("---")
                            
                            # Display answers
                            st.header("📋 Results")
                            
                            for i, answer in enumerate(answers):
                                confidence = answer['confidence']
                                conf_class = "high" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "low"
                                
                                with st.expander(f"❓ Question {i+1}: {answer['question'][:80]}...", expanded=True):
                                    
                                    # Confidence gauge
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.markdown(f"**Q:** {answer['question']}")
                                    
                                    with col2:
                                        # Simple confidence indicator
                                        conf_color = "#28a745" if confidence >= 0.8 else "#ffc107" if confidence >= 0.5 else "#dc3545"
                                        st.markdown(f"<div style='text-align: right; color: {conf_color}; font-weight: bold; font-size: 1.2rem'>{confidence:.0%} Confidence</div>", unsafe_allow_html=True)
                                    
                                    # Answer
                                    st.markdown("### 📝 Answer")
                                    st.markdown(f'<div class="answer-card confidence-{conf_class}">{answer["answer"]}</div>', unsafe_allow_html=True)
                                    
                                    # Details tabs
                                    tab1, tab2, tab3 = st.tabs(["📊 Sources", "🔍 Analysis", "⚠️ Conflicts"])
                                    
                                    with tab1:
                                        if answer['source_clauses']:
                                            for j, clause in enumerate(answer['source_clauses'], 1):
                                                st.markdown(f"{j}. {clause}")
                                        else:
                                            st.info("No sources identified")
                                    
                                    with tab2:
                                        st.code(answer['justification'])
                                    
                                    with tab3:
                                        conflicts = answer['conflicts_detected']
                                        if conflicts:
                                            st.warning(f"⚠️ {len(conflicts)} conflicts detected:")
                                            for conflict in conflicts:
                                                st.markdown(f"• {conflict}")
                                        else:
                                            st.success("✅ No conflicts detected")
                            
                            # Audit trail
                            with st.expander("🔍 Processing Steps"):
                                for step in result.get('audit_trail', []):
                                    st.text(step)
                            
                            # Export results
                            st.markdown("---")
                            st.subheader("💾 Export Results")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                json_data = json.dumps(result, indent=2, default=str)
                                st.download_button(
                                    "📄 Download JSON",
                                    json_data,
                                    f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    "application/json"
                                )
                            
                            with col2:
                                if answers:
                                    df_data = [{
                                        'Question': ans['question'],
                                        'Answer': ans['answer'],
                                        'Confidence': ans['confidence'],
                                        'Sources': '; '.join(ans['source_clauses'])
                                    } for ans in answers]
                                    
                                    df = pd.DataFrame(df_data)
                                    csv_data = df.to_csv(index=False)
                                    
                                    st.download_button(
                                        "📊 Download CSV",
                                        csv_data,
                                        f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        "text/csv"
                                    )
                    
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        
                        # Troubleshooting
                        with st.expander("🔧 Troubleshooting"):
                            st.markdown("""
                            **Common fixes:**
                            1. Check your Groq API key in Streamlit secrets
                            2. Ensure document URL is publicly accessible
                            3. Try with fewer questions
                            4. Check document format (PDF/DOCX/EML only)
                            5. Verify internet connection
                            """)
    
    # Sidebar with system info
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Check if LLM is ready
        if st.session_state.query_engine.llm:
            st.success("✅ Groq LLM Ready")
            st.info(f"🤖 Model: {Config.LLM_MODEL_NAME}")
        else:
            st.error("❌ LLM Not Ready")
            st.warning("Check Groq API key")
        
        st.markdown("---")
        
        # Configuration
        st.header("⚙️ Configuration")
        
        # Model selection
        selected_model = st.selectbox(
            "🧠 LLM Model",
            Config.LLM_MODEL_OPTIONS,
            index=0,
            help="Choose the Groq model to use"
        )
        
        if selected_model != Config.LLM_MODEL_NAME:
            Config.LLM_MODEL_NAME = selected_model
            if st.session_state.query_engine.llm:
                st.session_state.query_engine.llm.model_name = selected_model
            st.success(f"✅ Switched to {selected_model}")
        
        # Processing settings
        st.markdown("**Processing Settings:**")
        chunk_size = st.slider("Chunk Size", 256, 1024, Config.CHUNK_SIZE)
        top_k = st.slider("Retrieval K", 3, 10, Config.TOP_K_RETRIEVAL)
        temperature = st.slider("Temperature", 0.0, 1.0, Config.TEMPERATURE, 0.1)
        
        # Update config
        Config.CHUNK_SIZE = chunk_size
        Config.TOP_K_RETRIEVAL = top_k
        Config.TEMPERATURE = temperature
        
        st.markdown("---")
        
        # API Info
        st.header("📊 Model Info")
        st.json({
            "provider": "Groq API",
            "current_model": Config.LLM_MODEL_NAME,
            "embedding_model": Config.EMBEDDING_MODEL,
            "features": ["Hybrid Retrieval", "Knowledge Graph", "Conflict Detection"]
        })
        
        st.markdown("---")
        
        # Help section
        with st.expander("📚 Quick Help"):
            st.markdown("""
            **Setup:**
            1. Set GROQ_API_KEY in Streamlit secrets
            2. Provide publicly accessible document URL
            3. Ask specific questions
            4. Review results with confidence scores
            
            **Tips:**
            - Be specific in questions
            - Check confidence scores
            - Review source clauses
            - Look for conflicts
            """)

# ========================
# 🏃‍♂️ UNIFIED ENTRY POINT
# ========================

def main():
    """Unified entry point - detects mode and runs appropriate service"""
    
    if IS_STREAMLIT:
        # Streamlit mode - UI is already rendered above
        pass
    else:
        # FastAPI mode - start the server
        port = int(os.environ.get("PORT", 8000))
        
        logger.info("🧠🔥 AI DOCUMENT QUERY SYSTEM")
        logger.info("=" * 50)
        logger.info(f"🚀 Starting FastAPI server on port {port}")
        logger.info(f"🤖 Using Groq API with: {Config.LLM_MODEL_NAME}")
        logger.info(f"🔍 Hybrid retrieval enabled")
        logger.info(f"🕸️ Knowledge graph enabled")
        logger.info("=" * 50)
        
        uvicorn.run(
            "unified_app:app",  # This file
            host="0.0.0.0",
            port=port,
            log_level="info"
        )

if __name__ == "__main__":
    main()
