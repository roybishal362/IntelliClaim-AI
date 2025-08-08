"""
🧠🔥 ENHANCED LLM-POWERED DOCUMENT QUERY RETRIEVAL SYSTEM
Hybrid Retrieval + Knowledge Graph + Groq API Integration
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
from datetime import datetime
import tempfile
import gc
import re
from dataclasses import dataclass
import networkx as nx

# FastAPI and web components
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
import requests

# LangChain core components
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredEmailLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings  import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Groq API for LLM inference
from groq import Groq

# Hybrid retrieval components
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh.writing import AsyncWriter
import whoosh.index as index

# Vector search and utilities
import numpy as np
import faiss

import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# 🔧 ENHANCED CONFIGURATION
# ========================

class Config:
    # Groq API - Best open-source models
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
    
    # Best open-source models available on Groq (in order of preference)
    LLM_MODEL_OPTIONS = [
        "llama-3.1-70b-versatile",      # Meta's Llama 3.1 70B - Best overall
        "llama-3.1-8b-instant",        # Llama 3.1 8B - Fast and accurate
        "mixtral-8x7b-32768",          # Mixtral 8x7B - Excellent reasoning
        "gemma2-9b-it",                # Google's Gemma2 9B - Good balance
        "llama3-70b-8192",             # Llama 3 70B - Reliable fallback
    ]
    
    # Use the best available model
    LLM_MODEL_NAME = LLM_MODEL_OPTIONS[0]
    
    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Processing config
    MAX_TOKENS = 8192
    TEMPERATURE = 0.1
    TOP_P = 0.9
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K_RETRIEVAL = 8
    
    # Hybrid retrieval weights
    SEMANTIC_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3
    
    # API
    HACKRX_TOKEN = "90bf5fcfc3d0de340e50ac29a5cf53eb6da42e7a24af15f2111186878d510d6c"
    MAX_QUESTIONS = 10
    CONFIDENCE_THRESHOLD = 0.7

# ========================
# 📊 ENHANCED DATA MODELS
# ========================

@dataclass
class ClauseNode:
    """Knowledge graph node for a clause"""
    clause_id: str
    content: str
    clause_type: str  # coverage, exclusion, condition, benefit, etc.
    age_limit: Optional[int] = None
    waiting_period: Optional[int] = None
    amount_limit: Optional[float] = None
    dependencies: List[str] = None
    conflicts: List[str] = None

class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to document (PDF, DOCX, EML)")
    questions: List[str] = Field(..., description="List of questions to answer")

class AnswerItem(BaseModel):
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Structured answer")
    confidence: float = Field(..., description="Confidence score 0-1")
    source_clauses: List[str] = Field(..., description="Source clause references")
    justification: str = Field(..., description="Reasoning explanation")
    conflicts_detected: List[str] = Field(default=[], description="Any conflicting clauses found")

class QueryResponse(BaseModel):
    answers: List[AnswerItem] = Field(..., description="Structured answers with metadata")
    processing_time: float = Field(..., description="Total processing time in seconds")
    audit_trail: List[str] = Field(..., description="Processing steps audit log")

class ParsedQuery(BaseModel):
    intent: str
    target: str
    conditions: List[str]
    original_question: str
    query_expansion: List[str] = []

class RetrievedClause(BaseModel):
    content: str
    metadata: Dict[str, Any]
    semantic_score: float
    keyword_score: float
    final_score: float
    source_info: str
    clause_id: str

# ========================
# 🤖 GROQ LLM MANAGER
# ========================

class GroqLLM:
    """Groq API wrapper for high-quality LLM inference"""
    
    def __init__(self):
        if not Config.GROQ_API_KEY or Config.GROQ_API_KEY == "your_groq_api_key_here":
            raise ValueError("Please set GROQ_API_KEY environment variable")
        
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model_name = Config.LLM_MODEL_NAME
        logger.info(f"🚀 Initialized Groq LLM with model: {self.model_name}")
    
    async def generate_response(self, prompt: str, max_tokens: int = 1024, temperature: float = None) -> str:
        """Generate response using Groq API"""
        try:
            temp = temperature if temperature is not None else Config.TEMPERATURE
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert legal and insurance document analyst. Provide accurate, detailed, and well-structured responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temp,
                top_p=Config.TOP_P,
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            # Try fallback model
            if self.model_name != Config.LLM_MODEL_OPTIONS[1]:
                logger.info(f"Trying fallback model: {Config.LLM_MODEL_OPTIONS[1]}")
                self.model_name = Config.LLM_MODEL_OPTIONS[1]
                return await self.generate_response(prompt, max_tokens, temperature)
            
            return f"I apologize, but I encountered an error processing your request: {str(e)}"

# ========================
# 🧾 ENHANCED DOCUMENT PROCESSOR
# ========================

class EnhancedDocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def download_document(self, url: str) -> str:
        """Download document from URL"""
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
                suffix = '.pdf'  # Default
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(response.content)
                return tmp_file.name
                
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    
    async def parse_document(self, file_path: str) -> Tuple[List[Document], List[ClauseNode]]:
        """Parse document and extract structured clauses"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # Select appropriate loader
            if file_ext == '.pdf':
                loader = UnstructuredPDFLoader(file_path)
            elif file_ext in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_ext == '.eml':
                loader = UnstructuredEmailLoader(file_path)
            else:
                loader = UnstructuredPDFLoader(file_path)
            
            documents = loader.load()
            
            # Add metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source_file': Path(file_path).name,
                    'chunk_id': i,
                    'processed_at': datetime.now().isoformat()
                })
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Create structured clause nodes
            clause_nodes = []
            
            for i, chunk in enumerate(chunks):
                clause_id = f"clause_{i+1}"
                chunk.metadata['clause_id'] = clause_id
                chunk.metadata['clause_no'] = i + 1
                chunk.metadata['section_title'] = self._extract_section_title(chunk.page_content)
                
                # Extract structured information
                clause_node = self._extract_clause_structure(chunk.page_content, clause_id)
                clause_nodes.append(clause_node)
            
            logger.info(f"📄 Parsed document into {len(chunks)} chunks with structured nodes")
            return chunks, clause_nodes
            
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to parse document: {e}")
    
    def _extract_section_title(self, text: str) -> str:
        """Extract section title from text"""
        lines = text.strip().split('\n')
        for line in lines[:3]:
            line = line.strip()
            if len(line) < 100 and any(keyword in line.lower() for keyword in 
                ['section', 'clause', 'article', 'chapter', 'coverage', 'exclusion', 'benefits']):
                return line
        return "General"
    
    def _extract_clause_structure(self, content: str, clause_id: str) -> ClauseNode:
        """Extract structured information from clause content"""
        content_lower = content.lower()
        
        # Determine clause type
        clause_type = "general"
        if any(word in content_lower for word in ['cover', 'benefit', 'include']):
            clause_type = "coverage"
        elif any(word in content_lower for word in ['exclude', 'not cover', 'except']):
            clause_type = "exclusion"
        elif any(word in content_lower for word in ['condition', 'requirement', 'must']):
            clause_type = "condition"
        elif any(word in content_lower for word in ['limit', 'maximum', 'up to']):
            clause_type = "limit"
        
        # Extract age limits
        age_limit = None
        age_patterns = [r'age\s+(\d+)', r'(\d+)\s+years?\s+old', r'above\s+(\d+)', r'under\s+(\d+)']
        for pattern in age_patterns:
            match = re.search(pattern, content_lower)
            if match:
                age_limit = int(match.group(1))
                break
        
        # Extract waiting periods
        waiting_period = None
        waiting_patterns = [r'(\d+)\s+days?\s+wait', r'waiting\s+period\s+(\d+)', r'(\d+)\s+months?\s+wait']
        for pattern in waiting_patterns:
            match = re.search(pattern, content_lower)
            if match:
                waiting_period = int(match.group(1))
                break
        
        # Extract amount limits
        amount_limit = None
        amount_patterns = [r'\$(\d+(?:,\d+)*)', r'(\d+(?:,\d+)*)\s+dollars?', r'maximum\s+(\d+(?:,\d+)*)']
        for pattern in amount_patterns:
            match = re.search(pattern, content_lower)
            if match:
                amount_str = match.group(1).replace(',', '')
                amount_limit = float(amount_str)
                break
        
        return ClauseNode(
            clause_id=clause_id,
            content=content,
            clause_type=clause_type,
            age_limit=age_limit,
            waiting_period=waiting_period,
            amount_limit=amount_limit,
            dependencies=[],
            conflicts=[]
        )

# ========================
# 🕸️ KNOWLEDGE GRAPH MANAGER
# ========================

class KnowledgeGraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.clause_nodes = {}
    
    def build_graph(self, clause_nodes: List[ClauseNode]):
        """Build knowledge graph from clause nodes"""
        self.clause_nodes = {node.clause_id: node for node in clause_nodes}
        
        # Add nodes to graph
        for node in clause_nodes:
            self.graph.add_node(node.clause_id, **node.__dict__)
        
        # Add relationships
        self._detect_relationships(clause_nodes)
    
    def _detect_relationships(self, clause_nodes: List[ClauseNode]):
        """Detect relationships between clauses"""
        for i, node1 in enumerate(clause_nodes):
            for j, node2 in enumerate(clause_nodes):
                if i == j:
                    continue
                
                # Detect conflicts (exclusion vs coverage)
                if self._detect_conflict(node1, node2):
                    self.graph.add_edge(node1.clause_id, node2.clause_id, relation="conflicts")
                    node1.conflicts.append(node2.clause_id)
                
                # Detect dependencies
                if self._detect_dependency(node1, node2):
                    self.graph.add_edge(node1.clause_id, node2.clause_id, relation="depends_on")
                    node1.dependencies.append(node2.clause_id)
    
    def _detect_conflict(self, node1: ClauseNode, node2: ClauseNode) -> bool:
        """Detect if two clauses conflict"""
        if node1.clause_type == "coverage" and node2.clause_type == "exclusion":
            # Check for overlapping content
            words1 = set(node1.content.lower().split())
            words2 = set(node2.content.lower().split())
            overlap = len(words1.intersection(words2))
            return overlap > 5  # Threshold for conflict detection
        return False
    
    def _detect_dependency(self, node1: ClauseNode, node2: ClauseNode) -> bool:
        """Detect if one clause depends on another"""
        if node1.clause_type == "condition" and node2.clause_type == "coverage":
            # Simple dependency detection based on content similarity
            words1 = set(node1.content.lower().split())
            words2 = set(node2.content.lower().split())
            overlap = len(words1.intersection(words2))
            return overlap > 3
        return False
    
    def get_related_clauses(self, clause_id: str, relation_type: str = None) -> List[str]:
        """Get clauses related to given clause"""
        if clause_id not in self.graph:
            return []
        
        if relation_type:
            return [neighbor for neighbor in self.graph.neighbors(clause_id) 
                   if self.graph[clause_id][neighbor].get('relation') == relation_type]
        else:
            return list(self.graph.neighbors(clause_id))

# ========================
# 🔎 HYBRID RETRIEVAL MANAGER
# ========================

class HybridRetrievalManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.keyword_index = None
        self.documents = None
    
    async def setup_retrievers(self, documents: List[Document]) -> FAISS:
        """Setup both semantic and keyword retrievers"""
        self.documents = documents
        
        # Setup semantic retrieval (FAISS)
        logger.info("🔍 Creating semantic embeddings...")
        self.vectorstore = await asyncio.get_event_loop().run_in_executor(
            None, FAISS.from_documents, documents, self.embeddings
        )
        
        # Setup keyword retrieval (Whoosh)
        logger.info("🔤 Creating keyword index...")
        await self._setup_keyword_index(documents)
        
        logger.info(f"✅ Hybrid retrieval setup complete with {len(documents)} documents")
        return self.vectorstore
    
    async def _setup_keyword_index(self, documents: List[Document]):
        """Setup Whoosh keyword index"""
        try:
            # Create schema
            schema = Schema(
                clause_id=ID(stored=True),
                content=TEXT(stored=True),
                title=TEXT(stored=True)
            )
            
            # Create index in memory
            import tempfile
            index_dir = tempfile.mkdtemp()
            self.keyword_index = create_in(schema, index_dir)
            
            # Add documents to index
            writer = self.keyword_index.writer()
            for doc in documents:
                writer.add_document(
                    clause_id=doc.metadata.get('clause_id', ''),
                    content=doc.page_content,
                    title=doc.metadata.get('section_title', '')
                )
            writer.commit()
            
        except Exception as e:
            logger.warning(f"Keyword index setup failed: {e}")
            self.keyword_index = None
    
    def hybrid_retrieve(self, query: str, top_k: int = Config.TOP_K_RETRIEVAL) -> List[RetrievedClause]:
        """Perform hybrid retrieval combining semantic and keyword search"""
        semantic_results = self._semantic_search(query, top_k)
        keyword_results = self._keyword_search(query, top_k) if self.keyword_index else []
        
        # Combine and re-rank results
        combined_results = self._combine_results(semantic_results, keyword_results)
        
        return combined_results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[Document, float, str]]:
        """Perform semantic search using FAISS"""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            return [(doc, score, "semantic") for doc, score in results]
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[Document, float, str]]:
        """Perform keyword search using Whoosh"""
        try:
            if not self.keyword_index:
                return []
            
            with self.keyword_index.searcher() as searcher:
                parser = QueryParser("content", self.keyword_index.schema)
                parsed_query = parser.parse(query)
                results = searcher.search(parsed_query, limit=top_k)
                
                keyword_results = []
                for result in results:
                    clause_id = result['clause_id']
                    # Find corresponding document
                    doc = next((d for d in self.documents if d.metadata.get('clause_id') == clause_id), None)
                    if doc:
                        # Convert Whoosh score to distance-like score
                        score = 1.0 - min(result.score / 10.0, 1.0)  # Normalize
                        keyword_results.append((doc, score, "keyword"))
                
                return keyword_results
                
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
    
    def _combine_results(self, semantic_results: List[Tuple[Document, float, str]], 
                        keyword_results: List[Tuple[Document, float, str]]) -> List[RetrievedClause]:
        """Combine and re-rank semantic and keyword results"""
        
        # Create mapping of clause_id to results
        result_map = {}
        
        # Process semantic results
        for doc, score, source in semantic_results:
            clause_id = doc.metadata.get('clause_id', '')
            if clause_id not in result_map:
                result_map[clause_id] = {
                    'doc': doc,
                    'semantic_score': 1.0 - (score / 2.0),  # Normalize to 0-1
                    'keyword_score': 0.0
                }
            else:
                result_map[clause_id]['semantic_score'] = max(
                    result_map[clause_id]['semantic_score'], 
                    1.0 - (score / 2.0)
                )
        
        # Process keyword results
        for doc, score, source in keyword_results:
            clause_id = doc.metadata.get('clause_id', '')
            if clause_id not in result_map:
                result_map[clause_id] = {
                    'doc': doc,
                    'semantic_score': 0.0,
                    'keyword_score': 1.0 - score
                }
            else:
                result_map[clause_id]['keyword_score'] = max(
                    result_map[clause_id]['keyword_score'], 
                    1.0 - score
                )
        
        # Calculate final scores and create RetrievedClause objects
        retrieved_clauses = []
        for clause_id, result_data in result_map.items():
            doc = result_data['doc']
            semantic_score = result_data['semantic_score']
            keyword_score = result_data['keyword_score']
            
            # Weighted combination
            final_score = (Config.SEMANTIC_WEIGHT * semantic_score + 
                          Config.KEYWORD_WEIGHT * keyword_score)
            
            clause = RetrievedClause(
                content=doc.page_content,
                metadata=doc.metadata,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                final_score=final_score,
                source_info=f"Clause {doc.metadata.get('clause_no', 'N/A')}",
                clause_id=clause_id
            )
            retrieved_clauses.append(clause)
        
        # Sort by final score
        retrieved_clauses.sort(key=lambda x: x.final_score, reverse=True)
        return retrieved_clauses

# ========================
# 🧠 ENHANCED QUERY PROCESSOR
# ========================

class EnhancedQueryProcessor:
    def __init__(self, llm: GroqLLM):
        self.llm = llm
    
    async def parse_query(self, question: str) -> ParsedQuery:
        """Parse and expand natural language question"""
        try:
            prompt = f"""Analyze this insurance/legal document question and extract structured information.

Question: "{question}"

Provide JSON response with:
- intent: (coverage_check, exclusion_check, limit_check, waiting_period, age_requirement, general_query)
- target: (main subject being asked about)
- conditions: (list of specific conditions, ages, amounts mentioned)
- query_expansion: (3 alternative ways to phrase this question for better search)

JSON:"""
            
            response = await self.llm.generate_response(prompt, max_tokens=300)
            
            # Parse JSON response
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    parsed_data = json.loads(json_str)
                    
                    return ParsedQuery(
                        intent=parsed_data.get("intent", "general_query"),
                        target=parsed_data.get("target", question[:50]),
                        conditions=parsed_data.get("conditions", []),
                        original_question=question,
                        query_expansion=parsed_data.get("query_expansion", [])
                    )
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response from LLM")
            
            # Fallback parsing
            return ParsedQuery(
                intent="general_query",
                target=question[:50],
                conditions=[],
                original_question=question,
                query_expansion=[]
            )
            
        except Exception as e:
            logger.warning(f"Error parsing query: {e}")
            return ParsedQuery(
                intent="general_query",
                target=question[:50],
                conditions=[],
                original_question=question,
                query_expansion=[]
            )

# ========================
# 🎯 MAIN PROCESSING ENGINE
# ========================

class EnhancedQueryProcessingEngine:
    def __init__(self):
        logger.info("🚀 Initializing Enhanced Query Processing Engine...")
        
        # Initialize components
        self.llm = GroqLLM()
        self.doc_processor = EnhancedDocumentProcessor()
        self.retrieval_manager = HybridRetrievalManager()
        self.query_processor = EnhancedQueryProcessor(self.llm)
        self.kg_manager = KnowledgeGraphManager()
        
        logger.info("✅ Enhanced engine initialized successfully")
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Enhanced query processing pipeline"""
        start_time = datetime.now()
        audit_trail = []
        
        try:
            audit_trail.append(f"🚀 Started processing {len(request.questions)} questions")
            
            # Download and parse document
            audit_trail.append("📥 Downloading document...")
            file_path = await self.doc_processor.download_document(request.documents)
            
            audit_trail.append("📄 Parsing document and extracting structure...")
            documents, clause_nodes = await self.doc_processor.parse_document(file_path)
            
            # Build knowledge graph
            audit_trail.append("🕸️ Building knowledge graph...")
            self.kg_manager.build_graph(clause_nodes)
            
            # Setup hybrid retrieval
            audit_trail.append("🔍 Setting up hybrid retrieval system...")
            await self.retrieval_manager.setup_retrievers(documents)
            
            answers = []
            
            # Process each question
            for i, question in enumerate(request.questions, 1):
                try:
                    audit_trail.append(f"❓ Processing question {i}: {question[:50]}...")
                    
                    # Parse and expand query
                    parsed_query = await self.query_processor.parse_query(question)
                    audit_trail.append(f"🧠 Query parsed - Intent: {parsed_query.intent}")
                    
                    # Retrieve relevant clauses using hybrid search
                    retrieved_clauses = self._hybrid_retrieve_with_expansion(parsed_query)
                    audit_trail.append(f"🎯 Retrieved {len(retrieved_clauses)} relevant clauses")
                    
                    # Check for conflicts using knowledge graph
                    conflicts = self._detect_conflicts(retrieved_clauses)
                    if conflicts:
                        audit_trail.append(f"⚠️ Detected {len(conflicts)} potential conflicts")
                    
                    # Generate structured answer
                    answer_item = await self._generate_structured_answer(
                        parsed_query, retrieved_clauses, conflicts
                    )
                    answers.append(answer_item)
                    
                    audit_trail.append(f"✅ Question {i} processed with confidence: {answer_item.confidence:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error processing question {i}: {e}")
                    error_answer = AnswerItem(
                        question=question,
                        answer=f"I encountered an error while processing this question: {str(e)}",
                        confidence=0.0,
                        source_clauses=[],
                        justification="Error in processing",
                        conflicts_detected=[]
                    )
                    answers.append(error_answer)
                    audit_trail.append(f"❌ Error processing question {i}: {e}")
            
            # Cleanup
            try:
                os.unlink(file_path)
            except:
                pass
            
            processing_time = (datetime.now() - start_time).total_seconds()
            audit_trail.append(f"🏁 Processing completed in {processing_time:.2f} seconds")
            
            return QueryResponse(
                answers=answers,
                processing_time=processing_time,
                audit_trail=audit_trail
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced query processing: {e}")
            raise HTTPException(status_code=500, detail=f"Enhanced query processing failed: {e}")
    
    def _hybrid_retrieve_with_expansion(self, query: ParsedQuery) -> List[RetrievedClause]:
        """Retrieve clauses using hybrid search with query expansion"""
        all_results = []
        
        # Search with original query
        original_results = self.retrieval_manager.hybrid_retrieve(query.original_question)
        all_results.extend(original_results)
        
        # Search with expanded queries
        for expanded_query in query.query_expansion:
            expanded_results = self.retrieval_manager.hybrid_retrieve(expanded_query, top_k=3)
            all_results.extend(expanded_results)
        
        # Remove duplicates and re-rank
        seen_clauses = set()
        unique_results = []
        for result in all_results:
            if result.clause_id not in seen_clauses:
                seen_clauses.add(result.clause_id)
                unique_results.append(result)
        
        # Sort by final score
        unique_results.sort(key=lambda x: x.final_score, reverse=True)
        return unique_results[:Config.TOP_K_RETRIEVAL]
    
    def _detect_conflicts(self, retrieved_clauses: List[RetrievedClause]) -> List[str]:
        """Detect conflicts using knowledge graph"""
        conflicts = []
        for clause in retrieved_clauses:
            clause_conflicts = self.kg_manager.get_related_clauses(clause.clause_id, "conflicts")
            conflicts.extend(clause_conflicts)
        
        return list(set(conflicts))  # Remove duplicates
    
    async def _generate_structured_answer(self, query: ParsedQuery, clauses: List[RetrievedClause], 
                                         conflicts: List[str]) -> AnswerItem:
        """Generate structured answer with metadata"""
        try:
            if not clauses:
                return AnswerItem(
                    question=query.original_question,
                    answer="I couldn't find relevant information to answer this question in the provided document.",
                    confidence=0.0,
                    source_clauses=[],
                    justification="No relevant clauses found",
                    conflicts_detected=[]
                )
            
            # Prepare context from top clauses
            context = ""
            source_clauses = []
            
            for i, clause in enumerate(clauses[:4], 1):  # Top 4 clauses
                context += f"\n[Clause {clause.metadata.get('clause_no', i)}] {clause.content[:500]}...\n"
                source_clauses.append(f"Clause {clause.metadata.get('clause_no', i)} (Score: {clause.final_score:.2f})")
            
            # Add conflict information
            conflict_info = ""
            if conflicts:
                conflict_info = f"\n\nNOTE: Potential conflicts detected with clauses: {', '.join(conflicts)}"
            
            # Create enhanced prompt
            prompt = f"""You are an expert insurance and legal document analyst. Provide a comprehensive, structured answer based on the provided clauses.

Question: {query.original_question}
Query Intent: {query.intent}
Query Target: {query.target}
Conditions: {query.conditions}

Relevant Document Clauses:
{context}{conflict_info}

Instructions:
1. Provide a clear, factual answer based on the clauses
2. Include specific details like amounts, time periods, age limits, conditions
3. If coverage exists, state it clearly with any limitations
4. If exclusions apply, mention them explicitly
5. If conflicts exist between clauses, acknowledge them
6. Be comprehensive but concise
7. Structure your answer logically

Answer:"""
            
            # Generate response
            answer = await self.llm.generate_response(prompt, max_tokens=600)
            
            # Calculate confidence based on retrieval scores
            avg_score = sum(clause.final_score for clause in clauses[:3]) / min(3, len(clauses))
            confidence = min(0.95, max(0.1, avg_score))  # Cap between 0.1 and 0.95
            
            # Generate justification
            justification = self._generate_justification(query, clauses[:3])
            
            return AnswerItem(
                question=query.original_question,
                answer=answer.strip(),
                confidence=round(confidence, 2),
                source_clauses=source_clauses,
                justification=justification,
                conflicts_detected=conflicts
            )
            
        except Exception as e:
            logger.error(f"Error generating structured answer: {e}")
            return AnswerItem(
                question=query.original_question,
                answer=f"I apologize, but I encountered an error while analyzing this question: {str(e)}",
                confidence=0.0,
                source_clauses=[],
                justification="Error in answer generation",
                conflicts_detected=[]
            )
    
    def _generate_justification(self, query: ParsedQuery, top_clauses: List[RetrievedClause]) -> str:
        """Generate explanation of how the answer was derived"""
        justification_parts = []
        
        justification_parts.append(f"Query analyzed as '{query.intent}' targeting '{query.target}'")
        
        if query.conditions:
            justification_parts.append(f"Specific conditions considered: {', '.join(query.conditions)}")
        
        for i, clause in enumerate(top_clauses, 1):
            score_info = f"semantic: {clause.semantic_score:.2f}, keyword: {clause.keyword_score:.2f}"
            justification_parts.append(
                f"Clause {i} matched with combined score {clause.final_score:.2f} ({score_info})"
            )
        
        return "; ".join(justification_parts)

# ========================
# 🚀 FASTAPI APPLICATION
# ========================

app = FastAPI(
    title="🧠 Enhanced LLM Document Query System",
    description="Production-ready hybrid retrieval system with Groq API and knowledge graphs",
    version="3.0.0"
)

# Initialize global engine
query_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced query engine"""
    global query_engine
    try:
        logger.info("🚀 Starting enhanced application...")
        query_engine = EnhancedQueryProcessingEngine()
        logger.info("✅ Enhanced application ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize enhanced system: {e}")
        raise

def verify_token(authorization: str = Header(None)):
    """Verify authorization token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization[7:]
    if token != Config.HACKRX_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "🧠🔥 Enhanced LLM Document Query System",
        "version": "3.0.0", 
        "status": "ready",
        "features": [
            "🤖 Groq API integration with best open-source models",
            "🔍 Hybrid retrieval (Semantic + Keyword)",
            "🕸️ Knowledge graph with conflict detection", 
            "📊 Structured responses with confidence scores",
            "🔍 Query expansion for better coverage",
            "📋 Comprehensive audit trails"
        ],
        "model": Config.LLM_MODEL_NAME,
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/health",
            "info": "/system-info"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    try:
        # Test Groq API connection
        groq_status = "unknown"
        try:
            if query_engine and query_engine.llm:
                test_response = await query_engine.llm.generate_response("Test", max_tokens=10)
                groq_status = "connected" if test_response else "error"
        except:
            groq_status = "disconnected"
        
        return {
            "status": "healthy",
            "groq_api_status": groq_status,
            "model": Config.LLM_MODEL_NAME,
            "components": {
                "document_processor": "ready",
                "hybrid_retrieval": "ready", 
                "knowledge_graph": "ready",
                "llm": groq_status
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.get("/system-info")
async def system_info():
    """System configuration information"""
    return {
        "system": "Enhanced Document Query System",
        "llm_provider": "Groq API",
        "available_models": Config.LLM_MODEL_OPTIONS,
        "current_model": Config.LLM_MODEL_NAME,
        "embedding_model": Config.EMBEDDING_MODEL,
        "features": {
            "hybrid_retrieval": True,
            "knowledge_graph": True,
            "conflict_detection": True,
            "query_expansion": True,
            "structured_output": True,
            "audit_trail": True
        },
        "configuration": {
            "max_tokens": Config.MAX_TOKENS,
            "chunk_size": Config.CHUNK_SIZE,
            "top_k_retrieval": Config.TOP_K_RETRIEVAL,
            "semantic_weight": Config.SEMANTIC_WEIGHT,
            "keyword_weight": Config.KEYWORD_WEIGHT
        }
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Enhanced HackRx endpoint with full feature set"""
    
    # Validate input
    if not request.questions:
        raise HTTPException(status_code=400, detail="At least one question required")
    
    if len(request.questions) > Config.MAX_QUESTIONS:
        raise HTTPException(status_code=400, detail=f"Maximum {Config.MAX_QUESTIONS} questions allowed")
    
    if not query_engine:
        raise HTTPException(status_code=503, detail="Enhanced service not ready")
    
    # Process with enhanced pipeline
    return await query_engine.process_query(request)

# ========================
# 🏃‍♂️ MAIN ENTRY POINT
# ========================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment
    port = int(os.environ.get("PORT", 8000))
    
    logger.info("🧠🔥 ENHANCED DOCUMENT QUERY SYSTEM")
    logger.info("=" * 50)
    logger.info(f"🚀 Starting enhanced server on port {port}")
    logger.info(f"🤖 Using Groq API with model: {Config.LLM_MODEL_NAME}")
    logger.info(f"🔍 Hybrid retrieval enabled (Semantic + Keyword)")
    logger.info(f"🕸️ Knowledge graph with conflict detection")
    logger.info(f"📊 Structured outputs with confidence scoring")
    logger.info("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
            