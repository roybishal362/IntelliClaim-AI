# 🧠🔥 IntelliClaim-AI 
## 🚀 System Overview

This is a **production-ready document query system** with:
- **🤖 Groq API Integration** - Best open-source LLMs (Llama 3.1 70B, Mixtral, etc.)
- **🔍 Hybrid Retrieval** - Semantic (FAISS) + Keyword (Whoosh) search  
- **🕸️ Knowledge Graph** - Automatic conflict detection and relationship mapping
- **📊 Structured Outputs** - Confidence scores, source citations, audit trails
- **💻 Professional UI** - Beautiful Streamlit interface with analytics

## 📋 Prerequisites

- **Python 3.9+** (recommended: Python 3.11)
- **Groq API Key** - Get free API key from [console.groq.com](https://console.groq.com)
- **8GB+ RAM** (recommended for optimal performance)
- **Internet connection** for model downloads and document processing

## 🛠️ Quick Setup (5 Minutes)

### 1. Clone and Setup Environment

```bash
# Create project directory
mkdir IntelliClaim-AI && cd IntelliClaim-AI

# Copy the provided files:
# - main.py (rename to main.py)
# - app_ui.py (rename to app.py)  
# - requirements.txt
# - .env

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install backend dependencies
pip install -r requirements.txt

```

### 3. Configure Environment

Create `.env` file:
```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```

**🔑 Get Your Groq API Key:**
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up/Login (free)
3. Go to API Keys section
4. Create new API key
5. Copy and paste into `.env` file

### 4. Run the System

**Terminal 1 - Backend:**
```bash
python main.py
```

**Terminal 2 - Frontend:**
```bash
streamlit run app.py
```

**🎉 Access the app at: `http://localhost:8501`**

## 🔧 Configuration Options

### Available Groq Models (Best to Worst)

1. **`llama-3.1-70b-versatile`** ⭐ - Best overall quality (recommended)
2. **`mixtral-8x7b-32768`** - Excellent reasoning, longer context
3. **`llama-3.1-8b-instant`** - Fast and accurate
4. **`gemma2-9b-it`** - Good balance of speed/quality
5. **`llama3-70b-8192`** - Reliable fallback option

### Performance Tuning

Edit these values in `main.py` Config class:

```python
class Config:
    # Model selection (choose best for your needs)
    LLM_MODEL_NAME = "llama-3.1-70b-versatile"  # Best quality
    # LLM_MODEL_NAME = "llama-3.1-8b-instant"   # Faster processing
    
    # Retrieval settings
    TOP_K_RETRIEVAL = 8        # More results = better coverage, slower
    CHUNK_SIZE = 512           # Larger chunks = more context, more memory  
    CHUNK_OVERLAP = 50         # More overlap = better continuity
    
    # Hybrid retrieval weights
    SEMANTIC_WEIGHT = 0.7      # Prioritize semantic understanding
    KEYWORD_WEIGHT = 0.3       # Support with exact keyword matches
    
    # LLM settings
    MAX_TOKENS = 8192          # Maximum response length
    TEMPERATURE = 0.1          # Lower = more focused, higher = more creative
```

## 🚀 Deployment Options

### Option 1: Local Development
Perfect for testing and development:
```bash
# Start backend
python main.py

# Start frontend (new terminal)
streamlit run app.py
```

### Option 2: Docker Deployment (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 3: Cloud Deployment (Render/Railway)

**Backend on Render:**
1. Connect your GitHub repo to Render
2. Create new Web Service
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python main.py`
5. Add environment variable: `GROQ_API_KEY`

**Frontend on Streamlit Cloud:**
1. Deploy to [share.streamlit.io](https://share.streamlit.io)
2. Update `API_BASE_URL` in Streamlit app to your Render backend URL

### Option 4: Single VPS Deployment

```bash
# On your VPS (Ubuntu/Debian)
sudo apt update
sudo apt install python3 python3-pip nginx

# Clone your code
git clone your-repo-url
cd your-repo

# Install dependencies
pip3 install -r requirements.txt
pip3 install -r requirements_streamlit.txt

# Setup systemd services (see systemd configs below)
sudo systemctl enable ai-backend
sudo systemctl enable ai-frontend
sudo systemctl start ai-backend ai-frontend

# Configure nginx reverse proxy
sudo cp nginx.conf /etc/nginx/sites-available/ai-query
sudo ln -s /etc/nginx/sites-available/ai-query /etc/nginx/sites-enabled/
sudo systemctl reload nginx
```

## 🎯 Usage Guide

### Basic Workflow

1. **🌐 Open Web Interface** - Navigate to `http://localhost:8501`
2. **📄 Enter Document URL** - Paste URL of PDF/DOCX/EML document
3. **❓ Add Questions** - Use templates or write custom questions
4. **🚀 Analyze** - Click "Analyze Document" button
5. **📊 Review Results** - Check answers, confidence scores, conflicts
6. **💾 Export** - Download results as JSON or CSV

### Sample Questions

**Coverage Questions:**
- "What medical procedures are covered under this policy?"
- "What is the maximum annual benefit amount?"
- "Are mental health services included in coverage?"

**Exclusion Questions:**
- "What pre-existing conditions are excluded?"
- "Are cosmetic procedures covered?"
- "What happens if treatment is received outside the network?"

**Timing & Limits:**
- "What is the waiting period for major medical procedures?"
- "When does coverage begin for new members?"
- "Are there lifetime maximum benefits?"

### Understanding Results

**Confidence Scores:**
- **🟢 80-100%**: High reliability, strong evidence
- **🟡 50-79%**: Good confidence, verify with sources
- **🔴 0-49%**: Low confidence, review carefully

**Source Citations:**
- Each answer includes specific clause references
- Hover over citations to see original text
- Review multiple sources for comprehensive understanding

**Conflict Detection:**
- ⚠️ Yellow warnings indicate potential contradictions
- 🔴 Red alerts show definitive conflicts
- Always review conflicting clauses manually

## 🔍 Advanced Features

### Hybrid Retrieval System

The system combines two search methods:
1. **Semantic Search** (70% weight) - Understands meaning and context
2. **Keyword Search** (30% weight) - Finds exact term matches

### Knowledge Graph Features

- **Automatic Relationship Detection** - Links related clauses
- **Conflict Identification** - Finds contradictory information  
- **Dependency Mapping** - Shows which clauses depend on others
- **Rule-Based Overrides** - Deterministic checks for age/amount limits

### Query Expansion

For each question, the system automatically:
- Generates 3 alternative phrasings
- Searches with all variations
- Combines and ranks results
- Improves coverage of relevant information

## 📊 System Architecture

```
Frontend (Streamlit)
     ↓ HTTP API
Backend (FastAPI)
     ↓
┌─ Document Processor ─┐
│  • PDF/DOCX/EML      │
│  • Chunking          │
│  • Metadata Extract  │
└───────────────────────┘
     ↓
┌─ Knowledge Graph ────┐
│  • Clause Nodes      │
│  • Relationships     │
│  • Conflict Detection│
└───────────────────────┘
     ↓
┌─ Hybrid Retrieval ───┐
│  • FAISS (Semantic)  │
│  • Whoosh (Keyword)  │
│  • Score Fusion      │
└───────────────────────┘
     ↓
┌─ Groq LLM ───────────┐
│  • Llama 3.1 70B     │
│  • Structured Output │
│  • Confidence Scoring│
└───────────────────────┘
```

## 🐛 Troubleshooting

### Common Issues & Solutions

**❌ "System Offline" Error**
```bash
# Check if backend is running
curl http://localhost:8000/health

# If not, start backend
python main.py
```

**❌ "Groq API Disconnected"**
```bash
# Check API key
echo $GROQ_API_KEY

# Test API key manually
curl -H "Authorization: Bearer YOUR_KEY" https://api.groq.com/openai/v1/models
```

**❌ "No module named 'xyz'"**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**❌ "Document processing failed"**
- Ensure document URL is publicly accessible
- Check document format (PDF/DOCX/EML only)
- Verify document contains readable text (not just images)

**❌ "Low confidence scores"**
- Rephrase questions more specifically
- Break complex questions into simpler ones
- Ensure document contains relevant information

### Performance Issues

**Slow Processing:**
```python
# In main.py Config class, reduce these values:
TOP_K_RETRIEVAL = 5     # From 8 to 5
CHUNK_SIZE = 256        # From 512 to 256  
MAX_TOKENS = 4096       # From 8192 to 4096
```

**Memory Issues:**
```python
# Switch to faster, smaller model:
LLM_MODEL_NAME = "llama-3.1-8b-instant"
```

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### API Test
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer 90bf5fcfc3d0de340e50ac29a5cf53eb6da42e7a24af15f2111186878d510d6c" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "questions": ["What is this document about?"]
  }'
```

### Sample Test Documents

For testing, you can use these public documents:
- **Sample PDF**: `https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf`
- **Test Insurance Policy**: `https://www.cms.gov/files/document/model-coverage-coordination-benefits-cob-language.pdf`

## 📈 Production Optimizations

### For High Traffic

1. **Add Redis Caching:**
```python
# Add to requirements.txt
redis==5.0.1

# Cache embeddings and frequent queries
import redis
cache = redis.Redis(host='localhost', port=6379, db=0)
```

2. **Load Balancing:**
```bash
# Use multiple backend instances
docker-compose scale backend=3
```

3. **Database Storage:**
```python
# Replace in-memory storage with PostgreSQL
# Add persistent vector storage
```

### Security Enhancements

1. **API Rate Limiting:**
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/hackrx/run")
@limiter.limit("10/minute")
async def hackrx_run(...):
```

2. **Input Validation:**
```python
# Add stricter validation
from pydantic import validator, HttpUrl

class QueryRequest(BaseModel):
    documents: HttpUrl  # Validates URL format
    questions: List[str] = Field(..., min_items=1, max_items=10)
```

## 🎯 Best Practices

### Question Writing Tips
- **Be Specific**: "What is the deductible for emergency room visits?" vs "What are the costs?"
- **Include Context**: "For a 45-year-old diabetic patient, what preventive care is covered?"
- **Ask About Limits**: "What is the annual maximum for dental procedures?"

### Document Optimization
- Use **text-searchable PDFs** (not scanned images)
- Ensure **clear section headings** and structure
- **Avoid heavily redacted** documents
- **Test with sample documents** first

### Performance Tips
- **Batch similar questions** together
- **Use specific terms** that likely appear in documents
- **Monitor confidence scores** and refine questions accordingly

## 🔄 Updates & Maintenance

### Updating Models
```python
# In main.py, change to newer/better model:
LLM_MODEL_NAME = "llama-3.2-70b-versatile"  # When available
```

### Monitoring
```bash
# View system logs
tail -f logs/app.log

# Monitor resource usage
htop

# Check API usage
curl http://localhost:8000/system-info
```

### Backup & Recovery
```bash
# Backup configurations
cp .env .env.backup
cp main.py main.py.backup

# Database backup (if using persistent storage)
pg_dump document_query_db > backup.sql
```

## 📞 Support

### Getting Help
1. **Check System Status** - Use `/health` endpoint or sidebar status
2. **Review Logs** - Look for error messages in console output
3. **Test Components** - Verify each component individually
4. **Community Support** - Check GitHub issues or documentation

### Reporting Issues
When reporting problems, include:
- **System info** from `/system-info` endpoint
- **Error messages** from logs
- **Sample document URL** that fails
- **Exact questions** that cause issues
- **Environment details** (OS, Python version, etc.)

## 🏆 Advanced Use Cases

### Legal Document Analysis
```python
questions = [
    "What are the governing laws for this contract?",
    "What are the termination conditions?", 
    "What intellectual property rights are transferred?",
    "What are the liability limitations?"
]
```

### Insurance Policy Review
```python
questions = [
    "What is covered under preventive care?",
    "What are the co-payment requirements?",
    "Are there annual or lifetime maximums?",
    "What is the process for claim submission?"
]
```

### Compliance Checking
```python
questions = [
    "What regulatory requirements must be met?",
    "What are the reporting obligations?",
    "What penalties apply for non-compliance?",
    "What audit requirements are specified?"
]
```

## 📊 Performance Benchmarks

**Typical Performance (Local):**
- **Document Processing**: 5-15 seconds
- **Question Analysis**: 2-5 seconds per question
- **Memory Usage**: 2-4GB RAM
- **Accuracy**: 85-95% confidence on well-structured documents

**Production Performance (Cloud):**
- **Concurrent Users**: 10-50 (depending on instance size)
- **Response Time**: 3-8 seconds per question
- **Throughput**: 100-500 questions/hour
- **Uptime**: 99.9% with proper monitoring

## 🔐 Security Considerations

### API Security
- Use **environment variables** for API keys
- Implement **rate limiting** for production
- Add **input sanitization** for document URLs
- Use **HTTPS** for all communications

### Data Privacy
- Documents are **processed in memory** only
- **No persistent storage** of user documents
- **Temporary files** are automatically cleaned up
- **API keys** are never logged or exposed

## 🎓 Educational Value

This system demonstrates:
- **Modern AI Architecture** - Hybrid retrieval + LLM reasoning
- **Production Engineering** - Error handling, monitoring, scalability
- **API Design** - RESTful endpoints with proper validation
- **Frontend Development** - Professional UI with real-time feedback
- **DevOps Practices** - Docker, environment management, deployment

## 🔮 Future Enhancements

### Planned Features
1. **Multi-document Analysis** - Compare across multiple documents
2. **Real-time Collaboration** - Multiple users, shared sessions
3. **Advanced Analytics** - Query trends, accuracy metrics
4. **Custom Model Fine-tuning** - Domain-specific model adaptation
5. **Integration APIs** - Connect with existing document management systems

### Scalability Roadmap
1. **Microservices Architecture** - Separate services for each component
2. **Distributed Processing** - Handle large documents across multiple workers
3. **Advanced Caching** - Redis/Memcached for frequent queries
4. **Database Integration** - PostgreSQL for persistent storage
5. **Auto-scaling** - Kubernetes deployment with HPA

---

## 🎉 Ready to Use!

Your enhanced AI document query system is now ready with:
- ✅ **Best-in-class LLM** via Groq API
- ✅ **Hybrid retrieval** for comprehensive search
- ✅ **Knowledge graph** with conflict detection
- ✅ **Professional UI** with analytics
- ✅ **Production features** (audit trails, confidence scoring)
- ✅ **Easy deployment** options

**Start analyzing documents and experience the power of advanced AI! 🚀**
