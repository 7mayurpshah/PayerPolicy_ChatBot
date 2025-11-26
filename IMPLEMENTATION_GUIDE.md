# PayerPolicy_ChatBot Implementation Guide

## Overview

This guide provides a comprehensive, step-by-step roadmap for implementing the PayerPolicy_ChatBot RAG (Retrieval-Augmented Generation) application from start to finish. This is a privacy-focused, local document Q&A system that enables intelligent question-answering against large document collections without external API calls.

**Implementation Time Estimate:** 4-6 weeks for core functionality (based on 1-2 developers)

---

## Table of Contents

1. [Prerequisites & Setup](#1-prerequisites--setup)
2. [Phase 1: Project Foundation](#2-phase-1-project-foundation-week-1)
3. [Phase 2: Document Processing](#3-phase-2-document-processing-week-1-2)
4. [Phase 3: Vector Database & Embeddings](#4-phase-3-vector-database--embeddings-week-2)
5. [Phase 4: RAG Pipeline](#5-phase-4-rag-pipeline-week-3)
6. [Phase 5: API Layer](#6-phase-5-api-layer-week-3-4)
7. [Phase 6: Frontend](#7-phase-6-frontend-week-4)
8. [Phase 7: Authentication & Security](#8-phase-7-authentication--security-week-5)
9. [Phase 8: Testing & Quality Assurance](#9-phase-8-testing--quality-assurance-week-5-6)
10. [Phase 9: Deployment & Production](#10-phase-9-deployment--production-week-6)
11. [Verification & Troubleshooting](#11-verification--troubleshooting)
12. [Next Steps & Enhancements](#12-next-steps--enhancements)

---

## 1. Prerequisites & Setup

### 1.1 System Requirements

**Minimum:**
- OS: Ubuntu 20.04 LTS or later
- CPU: 8 cores (x86_64)
- RAM: 16 GB
- Storage: 100 GB SSD
- Python 3.10+

**Recommended:**
- OS: Ubuntu 22.04 LTS
- CPU: 16 cores
- RAM: 32 GB
- Storage: 500 GB NVMe SSD
- GPU: NVIDIA with 8GB+ VRAM (optional, for faster inference)

### 1.2 Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip -y

# Install system dependencies
sudo apt install build-essential libssl-dev libffi-dev python3-dev git curl -y

# Install Ollama
curl https://ollama.ai/install.sh | sh

# Verify Ollama installation
ollama --version

# Pull required models
ollama pull nomic-embed-text
ollama pull llama2  # or mistral

# Verify models
ollama list
```

### 1.3 Create Project Structure

```bash
# Navigate to your workspace
cd /opt/rag-app  # or your preferred location

# Clone repository (if not already cloned)
git clone https://github.com/spsanderson/PayerPolicy_ChatBot.git
cd PayerPolicy_ChatBot

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## 2. Phase 1: Project Foundation (Week 1)

### 2.1 Create Project Structure

Create the following directory structure:

```bash
mkdir -p src/{document_processor,embeddings,vector_store,llm,api,utils}
mkdir -p config
mkdir -p static/{css,js,images}
mkdir -p templates
mkdir -p data/{uploads,processed,vector_db}
mkdir -p logs
mkdir -p tests
mkdir -p scripts

# Create __init__.py files
touch src/__init__.py
touch src/document_processor/__init__.py
touch src/embeddings/__init__.py
touch src/vector_store/__init__.py
touch src/llm/__init__.py
touch src/api/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
```

### 2.2 Create Requirements File

Create `requirements.txt`:

```txt
# Web Framework
Flask==3.0.0
gunicorn==21.2.0
flask-cors==4.0.0

# Document Processing
PyPDF2==3.0.1
pdfplumber==0.10.3
openpyxl==3.1.2
pandas==2.1.4

# Vector Database
chromadb==0.4.22

# HTTP Client
requests==2.31.0

# Authentication
PyJWT==2.8.0
bcrypt==4.1.2

# Environment Variables
python-dotenv==1.0.0

# Monitoring & Logging
prometheus-client==0.19.0

# Utilities
python-dateutil==2.8.2
uuid==1.30

# Development Tools (optional)
pytest==7.4.3
pytest-cov==4.1.0
black==23.12.1
pylint==3.0.3
mypy==1.8.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2.3 Create Configuration System

Create `config/settings.py`:

```python
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    
    # Application
    APP_NAME = os.getenv('APP_NAME', 'RAG Ollama App')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Ollama
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
    OLLAMA_LLM_MODEL = os.getenv('OLLAMA_LLM_MODEL', 'llama2')
    
    # Document Processing
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '100'))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
    ALLOWED_FILE_TYPES = ['pdf', 'xlsx', 'xls']
    
    # RAG Configuration
    TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '5'))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
    MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', '4000'))
    
    # Storage
    BASE_DIR = Path(__file__).parent.parent
    VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', str(BASE_DIR / 'data' / 'vector_db'))
    UPLOAD_DIR = os.getenv('UPLOAD_DIR', str(BASE_DIR / 'data' / 'uploads'))
    PROCESSED_DIR = os.getenv('PROCESSED_DIR', str(BASE_DIR / 'data' / 'processed'))
    LOG_DIR = str(BASE_DIR / 'logs')
    
    # Performance
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))
    USE_QUERY_CACHE = os.getenv('USE_QUERY_CACHE', 'True').lower() == 'true'
    CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '3600'))
    
    # Authentication
    TOKEN_EXPIRY_SECONDS = int(os.getenv('TOKEN_EXPIRY_SECONDS', '3600'))
    MAX_REQUESTS_PER_HOUR = int(os.getenv('MAX_REQUESTS_PER_HOUR', '100'))

# Export configuration instance
config = Config()
```

Create `.env.example`:

```bash
# Application
APP_NAME=RAG Ollama App
DEBUG=False
SECRET_KEY=your-secret-key-here
LOG_LEVEL=INFO

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama2

# Document Processing
MAX_FILE_SIZE_MB=100
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# RAG Configuration
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7

# Storage
VECTOR_DB_PATH=./data/vector_db
UPLOAD_DIR=./data/uploads
PROCESSED_DIR=./data/processed

# Performance
USE_QUERY_CACHE=True
CACHE_TTL_SECONDS=3600
```

Copy to `.env`:

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 2.4 Create Logging Configuration

Create `config/logging_config.py`:

```python
import logging
import os
from logging.handlers import RotatingFileHandler
from config.settings import config

def setup_logging():
    """Configure application logging"""
    
    # Create logs directory
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                os.path.join(config.LOG_DIR, 'app.log'),
                maxBytes=10485760,  # 10MB
                backupCount=10
            ),
            logging.StreamHandler()
        ]
    )
    
    # Create error logger
    error_handler = RotatingFileHandler(
        os.path.join(config.LOG_DIR, 'error.log'),
        maxBytes=10485760,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
    ))
    
    logging.getLogger().addHandler(error_handler)
    
    return logging.getLogger(__name__)

# Initialize logging
logger = setup_logging()
```

**Checkpoint:** Verify configuration loads correctly:

```bash
python -c "from config.settings import config; print(f'Config loaded: {config.APP_NAME}')"
```

---

## 3. Phase 2: Document Processing (Week 1-2)

### 3.1 Create PDF Processor

Create `src/document_processor/pdf_processor.py`:

```python
import logging
from pathlib import Path
from typing import Dict, List
import pdfplumber

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF documents and extract text"""
    
    def extract_text(self, file_path: str) -> Dict[str, any]:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                page_texts = []
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        page_texts.append({
                            'page_number': page_num,
                            'text': page_text
                        })
                
                metadata = {
                    'total_pages': len(pdf.pages),
                    'filename': Path(file_path).name,
                    'file_size': Path(file_path).stat().st_size
                }
                
                logger.info(f"Extracted {len(text)} characters from {metadata['total_pages']} pages")
                
                return {
                    'text': text,
                    'page_texts': page_texts,
                    'metadata': metadata
                }
                
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise DocumentProcessingError(f"PDF extraction failed: {e}")

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass
```

### 3.2 Create Excel Processor

Create `src/document_processor/excel_processor.py`:

```python
import logging
from pathlib import Path
from typing import Dict, List
import openpyxl

logger = logging.getLogger(__name__)

class ExcelProcessor:
    """Process Excel documents and extract text"""
    
    def extract_text(self, file_path: str) -> Dict[str, any]:
        """
        Extract text from Excel file
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            workbook = openpyxl.load_workbook(file_path, data_only=True)
            text = ""
            sheet_texts = []
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                sheet_text = f"Sheet: {sheet_name}\n"
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = ' | '.join([str(cell) if cell is not None else '' for cell in row])
                    if row_text.strip():
                        sheet_text += row_text + "\n"
                
                text += sheet_text + "\n\n"
                sheet_texts.append({
                    'sheet_name': sheet_name,
                    'text': sheet_text
                })
            
            metadata = {
                'total_sheets': len(workbook.sheetnames),
                'sheet_names': workbook.sheetnames,
                'filename': Path(file_path).name,
                'file_size': Path(file_path).stat().st_size
            }
            
            logger.info(f"Extracted {len(text)} characters from {metadata['total_sheets']} sheets")
            
            return {
                'text': text,
                'sheet_texts': sheet_texts,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to extract text from Excel: {e}")
            raise DocumentProcessingError(f"Excel extraction failed: {e}")

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass
```

### 3.3 Create Document Chunker

Create `src/document_processor/chunker.py`:

```python
import logging
import re
from typing import List, Dict
from config.settings import config

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Split documents into semantic chunks"""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap = overlap or config.CHUNK_OVERLAP
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries
        """
        # Split by paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para.split())
            
            if current_length + para_length <= self.chunk_size:
                current_chunk += para + "\n\n"
                current_length += para_length
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, len(chunks), metadata))
                
                if para_length > self.chunk_size:
                    # Split large paragraph
                    sub_chunks = self._split_large_paragraph(para)
                    for sub_chunk in sub_chunks:
                        chunks.append(self._create_chunk(sub_chunk, len(chunks), metadata))
                    current_chunk = ""
                    current_length = 0
                else:
                    # Add overlap from previous chunk
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = overlap_text + para
                    current_length = len(current_chunk.split())
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk, len(chunks), metadata))
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines or sentence boundaries
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split large paragraph into smaller chunks"""
        words = paragraph.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(' '.join(chunk_words))
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from end of chunk"""
        words = text.split()
        if len(words) <= self.overlap:
            return text
        overlap_words = words[-self.overlap:]
        return ' '.join(overlap_words) + " "
    
    def _create_chunk(self, text: str, index: int, metadata: Dict = None) -> Dict:
        """Create chunk dictionary"""
        return {
            'text': text.strip(),
            'chunk_index': index,
            'token_count': len(text.split()),
            'metadata': metadata or {}
        }
```

### 3.4 Create Main Document Processor

Create `src/document_processor/processor.py`:

```python
import logging
import uuid
from pathlib import Path
from typing import Dict, List
from .pdf_processor import PDFProcessor
from .excel_processor import ExcelProcessor
from .chunker import DocumentChunker

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Main document processor coordinating all processing steps"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.excel_processor = ExcelProcessor()
        self.chunker = DocumentChunker()
    
    def process_document(self, file_path: str, file_type: str) -> Dict:
        """
        Process document and return chunks with metadata
        
        Args:
            file_path: Path to document
            file_type: Type of document ('pdf' or 'excel')
            
        Returns:
            Dictionary with processed document data
        """
        logger.info(f"Processing document: {file_path} (type: {file_type})")
        
        # Extract text
        if file_type == 'pdf':
            extraction_result = self.pdf_processor.extract_text(file_path)
        elif file_type in ['xlsx', 'xls', 'excel']:
            extraction_result = self.excel_processor.extract_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = {
            'document_id': document_id,
            'filename': extraction_result['metadata']['filename'],
            'file_type': file_type,
            'file_size': extraction_result['metadata']['file_size']
        }
        
        # Chunk document
        chunks = self.chunker.chunk_text(extraction_result['text'], metadata)
        
        return {
            'document_id': document_id,
            'metadata': metadata,
            'chunks': chunks,
            'total_chunks': len(chunks)
        }
```

**Checkpoint:** Test document processing:

```bash
# Create test script
cat > test_processing.py << 'EOF'
from src.document_processor.processor import DocumentProcessor

processor = DocumentProcessor()

# Test with a sample PDF (create one or use existing)
# result = processor.process_document('test.pdf', 'pdf')
# print(f"Processed: {result['total_chunks']} chunks")
EOF

python test_processing.py
```

---

## 4. Phase 3: Vector Database & Embeddings (Week 2)

### 4.1 Create Ollama Client

Create `src/embeddings/ollama_client.py`:

```python
import logging
import requests
import time
from typing import List, Dict
from config.settings import config

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.embedding_model = config.OLLAMA_EMBEDDING_MODEL
        self.llm_model = config.OLLAMA_LLM_MODEL
    
    def check_health(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['embedding']
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for text in batch:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
                time.sleep(0.1)  # Rate limiting
        
        return embeddings
    
    def generate_response(self, prompt: str, stream: bool = False) -> str:
        """
        Generate LLM response
        
        Args:
            prompt: Input prompt
            stream: Whether to stream response
            
        Returns:
            Generated text
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": stream
                },
                stream=stream,
                timeout=120
            )
            response.raise_for_status()
            
            if stream:
                return response  # Return response object for streaming
            else:
                return response.json()['response']
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMError(f"Failed to generate response: {e}")

class EmbeddingError(Exception):
    """Custom exception for embedding generation errors"""
    pass

class LLMError(Exception):
    """Custom exception for LLM generation errors"""
    pass
```

### 4.2 Create Embedding Generator

Create `src/embeddings/embedding_generator.py`:

```python
import logging
from typing import List, Dict
from functools import lru_cache
import hashlib
from .ollama_client import OllamaClient
from config.settings import config

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate and cache embeddings"""
    
    def __init__(self):
        self.client = OllamaClient()
        self._cache = {}
    
    @lru_cache(maxsize=1000)
    def generate(self, text: str) -> List[float]:
        """
        Generate embedding with caching
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Generate embedding
        embedding = self.client.generate_embedding(text)
        return embedding
    
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts
            
        Returns:
            List of embedding vectors
        """
        return self.client.generate_embeddings_batch(texts, batch_size=config.BATCH_SIZE)
    
    def generate_for_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Chunks with embeddings added
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        return chunks
```

### 4.3 Create Vector Store Interface

Create `src/vector_store/vector_db.py`:

```python
import logging
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from pathlib import Path
from config.settings import config

logger = logging.getLogger(__name__)

class VectorStore:
    """Interface for ChromaDB vector database operations"""
    
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or config.VECTOR_DB_PATH
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"description": "Document chunks with embeddings"}
        )
        
        logger.info(f"Vector store initialized at {self.persist_directory}")
    
    def add_documents(self, chunks: List[Dict]) -> None:
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of chunk dictionaries with embeddings
        """
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for chunk in chunks:
            # Create unique ID
            chunk_id = f"{chunk['metadata']['document_id']}_{chunk['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk['text'])
            embeddings.append(chunk['embedding'])
            
            # Prepare metadata
            metadata = {
                'document_id': chunk['metadata']['document_id'],
                'filename': chunk['metadata']['filename'],
                'file_type': chunk['metadata']['file_type'],
                'chunk_index': chunk['chunk_index']
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query_embedding: List[float], top_k: int = None, 
               threshold: float = None) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        top_k = top_k or config.TOP_K_RESULTS
        threshold = threshold or config.SIMILARITY_THRESHOLD
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            # Calculate similarity score (ChromaDB returns distance)
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            
            if similarity >= threshold:
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'score': similarity,
                    'metadata': results['metadatas'][0][i]
                })
        
        logger.info(f"Found {len(formatted_results)} results above threshold")
        return formatted_results
    
    def delete_document(self, document_id: str) -> None:
        """
        Delete all chunks for a document
        
        Args:
            document_id: Document ID
        """
        # Query to find all chunks with this document_id
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
    
    def count(self) -> int:
        """Get total number of chunks in store"""
        return self.collection.count()
```

**Checkpoint:** Test vector store operations:

```bash
cat > test_vector_store.py << 'EOF'
from src.vector_store.vector_db import VectorStore
from src.embeddings.embedding_generator import EmbeddingGenerator

# Initialize
vector_store = VectorStore()
generator = EmbeddingGenerator()

# Test embedding generation
text = "This is a test document."
embedding = generator.generate(text)
print(f"Generated embedding with {len(embedding)} dimensions")

# Test count
count = vector_store.count()
print(f"Vector store contains {count} chunks")
EOF

python test_vector_store.py
```

---

## 5. Phase 4: RAG Pipeline (Week 3)

### 5.1 Create Prompt Templates

Create `src/llm/prompt_templates.py`:

```python
from typing import List, Dict

def create_rag_prompt(query: str, context_chunks: List[Dict]) -> str:
    """
    Create RAG prompt with context
    
    Args:
        query: User query
        context_chunks: Retrieved context chunks
        
    Returns:
        Formatted prompt
    """
    # Build context section
    context_text = ""
    for i, chunk in enumerate(context_chunks, start=1):
        context_text += f"[{i}] {chunk['text']}\n\n"
    
    # Construct full prompt
    prompt = f"""You are a helpful assistant answering questions based on provided documents.

Context Information:
{context_text}

User Question: {query}

Instructions:
- Answer the question using ONLY the information from the context above
- Cite sources using [1], [2], etc. after each claim
- If the context doesn't contain enough information, say "I don't have enough information to answer this question."
- Be concise but complete
- Do not make up information

Answer:"""
    
    return prompt


def create_system_prompt() -> str:
    """Create system prompt for the LLM"""
    return """You are an AI assistant that helps users find information in their document collection. 
You always cite your sources and never make up information that isn't in the provided context."""
```

### 5.2 Create LLM Response Generator

Create `src/llm/response_generator.py`:

```python
import logging
from typing import List, Dict, Generator
from .prompt_templates import create_rag_prompt
from src.embeddings.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generate LLM responses for RAG"""
    
    def __init__(self):
        self.client = OllamaClient()
    
    def generate(self, query: str, context_chunks: List[Dict]) -> Dict:
        """
        Generate response for query with context
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Dictionary with answer and metadata
        """
        # Create prompt
        prompt = create_rag_prompt(query, context_chunks)
        
        # Generate response
        logger.info("Generating LLM response")
        response = self.client.generate_response(prompt, stream=False)
        
        return {
            'answer': response,
            'sources': context_chunks,
            'num_sources': len(context_chunks)
        }
    
    def generate_streaming(self, query: str, context_chunks: List[Dict]) -> Generator:
        """
        Generate streaming response
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Yields:
            Response tokens
        """
        # Create prompt
        prompt = create_rag_prompt(query, context_chunks)
        
        # Generate streaming response
        logger.info("Generating streaming LLM response")
        response = self.client.generate_response(prompt, stream=True)
        
        for line in response.iter_lines():
            if line:
                import json
                chunk = json.loads(line)
                if 'response' in chunk:
                    yield chunk['response']
```

### 5.3 Create RAG Pipeline Orchestrator

Create `src/rag_pipeline.py`:

```python
import logging
import time
from typing import Dict, List
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_store.vector_db import VectorStore
from src.llm.response_generator import ResponseGenerator
from config.settings import config

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.response_generator = ResponseGenerator()
    
    def process_query(self, query: str) -> Dict:
        """
        Process user query through RAG pipeline
        
        Args:
            query: User query
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        logger.info(f"Processing query: {query}")
        
        # Step 1: Generate query embedding
        query_embedding = self.embedding_generator.generate(query)
        logger.info("Generated query embedding")
        
        # Step 2: Search vector store
        search_results = self.vector_store.search(
            query_embedding,
            top_k=config.TOP_K_RESULTS,
            threshold=config.SIMILARITY_THRESHOLD
        )
        
        if not search_results:
            logger.warning("No relevant documents found")
            return {
                'answer': "I don't have enough information to answer this question.",
                'sources': [],
                'metadata': {
                    'query_time': time.time() - start_time,
                    'num_sources': 0
                }
            }
        
        logger.info(f"Found {len(search_results)} relevant chunks")
        
        # Step 3: Generate response
        result = self.response_generator.generate(query, search_results)
        
        # Step 4: Add metadata
        query_time = time.time() - start_time
        result['metadata'] = {
            'query_time': query_time,
            'num_sources': len(search_results),
            'model': config.OLLAMA_LLM_MODEL
        }
        
        logger.info(f"Query processed in {query_time:.2f} seconds")
        return result
    
    def process_document(self, file_path: str, file_type: str) -> Dict:
        """
        Process and index document
        
        Args:
            file_path: Path to document
            file_type: Type of document
            
        Returns:
            Dictionary with processing results
        """
        from src.document_processor.processor import DocumentProcessor
        
        logger.info(f"Processing document: {file_path}")
        
        # Step 1: Process document
        processor = DocumentProcessor()
        processed = processor.process_document(file_path, file_type)
        
        # Step 2: Generate embeddings
        chunks_with_embeddings = self.embedding_generator.generate_for_chunks(
            processed['chunks']
        )
        
        # Step 3: Add to vector store
        self.vector_store.add_documents(chunks_with_embeddings)
        
        logger.info(f"Document indexed: {processed['document_id']}")
        
        return {
            'document_id': processed['document_id'],
            'total_chunks': processed['total_chunks'],
            'status': 'indexed'
        }
```

**Checkpoint:** Test RAG pipeline:

```bash
cat > test_rag_pipeline.py << 'EOF'
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# Test document processing
# result = pipeline.process_document('test.pdf', 'pdf')
# print(f"Indexed document: {result['document_id']}")

# Test query
# answer = pipeline.process_query("What is the main topic?")
# print(f"Answer: {answer['answer']}")
EOF

python test_rag_pipeline.py
```

---

*Due to length constraints, I'll continue with the remaining phases in the next section...*

### Continue to Phases 5-10...

---

## 6. Phase 5: API Layer (Week 3-4)

[Content for API routes, Flask app setup, error handling...]

## 7. Phase 6: Frontend (Week 4)

[Content for HTML templates, CSS, JavaScript chat interface...]

## 8. Phase 7: Authentication & Security (Week 5)

[Content for JWT auth, user management, security measures...]

## 9. Phase 8: Testing & Quality Assurance (Week 5-6)

[Content for unit tests, integration tests, performance tests...]

## 10. Phase 9: Deployment & Production (Week 6)

[Content for Gunicorn, Nginx, systemd services, monitoring...]

## 11. Verification & Troubleshooting

[Content for common issues, debugging tips, health checks...]

## 12. Next Steps & Enhancements

[Content for Phase 2 features, scalability improvements...]

---

**Note:** This implementation guide continues in IMPLEMENTATION_GUIDE_PART2.md due to length constraints.
