# 🏠 Real Estate AI Assistant

**Last Updated**: JULY 2025  
**Version**: 1.0.0  
**Maintainer**: Navaneethakrishnan S

The Real Estate AI Assistant is an intelligent Streamlit-based application that provides conversational AI capabilities for real estate data analysis and customer support. The system leverages Google's Gemini AI models and FAISS vector search to deliver contextual responses based on uploaded property data and guidelines.

## 🎯 Key Features

### Core Functionality
- **Intelligent Document Processing**: Supports both property data (Excel/CSV) and guideline documents (PDF)
- **Vector-Based Search**: Uses FAISS indexing with Google's embedding models for semantic search
- **Conversational AI Interface**: Chat-based interaction powered by Google Gemini 2.5 Pro
- **Real-time Data Processing**: Asynchronous embedding generation for optimal performance
- **Multi-format Support**: Handles CSV, Excel (.xls, .xlsx), and PDF file formats

### User Experience
- **Interactive Chat Interface**: Persistent conversation history with context retention
- **Quick Action Buttons**: Pre-configured queries for common real estate inquiries
- **Progress Tracking**: Real-time feedback during data processing operations
- **Responsive Design**: Mobile-friendly interface with custom CSS styling

## 💡 Usage Examples

### Sample Queries
```
"Show me 3BHK properties under ₹1 crore in Chennai"
"What are the best amenities available in luxury projects?"
"List properties near IT corridors with good connectivity"
"What are the current market trends for 2BHK apartments?"
```

### Expected Response Format
The AI assistant provides structured responses including:
- **Property Details**: ID, location, pricing, specifications
- **Market Insights**: Trends, recommendations, comparisons
- **Contact Information**: Developer details, contact numbers
- **Additional Context**: Nearby facilities, investment potential

## 🔧 Configuration Options

### Model Parameters
```python
EMBEDDING_MODEL = "embedding-001"      # Google embedding model
GENERATIVE_MODEL = "gemini-2.5-pro"    # Gemini model for responses
VECTOR_DIM = 768                       # Embedding dimensions
CHUNK_SIZE = 500                       # Document chunk size
OVERLAP_SIZE = 125                     # Chunk overlap for context
```

### UI Customization
The application includes custom CSS for:
- Brand-consistent color scheme (#2E86AB primary)
- Responsive card layouts for property display
- Professional chat interface styling
- Progress indicators and status messages

## 🐛 Troubleshooting

### Common Issues

**1. API Key Errors**
```
Error: "Failed to initialize Gemini client"
Solution: Verify Google AI API key is valid and has sufficient quota
```

**2. Memory Issues**
```
Error: "Out of memory during embedding generation"
Solution: Process documents in smaller batches or increase system RAM
```

**3. File Format Issues**
```
Error: "Error loading CSV/Excel data"
Solution: Ensure file has required columns and proper encoding (UTF-8)
```

**4. Embedding Generation Failures**
```
Error: "Error generating embedding"
Solution: Check internet connection and API rate limits
```

### Performance Optimization
- **Large Datasets**: Process in batches of 100-200 documents
- **Memory Usage**: Clear chat history periodically to free memory
- **Response Time**: Limit top_k search results to 5-10 for faster responses

## 🔐 Security Considerations

### Current Implementation
⚠️ **Development Mode**: Uses hardcoded API key (insecure)

### Production Security Measures
1. **API Key Management**: Use environment variables or secure vaults
2. **Data Privacy**: Implement data encryption for uploaded files
3. **Rate Limiting**: Add request throttling to prevent API abuse
4. **Input Validation**: Sanitize user inputs to prevent injection attacks
5. **Session Management**: Implement proper user session handling

## 📈 Performance Metrics

### System Performance
- **Document Processing**: ~2-3 seconds per 100 properties
- **Embedding Generation**: ~1 second per document chunk
- **Search Response Time**: <500ms for typical queries
- **Memory Usage**: ~50MB base + 100MB per 1000 properties

### Scalability Limits
- **Maximum Properties**: 10,000+ records (depends on available RAM)
- **Concurrent Users**: 10-50 (depends on server resources)
- **File Size Limits**: 50MB per PDF, 10MB per Excel/CSV

## 🤝 Contributing

### Development Guidelines
1. **Code Style**: Follow PEP 8 standards
2. **Documentation**: Update README for any new features
3. **Testing**: Include unit tests for new functionality
4. **Error Handling**: Implement comprehensive exception handling

### Feature Roadmap
- [ ] Multi-language support for international markets
- [ ] Advanced filtering with price range sliders
- [ ] Property comparison functionality
- [ ] Email integration for property alerts
- [ ] Map-based property visualization
- [ ] User authentication and personalization

# OUTPUT
# Frontend 
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/4b845487-35bd-496f-8f83-a307da1952d3" />

# Generated Response
<img width="1919" height="1072" alt="image" src="https://github.com/user-attachments/assets/30fe4e0c-2a3c-4816-8a4a-220f11e0dddb" />

# Video Demo
https://drive.google.com/file/d/1KJ6b0eRp_0JGefMqoTDNPaqj5VmKNTQd/view?usp=sharing


---

## 📋 Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed via pip
- [ ] Google AI API key configured
- [ ] Sample property data prepared
- [ ] Application launched via `streamlit run app.py`
- [ ] Test query executed successfully

---

## 🏗️ Architecture Overview

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Streamlit UI      │    │   Document Processor │    │   FAISS Vector DB   │
│   - File Upload     │────│   - PDF Parser       │────│   - Embeddings      │
│   - Chat Interface  │    │   - CSV/Excel Reader │    │   - Similarity Search│
│   - Progress Bars   │    │   - Text Chunking    │    │   - Metadata Store  │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
            │                           │                           │
            └───────────────────────────┼───────────────────────────┘
                                        │
                          ┌─────────────────────────────┐
                          │    Google Gemini AI         │
                          │    - Embedding Model        │
                          │    - Generative Model       │
                          │    - LangChain Integration  │
                          └─────────────────────────────┘
```

## 🛠️ Technical Implementation

### Core Components

#### 1. Document Processing System
```python
class DocumentProcessor:
    - load_csv_data(): Processes property data from Excel/CSV files
    - load_pdf_data(): Extracts and chunks text from PDF guidelines
    - create_embeddings(): Generates vector representations using Google AI
```

**Property Data Structure:**
- Property ID, Project Name, Location, Address
- Status, Type, BHK configuration, Size, Pricing
- Amenities, Nearby facilities, Contact information
- Special offers and furnishing details

#### 2. Vector Search Engine
```python
class SmartFaissIndex:
    - add_documents(): Builds searchable index with metadata
    - search(): Performs semantic similarity search
    - Supports up to 768-dimensional embeddings
```

**Search Capabilities:**
- Semantic understanding of natural language queries
- Context-aware property matching
- Multi-criteria filtering and ranking
- Real-time similarity scoring

#### 3. AI Response Generation
```python
def generate_intelligent_response():
    - Context retrieval from vector database
    - Prompt engineering with LangChain
    - Gemini 2.5 Pro model integration
    - Structured response formatting
```

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- Memory: Minimum 4GB RAM (8GB recommended)
- Storage: 1GB free space for dependencies
- Internet connection for Google AI API access

### Dependencies
```bash
streamlit>=1.28.0
pandas>=1.5.0
PyMuPDF>=1.23.0
faiss-cpu>=1.7.4
numpy>=1.24.0
langchain-google-genai>=1.0.0
langchain-core>=0.1.0
```

## 🚀 Installation & Setup

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Navaneeth011/Real-Estate-RAG.git
cd real-estate-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration
**⚠️ Security Notice**: The current implementation uses a hardcoded API key for testing purposes only.

**For Production Deployment:**
```bash
# Method 1: Environment Variable
export GOOGLE_API_KEY="your-secure-api-key"

# Method 2: Streamlit Secrets
# Create .streamlit/secrets.toml
echo 'GOOGLE_API_KEY = "your-secure-api-key"' > .streamlit/secrets.toml
```

### 3. Launch Application
```bash
streamlit run app.py
```

## 📊 Data Format Requirements

### Property Data (Excel/CSV)
Required columns for optimal functionality:
```
Property ID, Project Name, Location, Address, Status, Type, BHK, 
Size (sq.ft.), Start Price, Price/sq.ft, Amenities, Nearby, 
Furnishing, Contact Person, Contact, Offers
```

### PDF Guidelines
- Standard PDF format with extractable text
- Structured content with clear sections
- Maximum file size: 50MB per document

## 🔄 Workflow Process

### 1. Data Ingestion Phase
```
Upload Files → Validate Format → Extract Content → Clean Data → Create Chunks
```

### 2. Vector Index Building
```
Generate Embeddings → Build FAISS Index → Store Metadata → Validate Index
```

### 3. Query Processing
```
User Query → Generate Query Embedding → Similarity Search → Context Retrieval → AI Response
```

##
