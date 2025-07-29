import streamlit as st
import os
import pandas as pd
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any
import time
import io
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Page configuration
st.set_page_config(
    page_title="🏠 Real Estate Assistant",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic styling with typewriter effect
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
    font-family: 'Orbitron', 'Arial', sans-serif;
    color: #e0e0e0;
}

/* Main Header */
.main-header {
    text-align: center;
    color: #00ddeb;
    font-size: 3rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    text-shadow: 0 0 10px #00ddeb, 0 0 20px #00ddeb;
    margin-bottom: 2.5rem;
    animation: glow 2s ease-in-out infinite alternate;
}

/* Chat Message with Typewriter Effect */
.chat-message {
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 1rem 0;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid #00ddeb;
    box-shadow: 0 0 15px rgba(0, 221, 235, 0.3);
    transition: all 0.3s ease;
}
.chat-message:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 25px rgba(0, 221, 235, 0.5);
}
.chat-message p {
    margin: 0;
    white-space: pre-wrap;
    overflow: hidden;
    animation: typewriter 0.1s steps(1, end) infinite;
}
@keyframes typewriter {
    from { width: 0; }
    to { width: 100%; }
}

/* Property Card */
.property-card {
    background: rgba(255, 255, 255, 0.1);
    padding: 2rem;
    border-radius: 1.2rem;
    border: 1px solid #415a77;
    box-shadow: 0 0 20px rgba(0, 221, 235, 0.2);
    margin: 1rem 0;
    transition: all 0.3s ease;
}
.property-card:hover {
    transform: scale(1.02);
    box-shadow: 0 0 30px rgba(0, 221, 235, 0.4);
}

/* Success Message */
.success-message {
    background: rgba(0, 255, 184, 0.2);
    color: #00ffbb;
    padding: 1rem;
    border-radius: 0.8rem;
    border: 1px solid #00ffbb;
    box-shadow: 0 0 15px rgba(0, 255, 184, 0.3);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(45deg, #00ddeb, #ff00ff);
    color: #ffffff;
    border: none;
    border-radius: 0.5rem;
    padding: 0.8rem 1.5rem;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(45deg, #ff00ff, #00ddeb);
    box-shadow: 0 0 20px rgba(0, 221, 235, 0.6);
    transform: translateY(-2px);
}

/* Sidebar */
.css-1d391kg {
    background: rgba(27, 38, 59, 0.9);
    border-right: 1px solid #00ddeb;
    box-shadow: 0 0 20px rgba(0, 221, 235, 0.2);
}

/* Input Fields */
.stTextInput>div>input {
    background: rgba(255, 255, 255, 0.1);
    color: #e0e0e0;
    border: 1px solid #00ddeb;
    border-radius: 0.5rem;
    padding: 0.8rem;
    transition: all 0.3s ease;
}
.stTextInput>div>input:focus {
    box-shadow: 0 0 15px rgba(0, 221, 235, 0.5);
    border-color: #ff00ff;
}

/* File Uploader */
.stFileUploader {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid #00ddeb;
    border-radius: 0.8rem;
    padding: 1rem;
}

/* Animation Keyframes */
@keyframes glow {
    from { text-shadow: 0 0 10px #00ddeb, 0 0 20px #00ddeb; }
    to { text-shadow: 0 0 20px #ff00ff, 0 0 30px #ff00ff; }
}

/* Progress Bar */
.stProgress .st-bo {
    background: #00ddeb;
    box-shadow: 0 0 10px #00ddeb;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #1b263b; }
::-webkit-scrollbar-thumb { background: #00ddeb; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #ff00ff; }
</style>
""", unsafe_allow_html=True)

# Constants
EMBEDDING_MODEL = "embedding-001"  # Google embedding model
GENERATIVE_MODEL = "gemini-2.5-pro"  # Gemini model for text generation
VECTOR_DIM = 768  # Dimension for embedding-001
DATA_DIR = "data"
API_KEY = "ENTER YOUR API KEY"  # Hardcoded API key (insecure, for testing only)

def get_gemini_client():
    """Initialize Gemini client with hardcoded API key"""
    try:
        st.warning("⚠️ Using hardcoded API key for testing. For security, revoke this key and store a new one in environment variables or .streamlit/secrets.toml.")
        return True
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {str(e)}")
        return None

async def get_embedding_async(texts: List[str]) -> List[np.ndarray]:
    """Asynchronously generate embeddings for multiple texts using GoogleGenerativeAIEmbeddings"""
    try:
        if not texts or not all(isinstance(t, str) for t in texts):
            st.error("Invalid input texts for embedding.")
            return []
        
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
        embeddings_list = await embeddings.aembed_documents(texts)
        return [np.array(e, dtype=np.float32) for e in embeddings_list]
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return []

def get_embedding(text: str) -> np.ndarray:
    """Synchronous wrapper for single embedding generation"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        embedding = loop.run_until_complete(get_embedding_async([text]))[0] if loop.run_until_complete(get_embedding_async([text])) else None
        loop.close()
        return embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def load_csv_data(csv_file) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Load and process Excel/CSV data into document format"""
    try:
        if hasattr(csv_file, 'read'):
            file_bytes = csv_file.read()
            if csv_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(io.BytesIO(file_bytes))
            else:
                df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            if csv_file.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(csv_file)
            else:
                df = pd.read_csv(csv_file)
        
        documents = []
        for idx, row in df.iterrows():
            content = f"""
Property ID: {row.get('Property ID', 'N/A')}
Project Name: {row.get('Project Name', 'N/A')}
Location: {row.get('Location', 'N/A')}
Address: {row.get('Address', 'N/A')}
Status: {row.get('Status', 'N/A')}
Type: {row.get('Type', 'N/A')}
BHK: {row.get('BHK', 'N/A')}
Size: {row.get('Size (sq.ft.)', 'N/A')} sq.ft.
Price: ₹{row.get('Start Price', 'N/A')}
Price per sq.ft.: ₹{row.get('Price/sq.ft', 'N/A')}
Amenities: {row.get('Amenities', 'N/A')}
Nearby: {row.get('Nearby', 'N/A')}
Furnishing: {row.get('Furnishing', 'N/A')}
Contact Person: {row.get('Contact Person', 'N/A')}
Contact Number: {row.get('Contact', 'N/A')}
Offers: {row.get('Offers', 'N/A')}
"""
            metadata = {
                "source": "property_data",
                "property_id": row.get('Property ID', 'N/A'),
                "location": row.get('Location', 'N/A'),
                "price": row.get('Start Price', 'N/A'),
                "bhk": row.get('BHK', 'N/A')
            }
            documents.append((f"property-{idx}", content.strip(), metadata))
        return documents
    except Exception as e:
        st.error(f"Error loading CSV/Excel data: {str(e)}")
        return []

def load_pdf_data(pdf_file) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Load and chunk PDF data"""
    try:
        if hasattr(pdf_file, 'read'):
            pdf_bytes = pdf_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        else:
            doc = fitz.open(pdf_file)
        
        full_text = ""
        for page in doc:
            text = page.get_text()
            if text:
                full_text += text + "\n"
        doc.close()
        
        # Create overlapping chunks
        chunks = []
        chunk_size = 500
        overlap = chunk_size // 4
        for i in range(0, len(full_text), chunk_size - overlap):
            chunk = full_text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append((f"guide-{i}", chunk, {"source": "guidelines"}))
        return documents
    except Exception as e:
        st.error(f"Error loading PDF data: {str(e)}")
        return []

class SmartFaissIndex:
    """Enhanced FAISS index with search capabilities"""
    def __init__(self):
        self.index = faiss.IndexFlatL2(VECTOR_DIM)
        self.metadata = []
        self.documents = []

    def add_documents(self, documents: List[Tuple[str, str, Dict[str, Any]]]):
        """Add documents to the index with embeddings"""
        if not documents:
            st.error("No documents provided to index.")
            return
        
        contents = [doc[1] for doc in documents]
        embeddings = asyncio.run(get_embedding_async(contents))  # Run async function synchronously
        if not embeddings or len(embeddings) != len(documents):
            st.error("Failed to generate embeddings for all documents.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (doc_id, _, meta) in enumerate(documents):
            status_text.text(f"Processing document {i+1}/{len(documents)}...")
            self.documents.append((doc_id, contents[i], meta))
            progress_bar.progress((i + 1) / len(documents))
        
        self.index.add(np.array(embeddings))
        self.metadata.extend(self.documents)
        status_text.text("✅ All documents processed successfully!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search for relevant documents"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                _, content, _ = self.documents[idx]
                results.append(content)
        return results

def generate_intelligent_response(query: str, context_chunks: List[str]) -> str:
    """Generate a response using LangChain ChatGoogleGenerativeAI with progress feedback"""
    if not context_chunks:
        return "I couldn't find relevant information to answer your question. Please try rephrasing or upload new data."
    
    context = "\n\n---\n\n".join(context_chunks)
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template="""
You are an expert real estate assistant with deep knowledge of property markets in Chennai.
Based on the following property data and community guidelines, provide a helpful, detailed response to the user's question.

CONTEXT INFORMATION:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer that:
- Directly addresses the user's question
- Uses specific details from the provided data (e.g., Property ID, prices, locations, amenities)
- Offers practical insights and recommendations
- Maintains a friendly, professional tone
- Organizes information clearly with bullet points when appropriate
- If the question is about specific properties, include details like Property ID, price, location, and amenities
- If the question involves community guidelines, reference specific rules or policies
- If the question is general, provide market insights based on the available data
"""
    )
    
    try:
        if not get_gemini_client():
            return "Gemini client not initialized."
        
        llm = ChatGoogleGenerativeAI(model=GENERATIVE_MODEL, google_api_key=API_KEY, temperature=0.7)
        prompt = prompt_template.format(context=context, query=query)
        with st.spinner("Generating response..."):
            response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"I apologize, but I encountered an error while generating the response: {str(e)}"

def main():
    st.markdown('<h1 class="main-header">🏠 Intelligent Real Estate Assistant</h1>', unsafe_allow_html=True)

    # Sidebar configuration with file upload options
    with st.sidebar:
        st.header("📁 Data Sources")
        csv_file = st.file_uploader(
            "Upload Property Data (Excel/CSV)",
            type=['csv', 'xls', 'xlsx'],
            help="Upload an Excel or CSV file containing property information"
        )
        pdf_file = st.file_uploader(
            "Upload Guidelines (PDF)",
            type=['pdf'],
            help="Upload a PDF file containing real estate guidelines"
        )

    # Main content area
    if not get_gemini_client():
        st.warning("⚠️ Gemini client initialization failed. Check the hardcoded API key or configuration.")
        return

    # Initialize session state
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Data loading section
    if st.session_state.index is None and (csv_file or pdf_file):
        st.header("📊 Data Loading")
        if st.button("🚀 Load and Process Data", type="primary", use_container_width=True):
            if not csv_file and not pdf_file:
                st.error("Please upload at least one file (Excel/CSV or PDF) in the sidebar.")
                return
            
            with st.spinner("Loading and processing your data..."):
                all_documents = []
                if csv_file:
                    csv_docs = load_csv_data(csv_file)
                    all_documents.extend(csv_docs)
                    st.success(f"✅ Loaded {len(csv_docs)} property records")
                if pdf_file:
                    pdf_docs = load_pdf_data(pdf_file)
                    all_documents.extend(pdf_docs)
                    st.success(f"✅ Loaded {len(pdf_docs)} guideline sections")
                
                if all_documents:
                    st.info("🔍 Building search index...")
                    index = SmartFaissIndex()
                    index.add_documents(all_documents)
                    st.session_state.index = index
                    st.markdown('<div class="success-message">🎉 Your real estate assistant is ready! Start asking questions below.</div>', unsafe_allow_html=True)
                else:
                    st.error("No documents were loaded. Please check your uploaded files.")

    # Chat interface
    if st.session_state.index is not None:
        st.header("💬 Chat with Your Real Estate Assistant")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**You:** <p>{question}</p>", unsafe_allow_html=True)
                st.markdown(f"**Assistant:** <p>{answer}</p>", unsafe_allow_html=True)
                st.divider()
        
        # Use a form to manage query input
        with st.form(key="query_form"):
            col1, col2 = st.columns([5, 1])
            with col1:
                user_query = st.text_input(
                    "Ask me anything about real estate...",
                    placeholder="e.g., Show me 3BHK properties under ₹1 crore in Chennai",
                    key="query_input"
                )
            with col2:
                submit_button = st.form_submit_button("Ask", type="primary")
        
        # Process query within the form submission
        if submit_button and user_query:
            with st.spinner("Searching and generating response..."):
                # Retrieve relevant documents
                relevant_docs = st.session_state.index.search(user_query, top_k=5)
                # Generate response
                response = generate_intelligent_response(user_query, relevant_docs)
                # Add to chat history
                st.session_state.chat_history.append((user_query, response))
                # Display latest response
                st.markdown('<div class="chat-message">', unsafe_allow_html=True)
                st.markdown(f"**Your Question:** <p>{user_query}</p>", unsafe_allow_html=True)
                st.markdown(f"**Assistant:** <p>{response}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.rerun()

        # Quick action buttons
        st.header("🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💰 Show Budget Properties", use_container_width=True):
                with st.form(key="budget_form"):
                    st.text_input("Ask me anything about real estate...", value="Show me affordable properties under ₹1 crore in Chennai", key="query_input")
                    if st.form_submit_button("Ask", type="primary"):
                        st.rerun()
        with col2:
            if st.button("🏙️ Popular Locations", use_container_width=True):
                with st.form(key="location_form"):
                    st.text_input("Ask me anything about real estate...", value="What are the most popular locations for real estate investment in Chennai?", key="query_input")
                    if st.form_submit_button("Ask", type="primary"):
                        st.rerun()
        with col3:
            if st.button("📈 Market Insights", use_container_width=True):
                with st.form(key="insight_form"):
                    st.text_input("Ask me anything about real estate...", value="Give me insights about current real estate market trends in Chennai", key="query_input")
                    if st.form_submit_button("Ask", type="primary"):
                        st.rerun()

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()
