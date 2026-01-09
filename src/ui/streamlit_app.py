"""
streamlit_app.py
----------------
Advanced RAG UI with document management, conversation history,
analytics dashboard, and user feedback collection.

Features:
- Document upload and management
- Multi-conversation support
- Real-time streaming responses
- Analytics dashboard
- Advanced settings
- User feedback collection
- Dark/light mode
- Session persistence

Dependencies:
    pip install streamlit pandas plotly requests python-dotenv watchdog
"""

# src/ui/streamlit_app.py

import streamlit as st
import requests
import json
import os
import time
import base64
import logging
from datetime import datetime, timedelta
import uuid
from io import BytesIO
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# Data processing and visualization
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
# API Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")
DEFAULT_USERNAME = os.environ.get("DEFAULT_USERNAME", "demo-user")
DEFAULT_PASSWORD = os.environ.get("DEFAULT_PASSWORD", "")

# UI Configuration
APP_TITLE = os.environ.get("APP_TITLE", "Advanced RAG PDF Chat")
APP_ICON = os.environ.get("APP_ICON", "📄")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "Default")
MAX_HISTORY_SIDEBAR = int(os.environ.get("MAX_HISTORY_SIDEBAR", "5"))
ENABLE_DARK_MODE = os.environ.get("ENABLE_DARK_MODE", "true").lower() == "true"
ENABLE_ANALYTICS = os.environ.get("ENABLE_ANALYTICS", "true").lower() == "true"

# UI color scheme
PRIMARY_COLOR = os.environ.get("PRIMARY_COLOR", "#4CAF50")  # Green
ACCENT_COLOR = os.environ.get("ACCENT_COLOR", "#2196F3")    # Blue
ERROR_COLOR = os.environ.get("ERROR_COLOR", "#F44336")      # Red
SUCCESS_COLOR = os.environ.get("SUCCESS_COLOR", "#4CAF50")  # Green
WARN_COLOR = os.environ.get("WARN_COLOR", "#FF9800")        # Orange

# Cached configuration (loaded from API)
system_info = None

# -------------------------------------------------------------
# Authentication and API Access
# -------------------------------------------------------------
class APIClient:
    """Client for interacting with the RAG API."""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.token = None
        self.token_expiry = None
        self.username = None
    
    def get_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests."""
        # Use token if available and not expired
        if self.token and self.token_expiry and datetime.now() < self.token_expiry:
            return {"Authorization": f"Bearer {self.token}"}
        
        # Use API key if available
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
            
        # No auth
        return {}
    
    def login(self, username: str, password: str) -> bool:
        """Login to the API and get a token."""
        try:
            response = requests.post(
                f"{self.base_url}/token",
                data={
                    "username": username,
                    "password": password,
                    "scope": "read:conversations write:conversations"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                self.username = data["username"]
                # Calculate token expiry time
                expires_at = data.get("expires_at")
                if expires_at:
                    self.token_expiry = datetime.fromtimestamp(expires_at)
                else:
                    # Default to 30 minutes
                    self.token_expiry = datetime.now() + timedelta(minutes=30)
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def upload_pdf(self, file, conversation_id: str = None) -> Dict[str, Any]:
        """Upload a PDF file to the API."""
        try:
            # Prepare form data
            files = {"file": file}
            data = {}
            if conversation_id:
                data["conversation_id"] = conversation_id
                
            # Make request
            response = requests.post(
                f"{self.base_url}/upload_pdf",
                headers=self.get_headers(),
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Upload failed: {response.text}")
                return {"status": "error", "message": f"Upload failed: {response.text}"}
                
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return {"status": "error", "message": str(e)}
    
    def send_query(self, query: str, conversation_id: str = None, 
                   use_optimized_prompts: bool = True, stream: bool = False) -> Dict[str, Any]:
        """Send a query to the API."""
        try:
            # Prepare request body
            payload = {
                "query": query,
                "conversation_id": conversation_id,
                "use_optimized_prompts": use_optimized_prompts,
                "stream": stream
            }
            
            if stream:
                # For streaming, we need to handle SSE (server-sent events)
                response = requests.post(
                    f"{self.base_url}/query",
                    json=payload,
                    headers=self.get_headers(),
                    stream=True
                )
                
                if response.status_code == 200:
                    return response  # Return the response object for streaming
                else:
                    logger.error(f"Query failed: {response.text}")
                    return {"status": "error", "message": f"Query failed: {response.text}"}
            else:
                # Regular query
                response = requests.post(
                    f"{self.base_url}/query",
                    json=payload,
                    headers=self.get_headers()
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Query failed: {response.text}")
                    return {"status": "error", "message": f"Query failed: {response.text}"}
                    
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation history."""
        try:
            response = requests.get(
                f"{self.base_url}/conversation/{conversation_id}",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get conversation: {response.text}")
                return {"status": "error", "message": f"Failed to get conversation: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return {"status": "error", "message": str(e)}
    
    def clear_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Clear a conversation's history."""
        try:
            response = requests.delete(
                f"{self.base_url}/conversation/{conversation_id}",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to clear conversation: {response.text}")
                return {"status": "error", "message": f"Failed to clear conversation: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            response = requests.get(
                f"{self.base_url}/system_performance",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get system performance: {response.text}")
                return {"status": "error", "message": f"Failed to get performance data: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            return {"status": "error", "message": str(e)}
    
    def submit_feedback(self, query: str, response: str, 
                        score: float, feedback_text: str = None) -> Dict[str, Any]:
        """Submit user feedback."""
        try:
            payload = {
                "query": query,
                "response": response,
                "feedback_score": score,
                "feedback_text": feedback_text
            }
            
            response = requests.post(
                f"{self.base_url}/user_feedback",
                json=payload,
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to submit feedback: {response.text}")
                return {"status": "error", "message": f"Failed to submit feedback: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return {"status": "error", "message": str(e)}

# Create API client
api_client = APIClient(API_URL, API_KEY)

# -------------------------------------------------------------
# State Initialization
# -------------------------------------------------------------
def initialize_session_state():
    """Initialize session state with default values."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = not AUTH_ENABLED
        
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}  # id -> metadata
        
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
        
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Current conversation messages
        
    if "documents" not in st.session_state:
        st.session_state.documents = []  # Current conversation documents
    
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "use_optimized_prompts": True,
            "use_memory": True,
            "stream_responses": True,
            "show_sources": True,
            "show_metrics": False,
            "dark_mode": ENABLE_DARK_MODE,
            "expert_mode": False
        }
    
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "queries": 0,
            "documents": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }

# Check if authentication is enabled in API
AUTH_ENABLED = DEFAULT_USERNAME and DEFAULT_PASSWORD

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
def set_page_configuration():
    """Configure Streamlit page settings."""
    if st.session_state.settings.get("dark_mode", False):
        theme_color = "dark"
    else:
        theme_color = "light"
        
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Report a bug': "https://github.com/yourusername/rag-system/issues",
            'About': f"{APP_TITLE} - An advanced RAG system for document question answering."
        }
    )
    
    # Apply custom CSS
    st.markdown(f"""
        <style>
        .stApp {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .chat-message {{
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: flex;
            flex-direction: column;
        }}
        .user-message {{
            background-color: {PRIMARY_COLOR}22;
            border-left: 5px solid {PRIMARY_COLOR};
        }}
        .assistant-message {{
            background-color: {ACCENT_COLOR}22;
            border-left: 5px solid {ACCENT_COLOR};
        }}
        .source-box {{
            background-color: #f0f2f6;
            border-radius: 5px;
            padding: 8px;
            margin-top: 5px;
            font-size: 0.9em;
            border: 1px solid #ddd;
        }}
        .metadata-box {{
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }}
        .main-header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .stButton>button {{
            background-color: {PRIMARY_COLOR};
            color: white;
        }}
        .feedback-buttons button {{
            min-width: 40px;
            height: 30px;
            padding: 0 10px;
            margin: 0 5px;
        }}
        </style>
    """, unsafe_allow_html=True)

def render_login_page():
    """Render the login page."""
    st.markdown("<h1 class='main-header'>🔐 Login Required</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("#### Please login to access the RAG system")
        username = st.text_input("Username", value=DEFAULT_USERNAME)
        password = st.text_input("Password", type="password", value=DEFAULT_PASSWORD)
        
        if st.button("Login", key="login_button"):
            if api_client.login(username, password):
                st.session_state.authenticated = True
                st.success("Login successful!")
                # Force a rerun to show the main interface
                st.rerun()
            else:
                st.error("Login failed. Please check your credentials.")

def load_conversations():
    """Load conversation list from API."""
    # In a real app, you would call an API endpoint to list conversations
    # For now, we'll just use what's in the session state
    return st.session_state.conversations

def create_new_conversation():
    """Create a new conversation and set it as current."""
    # Generate a new conversation ID
    conversation_id = str(uuid.uuid4())
    
    # Add to session state
    st.session_state.conversations[conversation_id] = {
        "id": conversation_id,
        "title": f"Conversation {len(st.session_state.conversations) + 1}",
        "created_at": datetime.now().isoformat(),
        "messages": [],
        "documents": []
    }
    
    # Set as current conversation
    st.session_state.current_conversation_id = conversation_id
    st.session_state.messages = []
    st.session_state.documents = []
    
    return conversation_id

def switch_conversation(conversation_id):
    """Switch to a different conversation."""
    if conversation_id in st.session_state.conversations:
        st.session_state.current_conversation_id = conversation_id
        
        # Load conversation data
        try:
            result = api_client.get_conversation(conversation_id)
            if result and "status" not in result:
                st.session_state.messages = result.get("messages", [])
                st.session_state.documents = result.get("documents", [])
            else:
                st.session_state.messages = []
                st.session_state.documents = []
                
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            st.session_state.messages = []
            st.session_state.documents = []
    else:
        # Create new conversation if ID doesn't exist
        create_new_conversation()

def clear_current_conversation():
    """Clear the current conversation history."""
    if st.session_state.current_conversation_id:
        result = api_client.clear_conversation(st.session_state.current_conversation_id)
        if result and result.get("status") == "success":
            st.session_state.messages = []
            return True
    return False

def update_conversation_title(conversation_id, title):
    """Update the title of a conversation."""
    if conversation_id in st.session_state.conversations:
        st.session_state.conversations[conversation_id]["title"] = title
        return True
    return False

def render_message(message, show_sources=True, show_metrics=False):
    """Render a single chat message with optional sources and metrics."""
    role = message.get("role", "unknown")
    content = message.get("content", "")
    
    # Render message container
    message_class = "user-message" if role == "user" else "assistant-message"
    
    st.markdown(f"<div class='chat-message {message_class}'>", unsafe_allow_html=True)
    
    # Message text
    st.markdown(f"**{role.capitalize()}**")
    st.markdown(content)
    
    # Sources (if available and enabled)
    if role == "assistant" and show_sources:
        metadata = message.get("metadata", {})
        sources = metadata.get("retrieved_documents", [])
        
        if sources:
            with st.expander("View sources"):
                for i, source in enumerate(sources):
                    source_content = source.get("content", "")
                    source_meta = source.get("metadata", {})
                    filename = source_meta.get("filename", "Unknown")
                    
                    st.markdown(f"**Source {i+1}** - {filename}")
                    st.markdown(f"{source_content}")
    
    # Evaluation metrics (if available and enabled)
    if role == "assistant" and show_metrics:
        metadata = message.get("metadata", {})
        eval_data = metadata.get("evaluation", {})
        timings = metadata.get("timings", {})
        
        if eval_data or timings:
            with st.expander("Performance metrics"):
                if eval_data:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Score", f"{eval_data.get('overall_score', 0):.2f}")
                    with col2:
                        st.metric("Faithfulness", f"{eval_data.get('faithfulness', 0):.2f}")
                    with col3:
                        st.metric("Relevance", f"{eval_data.get('relevance', 0):.2f}")
                
                if timings:
                    st.markdown("**Response times:**")
                    total_time = timings.get("total", 0)
                    st.markdown(f"Total: {total_time:.2f}s")
                    
                    # Show a breakdown of times
                    timing_data = {}
                    for key, value in timings.items():
                        if key != "total":
                            timing_data[key] = value
                    
                    if timing_data:
                        timing_df = pd.DataFrame({
                            "Component": list(timing_data.keys()),
                            "Time (s)": list(timing_data.values())
                        })
                        
                        fig = px.bar(
                            timing_df, 
                            x="Component", 
                            y="Time (s)",
                            title="Processing Time Breakdown"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # Close message container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add feedback buttons for assistant messages
    if role == "assistant":
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 3, 1])
        
        with col1:
            if st.button("👍", key=f"thumbs_up_{hash(content[:20])}", help="Good response"):
                submit_feedback(message, 1.0, "Positive feedback")
                
        with col2:
            if st.button("👎", key=f"thumbs_down_{hash(content[:20])}", help="Poor response"):
                # Show feedback form
                st.session_state.feedback_message = message
                st.session_state.show_feedback_form = True
                st.rerun()

def submit_feedback(message, score, feedback_text=None):
    """Submit feedback for a response."""
    # Find the preceding user query
    idx = st.session_state.messages.index(message) if message in st.session_state.messages else -1
    
    if idx > 0 and idx < len(st.session_state.messages):
        # Get the user query
        user_query = st.session_state.messages[idx-1].get("content", "")
        
        # Submit feedback
        result = api_client.submit_feedback(
            query=user_query,
            response=message.get("content", ""),
            score=score,
            feedback_text=feedback_text
        )
        
        if result and result.get("status") == "success":
            st.toast("Thank you for your feedback!", icon="🙏")
            return True
    
    return False

def render_feedback_form():
    """Render a feedback form for negative ratings."""
    if hasattr(st.session_state, "feedback_message"):
        message = st.session_state.feedback_message
        
        with st.form("feedback_form"):
            st.markdown("### Please tell us what was wrong with the response")
            
            issues = st.multiselect(
                "What issues did you notice?",
                options=[
                    "Incorrect information",
                    "Missed the point of my question",
                    "Incomplete answer",
                    "Irrelevant information",
                    "Confusing or unclear",
                    "Other"
                ]
            )
            
            detailed_feedback = st.text_area("Additional comments (optional)")
            
            submit_button = st.form_submit_button("Submit Feedback")
            
            if submit_button:
                feedback_text = f"Issues: {', '.join(issues)}\n\nDetails: {detailed_feedback}"
                if submit_feedback(message, 0.0, feedback_text):
                    st.success("Thank you for your feedback!")
                    st.session_state.pop("feedback_message", None)
                    st.session_state.show_feedback_form = False
                    time.sleep(1)  # Give user time to see success message
                    st.rerun()
                else:
                    st.error("Failed to submit feedback. Please try again.")

def process_streaming_response(response):
    """Process a streaming response from the API."""
    # Create placeholders for the answer
    answer_placeholder = st.empty()
    sources_placeholder = st.empty()
    eval_placeholder = st.empty()
    
    full_answer = ""
    retrieved_documents = []
    evaluation = {}
    
    # Process the streaming response
    try:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                
                # SSE format: lines starting with "data: "
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove "data: " prefix
                    try:
                        data = json.loads(data_str)
                        
                        # Handle different message types
                        msg_type = data.get('type', '')
                        
                        if msg_type == 'metadata':
                            # Handle metadata (conversation_id, etc)
                            continue
                            
                        elif msg_type == 'content':
                            # Update answer content
                            content = data.get('content', '')
                            full_answer += content + " "
                            answer_placeholder.markdown(full_answer)
                            
                        elif msg_type == 'sources':
                            # Store sources for display after completion
                            retrieved_documents = data.get('sources', [])
                            
                            # Show the sources
                            with sources_placeholder.container():
                                if retrieved_documents and st.session_state.settings.get("show_sources", True):
                                    with st.expander("View sources"):
                                        for i, source in enumerate(retrieved_documents):
                                            st.markdown(f"**Source {i+1}**")
                                            st.markdown(source.get("content", ""))
                            
                        elif msg_type == 'evaluation':
                            # Store evaluation data
                            evaluation = data.get('evaluation', {})
                            
                            # Show evaluation metrics
                            with eval_placeholder.container():
                                if evaluation and st.session_state.settings.get("show_metrics", False):
                                    with st.expander("Evaluation metrics"):
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Overall Score", f"{evaluation.get('overall_score', 0):.2f}")
                                        with col2:
                                            st.metric("Faithfulness", f"{evaluation.get('faithfulness', 0):.2f}")
                                        with col3:
                                            st.metric("Relevance", f"{evaluation.get('answer_relevancy', 0):.2f}")
                                            
                        elif msg_type == 'done':
                            # Response is complete
                            break
                            
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in streaming response: {data_str}")
                        continue
            
            # Add a small delay to avoid UI freezing
            time.sleep(0.01)
    
    except Exception as e:
        logger.error(f"Error processing streaming response: {e}")
        answer_placeholder.error(f"Error while receiving response: {str(e)}")
        return None
    
    # Return the complete response data
    return {
        "role": "assistant",
        "content": full_answer,
        "metadata": {
            "retrieved_documents": retrieved_documents,
            "evaluation": evaluation,
            "timestamp": time.time()
        }
    }

def update_metrics(response_time):
    """Update usage metrics."""
    metrics = st.session_state.metrics
    
    # Update query count
    metrics["queries"] += 1
    
    # Update response time metrics
    metrics["total_response_time"] += response_time
    metrics["avg_response_time"] = metrics["total_response_time"] / metrics["queries"]

# -------------------------------------------------------------
# Main UI Components
# -------------------------------------------------------------
def render_sidebar():
    """Render the sidebar with settings and options."""
    with st.sidebar:
        st.markdown(f"# {APP_ICON} {APP_TITLE}")
        
        # User info if authenticated
        if AUTH_ENABLED and st.session_state.authenticated:
            st.info(f"Logged in as: {api_client.username}")
        
        st.divider()
        
        # Conversations section
        st.markdown("### Conversations")
        
        # New conversation button
        if st.button("New Conversation", key="new_conversation"):
            create_new_conversation()
            st.rerun()
        
        # List existing conversations
        conversations = load_conversations()
        if conversations:
            st.markdown("#### Recent Conversations")
            
            # Sort conversations by time (newest first)
            sorted_convs = sorted(
                conversations.items(),
                key=lambda x: x[1].get("created_at", ""),
                reverse=True
            )
            
            # Display recent conversations
            for conv_id, conv in sorted_convs[:MAX_HISTORY_SIDEBAR]:
                title = conv.get("title", f"Conversation {conv_id[:6]}")
                
                # Highlight current conversation
                if st.session_state.current_conversation_id == conv_id:
                    title = f"**→ {title}**"
                    
                if st.sidebar.button(title, key=f"conv_{conv_id}"):
                    switch_conversation(conv_id)
                    st.rerun()
        
        st.divider()
        
        # Document Management
        st.markdown("### Documents")
        
        # Upload PDF
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file and st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                # Ensure we have a conversation to add the document to
                if not st.session_state.current_conversation_id:
                    create_new_conversation()
                
                # Upload the PDF
                result = api_client.upload_pdf(
                    uploaded_file,
                    st.session_state.current_conversation_id
                )
                
                if result and result.get("status") == "success":
                    # Add to documents list
                    st.session_state.documents.append(result["document"])
                    
                    # Update metrics
                    st.session_state.metrics["documents"] += 1
                    
                    st.success(f"PDF processed: {result['document']['filename']}")
                    st.rerun()
                else:
                    error_msg = result.get("message", "Unknown error")
                    st.error(f"Failed to process PDF: {error_msg}")
        
        # Show uploaded documents for current conversation
        if st.session_state.documents:
            st.markdown("#### Current Documents")
            for doc in st.session_state.documents:
                st.markdown(f"• {doc.get('filename', 'Unnamed document')}")
        
        st.divider()
        
        # Settings
        with st.expander("⚙️ Settings"):
            settings = st.session_state.settings
            
            # Chat settings
            st.checkbox("Use optimized prompts", value=settings.get("use_optimized_prompts", True), 
                       key="setting_optimized_prompts", 
                       on_change=lambda: settings.update({"use_optimized_prompts": st.session_state.setting_optimized_prompts}))
            
            st.checkbox("Use conversation memory", value=settings.get("use_memory", True), 
                       key="setting_use_memory", 
                       on_change=lambda: settings.update({"use_memory": st.session_state.setting_use_memory}))
            
            st.checkbox("Stream responses", value=settings.get("stream_responses", True), 
                       key="setting_stream_responses", 
                       on_change=lambda: settings.update({"stream_responses": st.session_state.setting_stream_responses}))
            
            # UI settings
            st.checkbox("Show sources", value=settings.get("show_sources", True), 
                       key="setting_show_sources", 
                       on_change=lambda: settings.update({"show_sources": st.session_state.setting_show_sources}))
            
            st.checkbox("Show performance metrics", value=settings.get("show_metrics", False), 
                       key="setting_show_metrics", 
                       on_change=lambda: settings.update({"show_metrics": st.session_state.setting_show_metrics}))
            
            st.checkbox("Dark mode", value=settings.get("dark_mode", ENABLE_DARK_MODE), 
                       key="setting_dark_mode", 
                       on_change=lambda: settings.update({"dark_mode": st.session_state.setting_dark_mode}))
            
            # Expert mode toggle
            st.checkbox("Expert mode", value=settings.get("expert_mode", False), 
                       key="setting_expert_mode", 
                       on_change=lambda: settings.update({"expert_mode": st.session_state.setting_expert_mode}))
            
            # Apply theme change button
            if st.button("Apply Theme"):
                st.rerun()
        
        # System Performance section
        if ENABLE_ANALYTICS:
            with st.expander("📊 System Stats"):
                if st.button("Refresh Stats"):
                    with st.spinner("Loading stats..."):
                        perf_data = api_client.get_system_performance()
                        if perf_data and "status" not in perf_data:
                            st.session_state.system_performance = perf_data
                        else:
                            st.error("Failed to load statistics")
                
                # Display stats if available
                if hasattr(st.session_state, "system_performance"):
                    perf = st.session_state.system_performance
                    
                    # Display summary metrics
                    avg_scores = perf.get("average_scores", {})
                    
                    st.markdown("##### System Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Conversations", perf.get("conversations", 0))
                        st.metric("Messages", perf.get("messages", 0))
                    with col2:
                        st.metric("Documents", perf.get("documents", 0))
                        st.metric("Overall Score", f"{avg_scores.get('overall', 0):.2f}")
                
                # Display local session metrics
                metrics = st.session_state.metrics
                st.markdown("##### Session Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Queries", metrics.get("queries", 0))
                    st.metric("Documents", metrics.get("documents", 0))
                with col2:
                    st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
        
        # About section in the sidebar footer
        st.sidebar.divider()
        st.sidebar.info(
            f"**About**  \n"
            f"{APP_TITLE} v1.0  \n"
            f"An advanced RAG system for document question answering."
        )

def render_chat_interface():
    """Render the main chat interface."""
    # Check if we need to create a new conversation
    if not st.session_state.current_conversation_id:
        create_new_conversation()
    
    # Display conversation title and header
    conv_id = st.session_state.current_conversation_id
    conv_data = st.session_state.conversations.get(conv_id, {})
    title = conv_data.get("title", "New Conversation")
    
    # Allow editing the title
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        new_title = st.text_input("Conversation Title", title, key="conv_title")
        if new_title != title:
            update_conversation_title(conv_id, new_title)
    
    with col3:
        if st.button("Clear Conversation"):
            if clear_current_conversation():
                st.success("Conversation cleared!")
                st.rerun()
    
    # Horizontal line
    st.divider()
    
    # Show feedback form if requested
    if hasattr(st.session_state, "show_feedback_form") and st.session_state.show_feedback_form:
        render_feedback_form()
    
    # Display expert mode UI if enabled
    if st.session_state.settings.get("expert_mode", False):
        render_expert_mode_ui()
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        # Display existing messages
        for message in st.session_state.messages:
            render_message(
                message,
                show_sources=st.session_state.settings.get("show_sources", True),
                show_metrics=st.session_state.settings.get("show_metrics", False)
            )
    
    # Input area
    st.divider()
    user_query = st.chat_input("Ask a question about your documents...")
    
    if user_query:
        # Add user message to the UI
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Add to chat display
        with chat_container:
            render_message({"role": "user", "content": user_query})
            
            # Create a placeholder for the response
            with st.spinner("Generating response..."):
                start_time = time.time()
                
                # Get settings
                use_optimized = st.session_state.settings.get("use_optimized_prompts", True)
                stream_response = st.session_state.settings.get("stream_responses", True)
                
                if stream_response:
                    # Get streaming response
                    response_obj = api_client.send_query(
                        query=user_query,
                        conversation_id=st.session_state.current_conversation_id,
                        use_optimized_prompts=use_optimized,
                        stream=True
                    )
                    
                    if isinstance(response_obj, requests.Response):
                        # Process streaming response
                        assistant_msg = process_streaming_response(response_obj)
                        
                        if assistant_msg:
                            # Add to session state
                            st.session_state.messages.append(assistant_msg)
                            
                            # Update metrics
                            response_time = time.time() - start_time
                            update_metrics(response_time)
                    else:
                        # Handle error
                        error_msg = response_obj.get("message", "Unknown error")
                        st.error(f"Error: {error_msg}")
                else:
                    # Get regular response
                    response = api_client.send_query(
                        query=user_query,
                        conversation_id=st.session_state.current_conversation_id,
                        use_optimized_prompts=use_optimized,
                        stream=False
                    )
                    
                    if response and "status" not in response:
                        # Create assistant message
                        assistant_msg = {
                            "role": "assistant",
                            "content": response["answer"],
                            "metadata": {
                                "retrieved_documents": response.get("retrieved_documents", []),
                                "evaluation": response.get("evaluation", {}),
                                "timings": response.get("timings", {})
                            }
                        }
                        
                        # Add to session state
                        st.session_state.messages.append(assistant_msg)
                        
                        # Display the message
                        render_message(
                            assistant_msg,
                            show_sources=st.session_state.settings.get("show_sources", True),
                            show_metrics=st.session_state.settings.get("show_metrics", False)
                        )
                        
                        # Update metrics
                        response_time = time.time() - start_time
                        update_metrics(response_time)
                    else:
                        # Handle error
                        error_msg = response.get("message", "Unknown error")
                        st.error(f"Error: {error_msg}")

def render_expert_mode_ui():
    """Render additional controls for expert mode."""
    st.markdown("### Expert Mode Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Test Retrieval"):
            query = st.session_state.messages[-1]["content"] if st.session_state.messages else "test query"
            
            with st.expander("Retrieval Test Results", expanded=True):
                with st.spinner("Testing retrieval..."):
                    # In a real app, this would call an API endpoint
                    st.markdown(f"Query: {query}")
                    st.markdown("Retrieved documents:")
                    
                    # Simulated results
                    for i in range(3):
                        st.markdown(f"**Document {i+1}**")
                        st.markdown("This is a document snippet that would be retrieved...")
    
    with col2:
        if st.button("View Query Analysis"):
            with st.expander("Query Analysis", expanded=True):
                # This would show a detailed analysis of the query
                st.markdown("### Query Understanding")
                st.markdown("- Intent: Information seeking")
                st.markdown("- Entities: [Entity1, Entity2]")
                st.markdown("- Context relevance: High")
    
    with col3:
        if st.button("Memory Inspection"):
            with st.expander("Memory Contents", expanded=True):
                st.markdown("### Conversation Memory")
                st.markdown("Recent context:")
                for i, msg in enumerate(st.session_state.messages[-5:]):
                    st.markdown(f"{msg['role']}: {msg['content'][:50]}...")

def render_analytics_dashboard():
    """Render an analytics dashboard page."""
    st.markdown("# 📊 RAG Analytics Dashboard")
    
    # Get system performance data
    with st.spinner("Loading performance data..."):
        perf_data = api_client.get_system_performance()
        
        if not perf_data or "status" in perf_data:
            st.error("Failed to load performance data")
            return
    
    # Overview metrics
    st.markdown("## System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Conversations", perf_data.get("conversations", 0))
    with col2:
        st.metric("Documents", perf_data.get("documents", 0))
    with col3:
        st.metric("Messages", perf_data.get("messages", 0))
    with col4:
        avg_scores = perf_data.get("average_scores", {})
        st.metric("Overall Score", f"{avg_scores.get('overall', 0):.2f}")
    
    # Quality metrics
    st.markdown("## Quality Metrics")
    
    metrics = perf_data.get("average_scores", {})
    
    if metrics:
        # Create DataFrame for metrics
        metrics_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Score": list(metrics.values())
        })
        
        # Filter out non-numeric and create chart
        metrics_df = metrics_df[metrics_df["Score"].apply(lambda x: isinstance(x, (int, float)))]
        
        fig = px.bar(
            metrics_df,
            x="Metric",
            y="Score",
            title="Quality Metrics",
            range_y=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Improvement suggestions
    suggestions = perf_data.get("improvement_suggestions", [])
    
    if suggestions:
        st.markdown("## Improvement Suggestions")
        
        for i, suggestion in enumerate(suggestions):
            with st.expander(f"Suggestion {i+1}: {suggestion.get('area', 'Improvement')}"):
                st.markdown(suggestion.get('suggestion', ''))
    
    # Problematic queries
    problem_queries = perf_data.get("problematic_queries", [])
    
    if problem_queries:
        st.markdown("## Problematic Queries")
        
        # Create a DataFrame
        queries_df = pd.DataFrame({"Query": problem_queries})
        
        # Display as a table
        st.dataframe(queries_df, use_container_width=True)

def render_settings_page():
    """Render a standalone settings page."""
    st.markdown("# ⚙️ Settings")
    
    # General settings
    st.markdown("## General Settings")
    
    settings = st.session_state.settings
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Chat Settings")
        
        optimized = st.checkbox(
            "Use optimized prompts",
            value=settings.get("use_optimized_prompts", True)
        )
        
        memory = st.checkbox(
            "Use conversation memory",
            value=settings.get("use_memory", True)
        )
        
        streaming = st.checkbox(
            "Stream responses",
            value=settings.get("stream_responses", True)
        )
        
        expert = st.checkbox(
            "Expert mode",
            value=settings.get("expert_mode", False)
        )
    
    with col2:
        st.markdown("### UI Settings")
        
        sources = st.checkbox(
            "Show sources",
            value=settings.get("show_sources", True)
        )
        
        metrics = st.checkbox(
            "Show performance metrics",
            value=settings.get("show_metrics", False)
        )
        
        dark_mode = st.checkbox(
            "Dark mode",
            value=settings.get("dark_mode", ENABLE_DARK_MODE)
        )
    
    # Save button
    if st.button("Save Settings"):
        settings.update({
            "use_optimized_prompts": optimized,
            "use_memory": memory,
            "stream_responses": streaming,
            "show_sources": sources,
            "show_metrics": metrics,
            "dark_mode": dark_mode,
            "expert_mode": expert
        })
        
        st.success("Settings saved!")
        time.sleep(1)
        st.rerun()
    
    # Advanced settings (if in expert mode)
    if settings.get("expert_mode", False):
        st.markdown("## Advanced Settings")
        
        st.warning("These settings are for advanced users only.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input(
                "Max tokens per request",
                min_value=100,
                max_value=4000,
                value=1000,
                step=100
            )
            
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
        
        with col2:
            st.number_input(
                "Chunk size",
                min_value=100,
                max_value=2000,
                value=500,
                step=50
            )
            
            st.slider(
                "Hybrid alpha",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Weight between dense (1.0) and sparse (0.0) retrieval"
            )
    
    # API connection settings
    with st.expander("API Connection"):
        api_url = st.text_input("API URL", value=API_URL)
        api_key = st.text_input("API Key", value=API_KEY, type="password")
        
        if st.button("Test Connection"):
            with st.spinner("Testing API connection..."):
                try:
                    response = requests.get(f"{api_url}/health")
                    if response.status_code == 200:
                        st.success("Connection successful!")
                    else:
                        st.error(f"Connection failed: Status {response.status_code}")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")

# -------------------------------------------------------------
# Main Application
# -------------------------------------------------------------
def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Apply page configuration
    set_page_configuration()
    
    # Check if user is authenticated
    if AUTH_ENABLED and not st.session_state.authenticated:
        render_login_page()
        return
    
    # Multi-page navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chat", "Analytics", "Settings"])
    
    # Render sidebar (common to all pages)
    render_sidebar()
    
    # Render selected page
    if page == "Chat":
        render_chat_interface()
    elif page == "Analytics":
        render_analytics_dashboard()
    elif page == "Settings":
        render_settings_page()

if __name__ == "__main__":
    main()