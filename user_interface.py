#!/usr/bin/env python3
"""
NCSU Research Assistant - Web Interface
========================================
A beautiful web interface for the NCSU Research Assistant with NC State branding.
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import time

# Get current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔑 Load API key from Streamlit secrets or .env file
try:
    os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["api_key"]
except (KeyError, FileNotFoundError, AttributeError):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass

# Import the researcher with better error handling
try:
    from ncsu_advanced_config_base import NCSUAdvancedResearcher
except ImportError as e:
    st.error(f"""
    ❌ **Import Error:** Cannot import NCSUAdvancedResearcher
    
    **Error details:** {str(e)}
    
    **Possible causes:**
    1. Missing `src/` folder with required modules
    2. Missing dependencies in requirements.txt
    3. File structure issue
    
    **Required file structure:**
    ```
    /
    ├── user_interface.py (this file)
    ├── ncsu_advanced_config_base.py
    ├── requirements.txt
    └── src/
        ├── scraper/
        │   ├── __init__.py
        │   ├── ncsu_scraper.py
        │   ├── content_aggregator.py
        │   └── models.py
        └── utils/
            ├── __init__.py
            └── logger.py
    ```
    
    **Please ensure all files are uploaded to your repository!**
    """)
    st.stop()

# Page configuration
st.set_page_config(
    page_title="NCSU Research Assistant",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for NC State red theme
st.markdown("""
<style>
    /* NC State Red Theme */
    :root {
        --ncsu-red: #CC0000;
        --ncsu-dark-red: #990000;
        --ncsu-light-red: #FF4444;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #CC0000 !important;
        font-weight: 700 !important;
    }
    
    /* Logo styling */
    .stImage {
        border-radius: 10px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #CC0000 0%, #990000 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #CC0000;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #990000;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(204, 0, 0, 0.3);
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {
        border: 2px solid #CC0000;
        border-radius: 8px;
    }
    
    /* Cards/Containers */
    .stExpander {
        border: 2px solid #CC0000;
        border-radius: 8px;
        background-color: white;
    }
    
    /* Success/Info boxes - make them less prominent */
    .stSuccess, .stInfo {
        background-color: rgba(204, 0, 0, 0.05);
        border-left: 4px solid #CC0000;
    }
    
    /* Answer container - prominent display */
    div[data-testid="stMarkdownContainer"] > div:has(h2:first-child) {
        background: white;
        padding: 30px;
        border-radius: 12px;
        border: 3px solid #CC0000;
        box-shadow: 0 4px 20px rgba(204, 0, 0, 0.15);
        margin: 20px 0;
        font-size: 1.05em;
        line-height: 1.7;
    }
    
    /* Answer section specific styling */
    .answer-section {
        background: white;
        padding: 30px;
        border-radius: 12px;
        border: 3px solid #CC0000;
        box-shadow: 0 4px 20px rgba(204, 0, 0, 0.15);
        margin: 20px 0;
        font-size: 1.05em;
        line-height: 1.7;
    }
    
    .answer-section h1, .answer-section h2, .answer-section h3 {
        color: #CC0000 !important;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    
    .answer-section a {
        color: #CC0000;
        text-decoration: none;
        font-weight: 600;
        border-bottom: 2px solid #CC0000;
    }
    
    .answer-section a:hover {
        background-color: rgba(204, 0, 0, 0.1);
    }
    
    .answer-section table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }
    
    .answer-section table th {
        background-color: #CC0000;
        color: white;
        padding: 10px;
        text-align: left;
        border: 1px solid #CC0000;
    }
    
    .answer-section table td {
        padding: 8px;
        border: 1px solid #ddd;
    }
    
    .answer-section table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #CC0000 !important;
        font-weight: bold !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #CC0000;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'query' not in st.session_state:
    st.session_state.query = ""

# Header with NC State logos on both sides
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    # NC State University Logo (left side)
    logo_path = os.path.join(CURRENT_DIR, "NC-State-University-Logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        st.markdown("<h1 style='text-align: center;'>🐺</h1>", unsafe_allow_html=True)

with col2:
    st.markdown("<h1 style='text-align: center; margin-top: 30px;'>NCSU Research Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1em;'>AI-Powered Research Tool for NC State University</p>", unsafe_allow_html=True)

with col3:
    # Wolfpack Logo (right side)
    wolfpack_logo_path = os.path.join(CURRENT_DIR, "NC_State_Wolfpack_logo.svg.png")
    if os.path.exists(wolfpack_logo_path):
        st.image(wolfpack_logo_path, width=150)
    else:
        st.markdown("<h1 style='text-align: center;'>🏛️</h1>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key section
    st.markdown("### 🔑 API Key")
    user_api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    
    if user_api_key:
        os.environ['OPENAI_API_KEY'] = user_api_key
        st.success("✅ API Key Set")
        
        # Test API Key button
        if st.button("🧪 Test API Key"):
            try:
                import openai
                client = openai.OpenAI(api_key=user_api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                st.success("✅ API Key is valid!")
            except Exception as e:
                st.error(f"❌ API Key test failed: {str(e)}")
    else:
        st.warning("⚠️ Please enter your API key to use the chatbot")
    
    st.markdown("---")
    
    # LLM Settings
    st.markdown("### 🤖 LLM Settings")
    llm_provider = st.selectbox(
        "Provider",
        ["openai", "anthropic", "mock"],
        index=0
    )
    
    llm_model = st.text_input(
        "Model",
        value="gpt-4.1-mini" if llm_provider == "openai" else "claude-3-sonnet-20240229"
    )
    
    llm_temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower values = more deterministic, Higher values = more creative"
    )

    llm_max_tokens = st.number_input(
        "Max Tokens",
        min_value=1000,
        max_value=8000,
        value=4000,
        step=500,
        help="Maximum length of the generated answer"
    )
    
    st.markdown("---")
    
    # Search Settings
    st.markdown("### 🔍 Search Settings")
    top_k = st.slider(
        "Top-K Results",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Number of initial search results to retrieve"
    )

    max_pages = st.slider(
        "Max Pages to Extract",
        min_value=5,
        max_value=30,
        value=20,
        step=5,
        help="Maximum number of pages to extract content from"
    )

    relevance_threshold = st.slider(
        "Relevance Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Minimum relevance score for content to be included"
    )
    
    st.markdown("---")
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        enable_grading = st.checkbox("Enable Content Grading", value=True, help="Use LLM to grade content relevance")
        selenium_enabled = st.checkbox("Enable Selenium", value=True, help="For JavaScript-heavy pages")
        enhanced_extraction = st.checkbox("Enhanced Extraction", value=True, help="More comprehensive content extraction")

        with st.expander("📊 Additional Options"):
            min_content_length = st.number_input(
                "Min Content Length (chars)",
                min_value=0,
                max_value=1000,
                value=100,
                step=50
            )
            max_content_length = st.number_input(
                "Max Content Length (chars)",
                min_value=1000,
                max_value=100000,
                value=50000,
                step=5000
            )
            timeout = st.number_input(
                "Timeout (seconds)",
                min_value=10,
                max_value=120,
                value=30,
                step=10
            )

# Main content area
st.markdown("### 📝 Enter Your Research Query")

query = st.text_area(
    "What would you like to research about NC State?",
    value=st.session_state.query,
    height=100,
    placeholder="Example: How can I get reimbursement for my travel expenses as a student?",
    key="query_input"
)

# Example queries
st.markdown("**💡 Example Queries:**")
examples_col1, examples_col2, examples_col3 = st.columns(3)

with examples_col1:
    if st.button("🎓 Graduate Programs", use_container_width=True):
        st.session_state.query = "What are the computer science graduate programs at NCSU?"
        st.rerun()

with examples_col2:
    if st.button("💰 Financial Aid", use_container_width=True):
        st.session_state.query = "What kinds of scholarships are available for students?"
        st.rerun()

with examples_col3:
    if st.button("✈️ Travel Reimbursement", use_container_width=True):
        st.session_state.query = "How can I get reimbursement for my travel expenses?"
        st.rerun()

st.markdown("---")

# Research button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_button = st.button("🔍 Start Research", use_container_width=True, type="primary")

# Perform research
if search_button and query:
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        st.error("❌ Please enter your OpenAI API key in the sidebar before starting research!")
        st.stop()
    
    st.session_state.running = True
    
    # Create config
    config = {
        'query': query,
        'llm_provider': llm_provider,
        'llm_model': llm_model,
        'llm_temperature': llm_temperature,
        'llm_max_tokens': llm_max_tokens,
        'top_k': top_k,
        'max_pages': max_pages,
        'relevance_threshold': relevance_threshold,
        'enable_grading': enable_grading,
        'selenium_enabled': selenium_enabled,
        'enhanced_extraction': enhanced_extraction,
        'min_content_length': min_content_length,
        'max_content_length': max_content_length,
        'output_dir': 'results',
        'timeout': timeout
    }
    
    # Progress tracking with percentage and status messages
    progress_container = st.container()
    
    try:
        with progress_container:
            # Create progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            percentage_text = st.empty()
            
            # Step 1: Initialize (0-20%)
            progress_bar.progress(5)
            percentage_text.markdown("**Progress: 5%**")
            status_text.info("🔧 Initializing researcher...")
            time.sleep(0.3)
            
            researcher = NCSUAdvancedResearcher(config)
            progress_bar.progress(20)
            percentage_text.markdown("**Progress: 20%**")
            status_text.success("✅ Researcher initialized successfully")
            time.sleep(0.5)
            
            # Step 2: Searching (20-40%)
            progress_bar.progress(25)
            percentage_text.markdown("**Progress: 25%**")
            status_text.info("🔍 Searching NCSU website...")
            time.sleep(0.3)
            
            progress_bar.progress(40)
            percentage_text.markdown("**Progress: 40%**")
            status_text.success("✅ Search completed, found results")
            time.sleep(0.5)
            
            # Step 3: Extracting content (40-70%)
            progress_bar.progress(50)
            percentage_text.markdown("**Progress: 50%**")
            status_text.info("📄 Extracting content from pages...")
            time.sleep(0.3)
            
            progress_bar.progress(60)
            percentage_text.markdown("**Progress: 60%**")
            status_text.info("🤖 Analyzing content with AI...")
            
            # Run research (this is the main work)
            results = researcher.research(query)
            
            progress_bar.progress(80)
            percentage_text.markdown("**Progress: 80%**")
            status_text.success("✅ Content analysis complete")
            time.sleep(0.5)
            
            # Step 4: Generating answer (80-95%)
            progress_bar.progress(90)
            percentage_text.markdown("**Progress: 90%**")
            status_text.info("💾 Saving results...")
            time.sleep(0.3)
            
            # Save results
            saved_files = researcher.save_results(results)
            
            # Complete (100%)
            progress_bar.progress(100)
            percentage_text.markdown("**Progress: 100%**")
            status_text.success("✅ Research complete!")
            time.sleep(0.8)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            percentage_text.empty()
        
        # Store in session state
        st.session_state.results = results
        st.session_state.saved_files = saved_files
        st.session_state.running = False
        
    except Exception as e:
        st.error(f"❌ Error during research: {str(e)}")
        st.session_state.running = False
        
        # Show helpful error message
        st.warning("""
        ### 💡 Common Issues and Solutions:
        
        **1. API Key Issues:**
        - Verify your API key is valid and has credits
        - Check API key permissions in your OpenAI dashboard
        
        **2. Network/Access Issues:**
        - Check your internet connection
        - Some sites may block automated access
        
        **3. Selenium/ChromeDriver Issues:**
        - Ensure `packages.txt` includes chromium and chromium-driver
        - Try disabling Selenium in Advanced Settings
        
        **4. Content Issues:**
        - Try a different query
        - Reduce Top-K Results or Max Pages in settings
        """)
        
        # Show error details in expander (collapsed by default)
        with st.expander("🔍 Show Technical Details"):
            import traceback
            error_trace = traceback.format_exc()
            st.code(error_trace, language="python")

# Display results - Answer at the top!
if st.session_state.results:
    results = st.session_state.results
    
    st.markdown("---")
    
    # Show answer FIRST at the top with prominent styling
    st.markdown("## 🤖 AI-Generated Answer")
    
    # Get the answer
    final_answer = results.get('final_answer', 'No answer generated')
    
    # Display answer using native Streamlit markdown (allows tables, lists, formatting)
    # The CSS styling defined at the top will apply automatically
    st.markdown(final_answer)
    
    st.markdown("---")
    
    # Then show metrics below
    st.markdown("### 📊 Research Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔍 Search Results",
            len(results.get('search_results', []))
        )
    
    with col2:
        st.metric(
            "📄 Pages Extracted",
            len(results.get('extracted_pages', []))
        )
    
    with col3:
        st.metric(
            "✅ Pages Filtered",
            len(results.get('filtered_pages', []))
        )
    
    with col4:
        total_words = sum(p.get('word_count', 0) for p in results.get('filtered_pages', []))
        st.metric(
            "📝 Total Words",
            f"{total_words:,}"
        )
    
    # Download answer
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if 'saved_files' in st.session_state:
            answer_file = st.session_state.saved_files.get('answer')
            if answer_file and os.path.exists(answer_file):
                with open(answer_file, 'r', encoding='utf-8') as f:
                    answer_content = f.read()
                st.download_button(
                    label="📥 Download Answer",
                    data=answer_content,
                    file_name=f"ncsu_research_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    # Sources
    st.markdown("### 📚 Sources")
    
    sources = results.get('sources', [])
    if sources:
        for i, source in enumerate(sources, 1):
            with st.expander(f"📄 Source {i}: {source['title']} (Relevance: {source['relevance_score']:.2f})"):
                st.markdown(f"""
                **URL:** [{source['url']}]({source['url']})
                
                **Relevance Score:** {source['relevance_score']:.3f}
                
                **Word Count:** {source['word_count']:,} words
                """)
    else:
        st.warning("No sources found in results")
    
    # Detailed data
    with st.expander("📊 View Detailed Research Data"):
        st.json(results)
    
    # Save info
    if 'saved_files' in st.session_state:
        st.markdown("### 💾 Saved Files")
        for file_type, file_path in st.session_state.saved_files.items():
            st.code(f"{file_type.upper()}: {file_path}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>🐺 NC State University Research Assistant</strong></p>
    <p>Powered by AI | Built with ❤️ for the Wolfpack</p>
    <p style='font-size: 0.9em;'>© 2025 NC State University</p>
</div>
""", unsafe_allow_html=True)
