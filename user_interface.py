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

# GitHub auto-commit function
def commit_results_to_github(file_paths, query):
    """Commit and push result files to GitHub"""
    try:
        import subprocess
        
        # Add files
        for file_path in file_paths.values():
            subprocess.run(['git', 'add', file_path], check=True, capture_output=True)
        
        # Commit
        commit_msg = f"Add research results: {query[:50]}"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True, capture_output=True)
        
        # Push
        subprocess.run(['git', 'push'], check=True, capture_output=True)
        
        return True, "Successfully committed to GitHub"
    except Exception:
        # Keep UI messages user-friendly; avoid showing raw git errors.
        return False, "GitHub sync unavailable"

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
    
    /* Make the shuffle button green and square */
    .st-key-shuffle_btn .stButton>button {
        background-color: #28a745 !important;
        min-width: 50px !important;
        max-width: 100px !important;
        height: 50px !important;
        padding: 0.5rem !important;
        font-size: 1.5em !important;
    }
    
    .st-key-shuffle_btn .stButton>button:hover {
        background-color: #218838 !important;
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
    
    /* Answer container: no frame */
    div[data-testid="stMarkdownContainer"] > div:has(h2:first-child) {
        background: transparent;
        padding: 0;
        border: none;
        box-shadow: none;
        margin: 0;
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
# NEW: store streamed answer so it survives Streamlit reruns
if 'final_answer' not in st.session_state:
    st.session_state.final_answer = ""
# Flag to trigger search from example question
if 'trigger_search' not in st.session_state:
    st.session_state.trigger_search = False
# Store the example question to search (overrides search bar)
if 'example_to_search' not in st.session_state:
    st.session_state.example_to_search = None

# Header with NC State logos on both sides
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    logo_path = os.path.join(CURRENT_DIR, "NC-State-University-Logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        st.markdown("<h1 style='text-align: center;'>🐺</h1>", unsafe_allow_html=True)

with col2:
    st.markdown("<h1 style='text-align: center; margin-top: 30px;'>NCSU Research Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1em;'>AI-Powered Research Tool for NC State University</p>", unsafe_allow_html=True)

with col3:
    wolfpack_logo_path = os.path.join(CURRENT_DIR, "NC_State_Wolfpack_logo.svg.png")
    if os.path.exists(wolfpack_logo_path):
        st.image(wolfpack_logo_path, width=150)
    else:
        st.markdown("<h1 style='text-align: center;'>🏛️</h1>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("### 🔑 API Key")
    user_api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    
    if user_api_key:
        os.environ['OPENAI_API_KEY'] = user_api_key
        st.success("✅ API Key Set")
        
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
st.markdown("### 🔍 Search")

query = st.text_area(
    "Search Query",
    value=st.session_state.query,
    height=100,
    placeholder="",
    key="query_input",
    label_visibility="collapsed"
)

st.markdown("**💡 Example Questions:**")

# Define all available example questions
import random
example_questions = [
    "Who should I contact for help with high-performance computing (HPC) at NC State University?",
    "How can a student request reimbursement for approved travel expenses?",
    "What is the course registration process at NC State University?",
    "What types of scholarships are available to students at NC State University?",
    "What does the Supply Chain Data Science Lab do at NC State University?"
]



# Initialize current example in session state
if 'current_example' not in st.session_state:
    st.session_state.current_example = random.choice(example_questions)

# Create columns for shuffle button and example question
col1, col2 = st.columns([0.8, 5.2])

with col1:
    if st.button("🔀SHUFFLE", key="shuffle_btn", type="primary"):
        st.session_state.current_example = random.choice(example_questions)
        st.rerun()

with col2:
    # Make the question clickable - clicking will search THIS question (ignores search bar)
    if st.button(f"📝 {st.session_state.current_example}", use_container_width=True, key="example_click", type="secondary"):
        st.session_state.example_to_search = st.session_state.current_example
        st.session_state.trigger_search = True
        st.rerun()

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_button = st.button("🔍 Start Research", use_container_width=True, type="primary")


# ── NEW: lightweight streaming helpers (no changes to researcher needed) ──────
def _stream_openai(prompt, model, temperature, max_tokens):
    """Generator that yields token chunks from OpenAI streaming API."""
    import openai
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    with client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

def _stream_anthropic(prompt, model, max_tokens):
    """Generator that yields token chunks from Anthropic streaming API."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ── Perform research ──────────────────────────────────────────────────────────
# Determine which query to use: example question overrides search bar
run_from_example = bool(st.session_state.trigger_search and st.session_state.example_to_search)
actual_query = st.session_state.example_to_search if run_from_example else query
answer_rendered_this_run = False

# Trigger search from either button click or example question click
if ((search_button and bool(query)) or run_from_example) and actual_query:
    # Reset flags only after we decide to run
    st.session_state.trigger_search = False
    st.session_state.example_to_search = None
    
    if not os.getenv('OPENAI_API_KEY'):
        st.error("❌ Please enter your OpenAI API key in the sidebar before starting research!")
        st.stop()
    
    st.session_state.running = True
    st.session_state.final_answer = ""  # reset previous answer
    
    # Create containers for real-time updates
    progress_container = st.container()
    
    # Create message log to accumulate all messages
    message_log = []
    
    # Create expander for message log (initially expanded)
    log_expander = st.expander("📋 Research Progress Log", expanded=True)
    log_container = log_expander.empty()
    
    # Message callback to receive updates from researcher
    def message_callback(msg: str):
        """Receive messages from researcher and accumulate in log"""
        message_log.append(msg)
        
        # Display all messages in scrollable text area
        log_text = "\n".join(message_log)
        log_container.text_area(
            "Progress",
            value=log_text,
            height=250,
            disabled=True,
            label_visibility="collapsed",
            key=f"log_{len(message_log)}"  # Unique key for each update
        )
    
    config = {
        'query': actual_query,
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
        'timeout': timeout,
        'message_callback': message_callback  # Add callback to config
    }
    
    try:
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            percentage_text = st.empty()
            
            progress_bar.progress(5)
            percentage_text.markdown("**Progress: 5%**")
            status_text.info("🔧 Initializing researcher...")
            time.sleep(0.3)
            
            researcher = NCSUAdvancedResearcher(config)
            progress_bar.progress(20)
            percentage_text.markdown("**Progress: 20%**")
            status_text.success("✅ Researcher initialized successfully")
            time.sleep(0.5)
        
        # Run research with live status updates from message_callback
        results = researcher.research(actual_query)
        
        with progress_container:
            progress_bar.progress(80)
            percentage_text.markdown("**Progress: 80%**")
            status_text.success("✅ Content analysis complete")
            time.sleep(0.5)
            
            progress_bar.progress(90)
            percentage_text.markdown("**Progress: 90% — Generating answer...**")
            status_text.info("💬 Streaming...")

        # ── Collapse the research log and show answer ──────────────────────────
        st.markdown("---")
        
        # Replace the expanded log with a collapsed version
        log_expander.empty()  # Clear the original expander
        
        
        st.markdown("## 📝 Answer")

        # Build the exact same prompt that generate_answer() would have used
        sources_text = "\n".join([
            f"=== SOURCE {i+1}: {s['title']} (Relevance: {s.get('relevance_score', 'N/A')}) ===\n"
            f"URL: {s['url']}\nContent: {s['content']}\n"
            for i, s in enumerate(results['filtered_pages'])
        ])
        prompt = f"""You are an expert research assistant. Based on the NCSU website content provided below, answer the user's question comprehensively and accurately.

USER QUESTION: {actual_query}

NCSU WEBSITE CONTENT:
{sources_text}

INSTRUCTIONS:
- Analyze all the provided content thoroughly
- Extract and synthesize relevant information to answer the question
- Provide a comprehensive, well-structured response
- Use specific details and facts from the content
- If the content contains the answer, provide it in full detail
- If the content is incomplete, mention what information is available
- Be accurate and factual - only use information from the provided content
- Organize your response logically with clear paragraphs
- Include specific details, names, dates, and facts when available
- IMPORTANT: When mentioning any webpage, department, office, tool, or resource, always include a clickable markdown hyperlink using the format [display text](URL) with the actual URL from the source content
- Link contact pages, department homepages, forms, and any other actionable resources so readers can click through directly
- For email addresses, use markdown mailto links like [email@ncsu.edu](mailto:email@ncsu.edu)
- For phone numbers, use markdown tel links like [919-515-XXXX](tel:919-515-XXXX)

COMPREHENSIVE ANSWER:"""

        # Choose streaming generator based on provider
        if llm_provider == 'openai':
            stream_gen = _stream_openai(prompt, llm_model, llm_temperature, llm_max_tokens)
        elif llm_provider == 'anthropic':
            stream_gen = _stream_anthropic(prompt, llm_model, llm_max_tokens)
        else:
            # mock: split existing answer into words and yield them
            mock_answer = researcher.answer_provider.generate_response(prompt)
            stream_gen = (w + ' ' for w in mock_answer.split())

        # st.write_stream renders each chunk live AND returns the full string
        final_answer = st.write_stream(stream_gen)
        answer_rendered_this_run = True

        # Finish progress
        with progress_container:
            progress_bar.progress(100)
            percentage_text.markdown("**Progress: 100%**")
            status_text.success("✅ Research complete!")
            time.sleep(0.8)
            progress_bar.empty()
            status_text.empty()
            percentage_text.empty()
        
        # Save answer + results
        results['final_answer'] = final_answer
        st.session_state.final_answer = final_answer
        saved_files = researcher.save_results(results)
        
        # Auto-commit to GitHub
        with st.spinner("📤 Committing results to GitHub..."):
            success, message = commit_results_to_github(saved_files, actual_query)
            if success:
                st.success(f"✅ {message}")
            else:
                st.info("Results saved locally.")
        
        st.session_state.results = results
        st.session_state.saved_files = saved_files
        st.session_state.running = False
        
    except Exception as e:
        st.error(f"❌ Error during research: {str(e)}")
        st.session_state.running = False
        
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
        
        with st.expander("🔍 Show Technical Details"):
            import traceback
            error_trace = traceback.format_exc()
            st.code(error_trace, language="python")


# ── Display persisted results on reruns ───────────────────────────────────────
if st.session_state.results and not st.session_state.running:
    results = st.session_state.results
    
    # Re-show the answer on reruns (already streamed, now just markdown)
    if st.session_state.final_answer and not search_button and not answer_rendered_this_run:
        st.markdown("---")
        st.markdown("## 📝 Answer")
        st.markdown(st.session_state.final_answer)
    
    st.markdown("---")
    
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
    
    with st.expander("📊 View Detailed Research Data"):
        st.json(results)
    
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