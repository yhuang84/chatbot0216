#!/usr/bin/env python3
"""
NCSU Research Assistant - Web Interface with Streaming
=======================================================
Uses st.write_stream() to display the AI answer word-by-word,
exactly like ChatGPT / Claude interfaces.
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load API key
try:
    os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["api_key"]
except (KeyError, FileNotFoundError, AttributeError):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

# Import researcher
try:
    from ncsu_advanced_config_base_optimized import NCSUAdvancedResearcher
except ImportError as e:
    st.error(f"""
    ❌ **Import Error:** Cannot import NCSUAdvancedResearcher

    **Error:** {str(e)}

    Make sure `ncsu_advanced_config_base_optimized.py` is in the same folder
    and the `src/` directory is present with all required modules.
    """)
    st.stop()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NCSU Research Assistant",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── NC State CSS theme ───────────────────────────────────────────────────────
st.markdown("""
<style>
    :root { --ncsu-red: #CC0000; }

    .stApp { background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%); }

    h1, h2, h3 { color: #CC0000 !important; font-weight: 700 !important; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #CC0000 0%, #990000 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }

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
        box-shadow: 0 4px 12px rgba(204,0,0,0.3);
    }

    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {
        border: 2px solid #CC0000;
        border-radius: 8px;
    }

    [data-testid="stMetricValue"] {
        color: #CC0000 !important;
        font-weight: bold !important;
    }

    /* Answer streaming container */
    .answer-box {
        background: white;
        border: 3px solid #CC0000;
        border-radius: 12px;
        padding: 28px 32px;
        box-shadow: 0 4px 20px rgba(204,0,0,0.12);
        font-size: 1.05em;
        line-height: 1.75;
        margin: 16px 0 24px 0;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ('results', None),
    ('final_answer', ''),
    ('query', ''),
    ('saved_files', {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    logo = os.path.join(CURRENT_DIR, "NC-State-University-Logo.png")
    if os.path.exists(logo):
        st.image(logo, width=150)
    else:
        st.markdown("<h1 style='text-align:center'>🐺</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='text-align:center;margin-top:30px'>NCSU Research Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#666;font-size:1.1em'>AI-Powered Research Tool for NC State University</p>", unsafe_allow_html=True)
with col3:
    wolfpack = os.path.join(CURRENT_DIR, "NC_State_Wolfpack_logo.svg.png")
    if os.path.exists(wolfpack):
        st.image(wolfpack, width=150)
    else:
        st.markdown("<h1 style='text-align:center'>🏛️</h1>", unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────────
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
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                st.success("✅ API Key is valid!")
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
    else:
        st.warning("⚠️ Please enter your API key")

    st.markdown("---")

    st.markdown("### 🤖 LLM Settings")
    llm_provider = st.selectbox("Provider", ["openai", "anthropic", "mock"], index=0)
    llm_model = st.text_input(
        "Model",
        value="gpt-4o" if llm_provider == "openai" else "claude-3-sonnet-20240229"
    )
    llm_temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    llm_max_tokens = st.number_input("Max Tokens", 1000, 8000, 4000, 500)

    st.markdown("---")

    st.markdown("### 🔍 Search Settings")
    top_k = st.slider("Top-K Results", 5, 50, 20, 5)
    max_pages = st.slider("Max Pages to Extract", 5, 30, 20, 5)
    relevance_threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.1, 0.1)

    st.markdown("---")

    with st.expander("⚙️ Advanced Settings"):
        enable_grading = st.checkbox("Enable Content Grading", value=True)
        selenium_enabled = st.checkbox("Enable Selenium", value=True)
        enhanced_extraction = st.checkbox("Enhanced Extraction", value=True)
        min_content_length = st.number_input("Min Content Length (chars)", 0, 1000, 100, 50)
        max_content_length = st.number_input("Max Content Length (chars)", 1000, 100000, 50000, 5000)
        timeout = st.number_input("Timeout (seconds)", 10, 120, 30, 10)

# ── Query input ────────────────────────────────────────────────────────────────
st.markdown("### 📝 Enter Your Research Query")

query = st.text_area(
    "What would you like to research about NC State?",
    value=st.session_state.query,
    height=100,
    placeholder="Example: What are the requirements for the Computer Science major?",
    key="query_input"
)

# Example query buttons
st.markdown("**💡 Example Queries:**")
ex1, ex2, ex3 = st.columns(3)
with ex1:
    if st.button("🎓 Graduate Programs", use_container_width=True):
        st.session_state.query = "What are the computer science graduate programs at NCSU?"
        st.rerun()
with ex2:
    if st.button("💰 Financial Aid", use_container_width=True):
        st.session_state.query = "What kinds of scholarships are available for students?"
        st.rerun()
with ex3:
    if st.button("✈️ Travel Reimbursement", use_container_width=True):
        st.session_state.query = "How can I get reimbursement for my travel expenses?"
        st.rerun()

st.markdown("---")

# ── Search button ──────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    search_button = st.button("🔍 Start Research", use_container_width=True, type="primary")

# ── Main research flow ─────────────────────────────────────────────────────────
if search_button and query:
    if not os.getenv('OPENAI_API_KEY') and llm_provider == 'openai':
        st.error("❌ Please enter your OpenAI API key in the sidebar!")
        st.stop()

    config = {
        'query': query,
        'llm_provider': llm_provider,
        'llm_model': llm_model,
        'llm_temperature': llm_temperature,
        'llm_max_tokens': llm_max_tokens,
        'max_context_tokens': 120000,
        'grading_provider': llm_provider,
        'grading_model': 'gpt-4o-mini' if llm_provider == 'openai' else 'claude-3-haiku-20240307',
        'grading_temperature': 0.3,
        'grading_max_tokens': 10,
        'max_grading_content_length': 2000,
        'top_k': top_k,
        'max_pages': max_pages,
        'relevance_threshold': relevance_threshold,
        'enable_grading': enable_grading,
        'parallel_extraction': True,
        'extraction_workers': 5,
        'parallel_grading': True,
        'grading_workers': 5,
        'enable_caching': True,
        'enable_early_stopping': True,
        'early_stop_threshold': 0.85,
        'early_stop_min_pages': 3,
        'selenium_enabled': selenium_enabled,
        'enhanced_extraction': enhanced_extraction,
        'min_content_length': min_content_length,
        'max_content_length': max_content_length,
        'output_dir': 'results',
        'timeout': timeout,
    }

    # ── Phase 1: Research (search + extract + grade + filter) ─────────────────
    st.markdown("---")
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.info("🔧 Initializing researcher...")
        progress_bar.progress(5)
        researcher = NCSUAdvancedResearcher(config)

        status_text.info("🔍 Searching NCSU website...")
        progress_bar.progress(15)

        status_text.info("📄 Extracting & analyzing pages...")
        progress_bar.progress(30)

        results = researcher.research(query)   # ← all steps except answer gen

        progress_bar.progress(75)
        status_text.info("💡 Generating answer (streaming)...")

        # ── Phase 2: Stream the answer into the UI ────────────────────────────
        st.markdown("## 🤖 AI-Generated Answer")

        # st.write_stream() accepts a generator and renders each chunk
        # as it arrives — exactly like ChatGPT word-by-word output.
        # It also returns the full concatenated string when done.
        answer_placeholder = st.empty()
        with answer_placeholder.container():
            full_answer = st.write_stream(
                researcher.get_answer_stream(query, results['filtered_pages'])
            )

        progress_bar.progress(95)
        status_text.info("💾 Saving results...")

        results['final_answer'] = full_answer
        saved_files = researcher.save_results(results)

        progress_bar.progress(100)
        status_text.success("✅ Research complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        # Persist to session state
        st.session_state.results = results
        st.session_state.final_answer = full_answer
        st.session_state.saved_files = saved_files

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error during research: {str(e)}")
        with st.expander("🔍 Show Technical Details"):
            import traceback
            st.code(traceback.format_exc(), language="python")
        st.stop()

# ── Show persisted results (after page re-runs) ────────────────────────────────
elif st.session_state.results:
    results = st.session_state.results

    st.markdown("---")
    st.markdown("## 🤖 AI-Generated Answer")

    # Re-display the already-generated answer (not re-streamed)
    st.markdown(
        f'<div class="answer-box">{st.session_state.final_answer}</div>',
        unsafe_allow_html=True
    )

# ── Stats & sources (shown after any research run) ────────────────────────────
if st.session_state.results:
    results = st.session_state.results

    st.markdown("---")
    st.markdown("### 📊 Research Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔍 Search Results",   len(results.get('search_results', [])))
    c2.metric("📄 Pages Extracted",  len(results.get('extracted_pages', [])))
    c3.metric("✅ Pages Filtered",   len(results.get('filtered_pages', [])))
    total_words = sum(p.get('word_count', 0) for p in results.get('filtered_pages', []))
    c4.metric("📝 Total Words", f"{total_words:,}")

    # Download button
    _, dl_col, _ = st.columns([1, 1, 1])
    with dl_col:
        if st.session_state.saved_files:
            answer_file = st.session_state.saved_files.get('answer')
            if answer_file and os.path.exists(answer_file):
                with open(answer_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                st.download_button(
                    label="📥 Download Answer",
                    data=content,
                    file_name=f"ncsu_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # Sources
    st.markdown("### 📚 Sources")
    for i, source in enumerate(results.get('sources', []), 1):
        with st.expander(f"📄 Source {i}: {source['title']} — Relevance: {source['relevance_score']:.2f}"):
            st.markdown(f"""
**URL:** [{source['url']}]({source['url']})

**Relevance Score:** {source['relevance_score']:.3f}

**Word Count:** {source['word_count']:,} words
            """)

    with st.expander("📊 View Raw Research Data"):
        st.json(results)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;padding:20px'>
    <p><strong>🐺 NC State University Research Assistant</strong></p>
    <p>Powered by AI | Built with ❤️ for the Wolfpack</p>
    <p style='font-size:0.9em'>© 2025 NC State University</p>
</div>
""", unsafe_allow_html=True)
