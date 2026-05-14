#!/usr/bin/env python3
"""
NCSU Research Assistant - Web Interface (v2, modernized)
========================================================
Design: clean, minimal, Perplexity-inspired.
- Inline [n] citations rendered as clickable domain chips
- Sources shown as a card grid with relevance bars
- Light sidebar, accent-only NC State red
"""

import streamlit as st
import os
import sys
import re
import time
import json
import random
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# Get current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔑 Load API key
try:
    os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["api_key"]
except (KeyError, FileNotFoundError, AttributeError):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

# Import the researcher
try:
    from ncsu_advanced_config_base import NCSUAdvancedResearcher
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def domain_of(url: str) -> str:
    """Extract clean domain from URL (strip www.)."""
    try:
        return urlparse(url).netloc.replace('www.', '')
    except Exception:
        return url


def favicon_url(url: str) -> str:
    """Google's favicon service — works for any public site."""
    d = domain_of(url)
    return f"https://www.google.com/s2/favicons?domain={d}&sz=32"


def render_answer_with_citations(answer_text: str, sources: list) -> str:
    """
    Replace [n] markers in the LLM's answer with clickable domain chips.
    Handles single [1] and grouped [1][2][3].
    """
    def replace_single(m):
        n = int(m.group(1)) - 1
        if 0 <= n < len(sources):
            url = sources[n]['url']
            d = domain_of(url)
            # Inline anchor styled as a chip
            return (
                f'<a href="{url}" target="_blank" class="cite-chip" '
                f'title="{sources[n]["title"]}">{d}<sup>{n+1}</sup></a>'
            )
        return m.group(0)

    # Replace each [n] individually; CSS handles tight spacing for adjacent chips
    return re.sub(r'\[(\d+)\]', replace_single, answer_text)


def commit_results_to_github(file_paths, query):
    try:
        import subprocess
        for fp in file_paths.values():
            subprocess.run(['git', 'add', fp], check=True, capture_output=True)
        subprocess.run(
            ['git', 'commit', '-m', f"Add research results: {query[:50]}"],
            check=True, capture_output=True,
        )
        subprocess.run(['git', 'push'], check=True, capture_output=True)
        return True, "Successfully committed to GitHub"
    except Exception:
        return False, "GitHub sync unavailable"


# ════════════════════════════════════════════════════════════════════
# Page config + CSS
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NCSU Research Assistant",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ────────── Design tokens ────────── */
    :root {
        --ncsu-red: #CC0000;
        --ncsu-red-soft: rgba(204, 0, 0, 0.08);
        --ncsu-red-border: rgba(204, 0, 0, 0.2);
        --bg: #fafafa;
        --surface: #ffffff;
        --border: #e5e5e7;
        --text: #1d1d1f;
        --text-muted: #6e6e73;
        --text-faint: #a1a1a6;
    }

    /* ────────── Global ────────── */
    .stApp {
        background: var(--bg);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
        color: var(--text);
    }

    /* Remove default header padding, tighten layout */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1100px;
    }

    /* ────────── Typography ────────── */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text) !important;
        font-size: 2rem !important;
    }
    h2 {
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        color: var(--text) !important;
        font-size: 1.25rem !important;
        margin-top: 2rem !important;
    }
    h3 {
        font-weight: 600 !important;
        color: var(--text) !important;
        font-size: 1rem !important;
    }

    /* ────────── Sidebar: light, recede ────────── */
    [data-testid="stSidebar"] {
        background: #f5f5f7 !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }

    /* ────────── Inputs ────────── */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stNumberInput>div>div>input {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        background: var(--surface) !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        transition: border-color 0.15s, box-shadow 0.15s;
    }
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: var(--ncsu-red) !important;
        box-shadow: 0 0 0 3px var(--ncsu-red-soft) !important;
        outline: none !important;
    }

    /* ────────── Buttons ────────── */
    .stButton>button {
        background: var(--ncsu-red);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.15s;
        box-shadow: none;
    }
    .stButton>button:hover {
        background: #b30000;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(204, 0, 0, 0.18);
    }
    .stButton>button:active {
        transform: translateY(0);
    }

    /* Secondary button (example questions) — subtle */
    .stButton>button[kind="secondary"] {
        background: var(--surface);
        color: var(--text-muted);
        border: 1px solid var(--border);
        text-align: left;
        font-weight: 400;
    }
    .stButton>button[kind="secondary"]:hover {
        background: var(--ncsu-red-soft);
        color: var(--text);
        border-color: var(--ncsu-red-border);
        transform: none;
        box-shadow: none;
    }

    /* Shuffle button — small, neutral */
    .st-key-shuffle_btn .stButton>button {
        background: var(--surface) !important;
        color: var(--text-muted) !important;
        border: 1px solid var(--border) !important;
        width: 44px !important;
        height: 44px !important;
        padding: 0 !important;
        font-size: 1.1em !important;
    }
    .st-key-shuffle_btn .stButton>button:hover {
        background: #f0f0f0 !important;
        color: var(--text) !important;
    }

    /* ────────── Expanders ────────── */
    .stExpander {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        background: var(--surface) !important;
        box-shadow: none !important;
    }
    .stExpander summary {
        font-weight: 500;
    }

    /* ────────── Alert boxes — subtle ────────── */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        box-shadow: none !important;
    }
    .stSuccess {
        background: rgba(40, 167, 69, 0.06) !important;
        border-left: 3px solid #28a745 !important;
    }
    .stInfo {
        background: rgba(0, 122, 255, 0.06) !important;
        border-left: 3px solid #007aff !important;
    }
    .stWarning {
        background: rgba(255, 149, 0, 0.06) !important;
        border-left: 3px solid #ff9500 !important;
    }
    .stError {
        background: var(--ncsu-red-soft) !important;
        border-left: 3px solid var(--ncsu-red) !important;
    }

    /* ────────── Metrics ────────── */
    [data-testid="stMetric"] {
        background: var(--surface);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
    }

    /* ────────── Progress bar ────────── */
    .stProgress > div > div > div {
        background: var(--ncsu-red) !important;
        border-radius: 4px;
    }
    .stProgress > div > div {
        background: var(--border) !important;
        border-radius: 4px;
    }

    /* ────────── Citation chips (the Perplexity-style inline cites) ────────── */
    .cite-chip {
        display: inline-flex;
        align-items: center;
        gap: 2px;
        background: var(--ncsu-red-soft);
        color: var(--ncsu-red) !important;
        padding: 1px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        text-decoration: none !important;
        border: 1px solid var(--ncsu-red-border);
        margin: 0 1px;
        vertical-align: 1px;
        transition: all 0.15s;
        white-space: nowrap;
    }
    .cite-chip:hover {
        background: var(--ncsu-red);
        color: white !important;
        border-color: var(--ncsu-red);
    }
    .cite-chip sup {
        font-size: 0.65rem;
        margin-left: 1px;
        opacity: 0.7;
    }

    /* Answer body — generous line height, no frame */
    .answer-body {
        font-size: 1.05rem;
        line-height: 1.75;
        color: var(--text);
        margin: 1rem 0 2rem 0;
    }
    .answer-body p {
        margin-bottom: 1em;
    }

    /* ────────── Source cards ────────── */
    .source-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 12px;
        margin-top: 1rem;
    }
    .source-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
        transition: all 0.15s;
        text-decoration: none !important;
        color: inherit !important;
        display: block;
    }
    .source-card:hover {
        border-color: var(--ncsu-red-border);
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        transform: translateY(-1px);
    }
    .source-card-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .source-card-favicon {
        width: 16px;
        height: 16px;
        border-radius: 3px;
    }
    .source-card-domain {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-weight: 500;
    }
    .source-card-number {
        margin-left: auto;
        font-size: 0.7rem;
        color: var(--text-faint);
        background: var(--bg);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    .source-card-title {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text);
        line-height: 1.4;
        margin-bottom: 10px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .source-card-meta {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.7rem;
        color: var(--text-faint);
    }
    .relevance-bar {
        flex: 1;
        height: 3px;
        background: var(--border);
        border-radius: 2px;
        overflow: hidden;
        margin-left: 8px;
        max-width: 80px;
    }
    .relevance-bar-fill {
        height: 100%;
        background: var(--ncsu-red);
        border-radius: 2px;
    }

    /* ────────── Header ────────── */
    .header-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 0.5rem 0 1.5rem 0;
    }
    .header-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text);
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin: 0;
    }
    .header-accent {
        width: 4px;
        height: 36px;
        background: var(--ncsu-red);
        border-radius: 2px;
    }

    /* Dividers — softer */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.5rem 0 !important;
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# Session state
# ════════════════════════════════════════════════════════════════════

if 'results' not in st.session_state: st.session_state.results = None
if 'running' not in st.session_state: st.session_state.running = False
if 'query' not in st.session_state: st.session_state.query = ""
if 'final_answer' not in st.session_state: st.session_state.final_answer = ""
if 'trigger_search' not in st.session_state: st.session_state.trigger_search = False
if 'example_to_search' not in st.session_state: st.session_state.example_to_search = None


# ════════════════════════════════════════════════════════════════════
# Header
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# Header (with logos)
# ════════════════════════════════════════════════════════════════════

logo_path = os.path.join(CURRENT_DIR, "NC-State-University-Logo.png")
wolfpack_path = os.path.join(CURRENT_DIR, "NC_State_Wolfpack_logo.svg.png")

hdr_col1, hdr_col2, hdr_col3 = st.columns([1, 4, 1], vertical_alignment="center")

with hdr_col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        st.markdown('<div style="font-size:2.5rem;text-align:center;">🐺</div>', unsafe_allow_html=True)

with hdr_col2:
    st.markdown("""
<div class="header-wrap">
<div class="header-accent"></div>
<div>
<h1 class="header-title">NCSU Research Assistant</h1>
<p class="header-subtitle">AI-powered search across NC State University</p>
</div>
</div>
""", unsafe_allow_html=True)

with hdr_col3:
    if os.path.exists(wolfpack_path):
        st.image(wolfpack_path, width=90)
    else:
        st.markdown('<div style="font-size:2.5rem;text-align:center;">🏛️</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### API")
    user_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        label_visibility="collapsed",
        help="Get a key at platform.openai.com/api-keys",
    )
    if user_api_key:
        os.environ['OPENAI_API_KEY'] = user_api_key
        st.caption("✓ Key set")

    st.markdown("### Model")
    llm_provider = st.selectbox("Provider", ["openai", "anthropic", "mock"], index=0, label_visibility="collapsed")
    llm_model = st.text_input(
        "Model",
        value="gpt-4o" if llm_provider == "openai" else "claude-3-sonnet-20240229",
        label_visibility="collapsed",
    )
    llm_temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    llm_max_tokens = st.number_input("Max tokens", 500, 4000, 1500, 100)

    st.markdown("### Search")
    top_k = st.slider("Top-K results", 5, 50, 30, 5)
    max_pages = st.slider("Max pages", 5, 30, 20, 5)
    relevance_threshold = st.slider(
        "Relevance threshold", 0.0, 1.0, 0.25, 0.05,
        help="Pages scoring below this are filtered out (unless min sources kicks in).",
    )
    min_sources = st.slider(
        "Minimum sources", 1, 15, 5, 1,
        help="Always keep at least this many top-scored sources, even if they fall below threshold.",
    )

    with st.expander("Advanced"):
        enable_grading = st.checkbox("Enable grading", value=True)
        selenium_enabled = st.checkbox("Enable Selenium", value=True)
        enhanced_extraction = st.checkbox("Enhanced extraction", value=True)
        min_content_length = st.number_input("Min content (chars)", 0, 1000, 100, 50)
        max_content_length = st.number_input("Max content (chars)", 1000, 100000, 50000, 5000)
        timeout = st.number_input("Timeout (s)", 10, 120, 30, 10)


# ════════════════════════════════════════════════════════════════════
# Search input
# ════════════════════════════════════════════════════════════════════

query = st.text_area(
    "Search Query",
    value=st.session_state.query,
    height=90,
    placeholder="Ask anything about NC State University...",
    key="query_input",
    label_visibility="collapsed",
)

# Example questions (compact, subtle)
example_questions = [
    "Who should I contact for help with high-performance computing (HPC)?",
    "How can a student request reimbursement for travel expenses?",
    "What is the course registration process?",
    "What scholarships are available to students?",
    "What does the Supply Chain Data Science Lab do?",
]

if 'current_example' not in st.session_state:
    st.session_state.current_example = random.choice(example_questions)

st.caption("Try an example")
col1, col2 = st.columns([0.5, 5.5])
with col1:
    if st.button("🔀", key="shuffle_btn", help="Shuffle example"):
        st.session_state.current_example = random.choice(example_questions)
        st.rerun()
with col2:
    if st.button(
        st.session_state.current_example,
        use_container_width=True,
        key="example_click",
        type="secondary",
    ):
        st.session_state.example_to_search = st.session_state.current_example
        st.session_state.trigger_search = True
        st.rerun()

st.markdown("")  # spacer
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_button = st.button("🔍  Search", use_container_width=True, type="primary")


# ════════════════════════════════════════════════════════════════════
# Streaming helpers
# ════════════════════════════════════════════════════════════════════

def _stream_openai(prompt, model, temperature, max_tokens):
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
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ════════════════════════════════════════════════════════════════════
# Run research
# ════════════════════════════════════════════════════════════════════

run_from_example = bool(st.session_state.trigger_search and st.session_state.example_to_search)
actual_query = st.session_state.example_to_search if run_from_example else query
answer_rendered_this_run = False

if ((search_button and bool(query)) or run_from_example) and actual_query:
    st.session_state.trigger_search = False
    st.session_state.example_to_search = None

    if not os.getenv('OPENAI_API_KEY'):
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    st.session_state.running = True
    st.session_state.final_answer = ""

    progress_container = st.container()
    message_log = []
    log_expander = st.expander("Research progress", expanded=True)
    log_container = log_expander.empty()

    def message_callback(msg: str):
        message_log.append(msg)
        log_container.text_area(
            "Progress",
            value="\n".join(message_log),
            height=220,
            disabled=True,
            label_visibility="collapsed",
            key=f"log_{len(message_log)}",
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
        'min_sources': min_sources,
        'enable_grading': enable_grading,
        'selenium_enabled': selenium_enabled,
        'enhanced_extraction': enhanced_extraction,
        'min_content_length': min_content_length,
        'max_content_length': max_content_length,
        'output_dir': 'results',
        'timeout': timeout,
        # Grading model defaults
        'grading_provider': 'openai',
        'grading_model': 'gpt-4o-mini',
        'grading_temperature': 0.0,
        'grading_max_tokens': 10,
        'message_callback': message_callback,
    }

    try:
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            progress_bar.progress(5)
            status_text.info("Initializing…")
            time.sleep(0.2)
            researcher = NCSUAdvancedResearcher(config)
            progress_bar.progress(20)
            status_text.info("Searching…")

        results = researcher.research(actual_query)

        with progress_container:
            progress_bar.progress(85)
            status_text.info("Generating answer…")

        # Collapse log, build prompt
        log_expander.empty()
        with st.expander("Research progress (completed)", expanded=False):
            st.text_area(
                "Progress",
                value="\n".join(message_log),
                height=220,
                disabled=True,
                label_visibility="collapsed",
                key="log_final",
            )

        st.markdown("## Answer")

        # Build prompt via the researcher (single source of truth)
        prompt = researcher.build_prompt(actual_query, results['filtered_pages'])

        # Choose stream
        if llm_provider == 'openai':
            stream_gen = _stream_openai(prompt, llm_model, llm_temperature, llm_max_tokens)
        elif llm_provider == 'anthropic':
            stream_gen = _stream_anthropic(prompt, llm_model, llm_max_tokens)
        else:
            mock_answer = researcher.answer_provider.generate_response(prompt)
            stream_gen = (w + ' ' for w in mock_answer.split())

        # Stream raw answer first (will show [1][2] markers live)
        raw_answer = st.write_stream(stream_gen)
        answer_rendered_this_run = True

        # Then re-render with citation chips (replaces the streamed plain version)
        # Use a placeholder above and below to keep flow tight
        sources_for_cites = results.get('sources', [])
        rendered = render_answer_with_citations(raw_answer, sources_for_cites)
        # Replace the streamed text with HTML version
        st.markdown(f'<div class="answer-body">{rendered}</div>', unsafe_allow_html=True)

        with progress_container:
            progress_bar.progress(100)
            status_text.success("Done")
            time.sleep(0.4)
            progress_bar.empty()
            status_text.empty()

        results['final_answer'] = raw_answer
        st.session_state.final_answer = raw_answer
        saved_files = researcher.save_results(results)

        with st.spinner("Saving…"):
            success, message = commit_results_to_github(saved_files, actual_query)
            if not success:
                st.caption("Results saved locally.")

        st.session_state.results = results
        st.session_state.saved_files = saved_files
        st.session_state.running = False

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.running = False
        with st.expander("Technical details"):
            import traceback
            st.code(traceback.format_exc(), language="python")


# ════════════════════════════════════════════════════════════════════
# Persisted results (after rerun)
# ════════════════════════════════════════════════════════════════════

if st.session_state.results and not st.session_state.running:
    results = st.session_state.results

    # Re-render answer with chips on reruns
    if st.session_state.final_answer and not search_button and not answer_rendered_this_run:
        st.markdown("## Answer")
        rendered = render_answer_with_citations(
            st.session_state.final_answer,
            results.get('sources', []),
        )
        st.markdown(f'<div class="answer-body">{rendered}</div>', unsafe_allow_html=True)

    # ── Sources as card grid ─────────────────────────────────────────
    # IMPORTANT: HTML must have NO leading whitespace per line, or Streamlit
    # markdown treats indented blocks as code. Keep everything left-aligned.
    sources = results.get('sources', [])
    if sources:
        st.markdown("## Sources")

        cards_parts = ['<div class="source-grid">']
        for i, s in enumerate(sources, 1):
            d = domain_of(s['url'])
            fav = favicon_url(s['url'])
            score = s.get('relevance_score', 0)
            score_pct = int(score * 100)
            title = (s['title'] or 'Untitled').replace('<', '&lt;').replace('>', '&gt;')
            url_safe = s['url'].replace('"', '&quot;')
            # Single-line per card — no leading whitespace anywhere
            card = (
                f'<a href="{url_safe}" target="_blank" class="source-card">'
                f'<div class="source-card-header">'
                f'<img src="{fav}" class="source-card-favicon" onerror="this.style.display=\'none\'">'
                f'<span class="source-card-domain">{d}</span>'
                f'<span class="source-card-number">{i}</span>'
                f'</div>'
                f'<div class="source-card-title">{title}</div>'
                f'<div class="source-card-meta">'
                f'<span>{score:.2f} relevance</span>'
                f'<div class="relevance-bar">'
                f'<div class="relevance-bar-fill" style="width: {score_pct}%"></div>'
                f'</div>'
                f'</div>'
                f'</a>'
            )
            cards_parts.append(card)
        cards_parts.append('</div>')
        # Join with newlines (not indented) — safe for markdown
        cards_html = '\n'.join(cards_parts)
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── Stats ─────────────────────────────────────────────────────────
    st.markdown("## Stats")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Searched", len(results.get('search_results', [])))
    with col2:
        st.metric("Extracted", len(results.get('extracted_pages', [])))
    with col3:
        st.metric("Relevant", len(results.get('filtered_pages', [])))
    with col4:
        total_words = sum(p.get('word_count', 0) for p in results.get('filtered_pages', []))
        st.metric("Total words", f"{total_words:,}")

    # ── Actions ───────────────────────────────────────────────────────
    if 'saved_files' in st.session_state:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            answer_file = st.session_state.saved_files.get('answer')
            if answer_file and os.path.exists(answer_file):
                with open(answer_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                st.download_button(
                    label="Download answer",
                    data=content,
                    file_name=f"ncsu_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    with st.expander("Raw research data"):
        st.json(results)


# ════════════════════════════════════════════════════════════════════
# Footer
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align: center; color: #a1a1a6; padding: 3rem 0 1rem 0; font-size: 0.8rem;'>
    NC State University · Research Assistant
</div>
""", unsafe_allow_html=True)
