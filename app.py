import streamlit as st
import os
from ncsu_advanced_config_base import NCSUAdvancedResearcher

# 🔑 Load API key from Streamlit secrets
os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["api_key"]

st.title("NCSU Research Assistant")

query = st.text_input("Enter your research query:")

if st.button("Run Research") and query:
    config = {
        "query": query,
        "llm_provider": "openai",
        "llm_model": "gpt-4.1-mini",
        "llm_temperature": 0.3,
        "llm_max_tokens": 8000,
        "top_k": 30,
        "max_pages": 5,
        "relevance_threshold": 0.1,
        "selenium_enabled": True,
        "enhanced_extraction": True,
        "output_dir": "results"
    }

    st.info("Running advanced research... please wait.")
    researcher = NCSUAdvancedResearcher(config)
    results = researcher.research(query)

    st.success("Research Complete!")
    st.subheader("Answer")
    st.write(results['final_answer'])

    st.subheader("Sources")
    for source in results['sources']:
        st.markdown(f"- [{source['title']}]({source['url']}) (Relevance: {source['relevance_score']:.2f})")
