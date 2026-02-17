# 🔧 NCSU Research Assistant - Embedded Configuration Guide

## 🎯 **How to Configure the Script**

All configuration is done by editing the `config` dictionary in the `main()` function of `ncsu_advanced_config.py`.

### **📍 Location:**
Open `ncsu_advanced_config.py` and find the configuration section around line 540:

```python
# ========================================
# 🔧 CONFIGURATION SECTION - EDIT HERE
# ========================================

config = {
    # Edit these values below
    'query': 'Your question here',
    'llm_provider': 'mock',
    # ... more settings
}
```

---

## 🔧 **Configuration Options**

### **📋 Query Configuration**
```python
'query': 'What are the computer science graduate programs at NCSU?',
```
**What it does:** The research question you want to ask about NCSU  
**Examples:**
- `'What is NCSU?'`
- `'Tell me about the engineering programs'`
- `'How do I apply for graduate school?'`

### **🤖 LLM Configuration**
```python
'llm_provider': 'mock',        # Options: 'mock', 'openai', 'anthropic'
'llm_model': 'gpt-4o',         # Model name
'llm_temperature': 0.7,        # Temperature (0.0-1.0)
'llm_max_tokens': 2000,        # Maximum tokens for responses
```

**LLM Provider Options:**
- `'mock'` - For testing (no API key needed)
- `'openai'` - Use OpenAI GPT models
- `'anthropic'` - Use Anthropic Claude models

**Popular Models:**
- OpenAI: `'gpt-4o'`, `'gpt-4'`, `'gpt-3.5-turbo'`
- Anthropic: `'claude-3-sonnet-20240229'`, `'claude-3-haiku-20240307'`

### **🔍 Search Configuration**
```python
'top_k': 10,                   # Number of search results (1-50)
'max_pages': 5,                # Max pages to extract content from (1-20)
'search_timeout': 30,          # Search timeout in seconds
'extraction_timeout': 30,      # Content extraction timeout
```

### **📊 Content Processing Configuration**
```python
'relevance_threshold': 0.6,    # Relevance threshold (0.0-1.0)
'enable_grading': True,        # Enable LLM-based content grading
'min_content_length': 100,     # Minimum content length in characters
'max_content_length': 50000,   # Maximum content length in characters
```

**Relevance Threshold Guide:**
- `0.8-1.0` - Very strict (only highly relevant content)
- `0.6-0.8` - Moderate (good balance)
- `0.3-0.6` - Lenient (includes somewhat relevant content)
- `0.0-0.3` - Very lenient (includes most content)

### **🚀 Extraction Configuration**
```python
'selenium_enabled': False,     # Enable Selenium for JavaScript pages
'enhanced_extraction': True,   # Enable enhanced extraction features
'user_agent': 'NCSU Research Assistant Bot 1.0',  # Custom User-Agent
'delay': 1.0,                  # Delay between requests in seconds
'max_retries': 3,              # Maximum retry attempts
```

### **💾 Output Configuration**
```python
'output_dir': 'results',       # Output directory for results
'save_config': True,           # Save configuration to YAML file
'save_data': True,             # Save detailed data to JSON file
'save_answer': True,           # Save answer to text file
'log_level': 'INFO',           # Logging level: DEBUG, INFO, WARNING, ERROR
```

### **🔑 API Keys (Optional)**
```python
# Uncomment and add your API keys:
# 'openai_api_key': 'sk-your-openai-key-here',
# 'anthropic_api_key': 'sk-ant-your-anthropic-key-here',
```

**Alternative:** Set environment variables:
```bash
export OPENAI_API_KEY="sk-your-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

---

## 🎯 **Common Configuration Examples**

### **Basic Research (Mock LLM):**
```python
config = {
    'query': 'What is NCSU?',
    'llm_provider': 'mock',
    'top_k': 5,
    'max_pages': 3,
    'relevance_threshold': 0.5,
    'output_dir': 'results',
}
```

### **High-Quality Research (OpenAI GPT-4):**
```python
config = {
    'query': 'What are the research opportunities in computer science?',
    'llm_provider': 'openai',
    'llm_model': 'gpt-4o',
    'llm_temperature': 0.3,
    'llm_max_tokens': 3000,
    'top_k': 15,
    'max_pages': 8,
    'relevance_threshold': 0.7,
    'enable_grading': True,
    'output_dir': 'research_results',
    # 'openai_api_key': 'sk-your-key-here',  # Uncomment and add your key
}
```

### **Fast Research (Anthropic Claude):**
```python
config = {
    'query': 'How do I apply to NCSU?',
    'llm_provider': 'anthropic',
    'llm_model': 'claude-3-haiku-20240307',
    'llm_temperature': 0.5,
    'top_k': 8,
    'max_pages': 4,
    'relevance_threshold': 0.6,
    'delay': 0.5,  # Faster scraping
    'output_dir': 'admissions_info',
    # 'anthropic_api_key': 'sk-ant-your-key-here',  # Uncomment and add your key
}
```

### **Debug Mode:**
```python
config = {
    'query': 'Tell me about NCSU faculty',
    'llm_provider': 'mock',
    'top_k': 3,
    'max_pages': 2,
    'relevance_threshold': 0.3,
    'log_level': 'DEBUG',
    'save_config': True,
    'output_dir': 'debug_results',
}
```

---

## 🚀 **How to Run**

1. **Edit Configuration:**
   - Open `ncsu_advanced_config.py`
   - Find the `config = {` section around line 540
   - Modify the values as needed

2. **Run the Script:**
   ```bash
   python ncsu_advanced_config.py
   ```

3. **Check Results:**
   - Answer: `results/answer_*.txt`
   - Data: `results/data_*.json`
   - Config: `results/config_*.yaml`

---

## 🔧 **Pro Tips**

### **For Better Results:**
- Use `'openai'` or `'anthropic'` instead of `'mock'`
- Set `relevance_threshold` to 0.7+ for high-quality content
- Increase `top_k` and `max_pages` for comprehensive research
- Use `temperature` 0.2-0.4 for factual answers

### **For Faster Processing:**
- Reduce `top_k` and `max_pages`
- Set `delay` to 0.5 or lower
- Use `'claude-3-haiku-20240307'` for fast responses

### **For Debugging:**
- Set `log_level` to `'DEBUG'`
- Enable `save_config` to see what settings were used
- Use `'mock'` provider to test without API costs

---

## ⚠️ **Important Notes**

1. **API Keys:** Required for `'openai'` and `'anthropic'` providers
2. **Rate Limits:** Respect API rate limits by adjusting `delay`
3. **Costs:** Real LLM providers charge per token - monitor usage
4. **Results:** Check the `results/` directory for all outputs

**Your NCSU research assistant is ready to use with embedded configuration!** 🎉
