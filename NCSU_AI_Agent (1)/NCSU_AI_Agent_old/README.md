# 🎯 NCSU Research Assistant

A powerful AI-powered research assistant that searches NC State University's website, extracts content, grades relevance, and generates comprehensive answers using advanced LLMs.

## 🚀 Quick Start

### 1. **Prerequisites**
- Python 3.10+ 
- Chrome browser (for Selenium web scraping)
- OpenAI API key (recommended) or Anthropic API key

### 2. **Installation**

```bash
# Clone or download the project
cd NCSU

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for PDF support (optional)
pip install markitdown[pdf]

# Install Selenium WebDriver
pip install selenium
```

### 3. **Setup API Key**

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env
```

Add your API key to `.env`:
```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Or** for Anthropic:
```
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

### 4. **Configure Your Research Query**

Open `ncsu_advanced_config.py` and find the configuration section (around line 613):

```python
config = {
    # 📋 Query Configuration - EDIT THIS
    'query': 'Your research question here',
    
    # 🤖 LLM Configuration
    'llm_provider': 'openai',  # or 'anthropic'
    'llm_model': 'gpt-4o',
    
    # 🔍 Search Configuration
    'top_k': 20,        # Number of search results to find
    'max_pages': 20,    # Number of pages to extract content from
    
    # 📊 Content Processing
    'relevance_threshold': 0.1,  # Filter threshold (0.0-1.0)
    # ... more options
}
```

### 5. **Run the Research Assistant**

```bash
python ncsu_advanced_config.py
```

### 6. **Check Results**

Results are saved in the `results/` directory:
- **Answer**: `results/answer_*.txt` - The final AI-generated answer
- **Data**: `results/data_*.json` - Complete research data
- **Config**: `results/config_*.yaml` - Configuration used

---

## 📋 Configuration Guide

### **🔍 Search Settings**
```python
'top_k': 20,             # Max search results to find (1-50)
'max_pages': 20,         # Max pages to extract content from (1-20)
'relevance_threshold': 0.1,  # Filter low-relevance content (0.0-1.0)
```

### **🤖 LLM Settings**
```python
'llm_provider': 'openai',    # Options: 'openai', 'anthropic', 'mock'
'llm_model': 'gpt-4o',       # Model name
'llm_temperature': 0.3,      # Creativity (0.0-1.0)
'llm_max_tokens': 4000,      # Response length
```

### **⚙️ Advanced Settings**
```python
'selenium_enabled': True,     # Enable JavaScript rendering (required for NCSU)
'enhanced_extraction': True,  # Better content extraction
'enable_grading': True,       # LLM-based relevance scoring
```

---

## 🎯 Example Queries

### **Academic Programs**
```python
'query': 'What are the computer science graduate programs at NCSU?'
```

### **Faculty Research**
```python
'query': 'Which faculty are doing research on artificial intelligence?'
```

### **Course Information**
```python
'query': 'What are the prerequisites for CSC 316?'
```

### **Admissions**
```python
'query': 'How do I apply to NC State graduate school?'
```

### **Student Life**
```python
'query': 'What orientation programs are available for new students?'
```

---

## 📊 Understanding the Output

### **Console Output**
The script shows detailed progress:
```
🔍 ADVANCED NCSU RESEARCH
📋 Query: 'Your question'
🔍 Top-K Results: 20
📊 Relevance Threshold: 0.1

📋 STEP 1: Searching NCSU website...
✅ Found 15 search results

📋 STEP 2: Extracting content from 15 pages...
✅ Extracted content from 12 pages (25,430 words)

📋 STEP 3: Grading content relevance...
✅ Graded 12 pages using LLM

📋 STEP 4: Filtering by relevance...
✅ 8 pages meet relevance threshold

📋 STEP 5: Generating LLM answer...
✅ Generated comprehensive answer
```

### **Result Files**
- **`answer_*.txt`**: Clean, formatted answer ready to read
- **`data_*.json`**: Complete data including all URLs, content, and scores
- **`config_*.yaml`**: Settings used for this research session

---

## 🔧 Troubleshooting

### **Common Issues**

#### **"API key not found"**
```bash
# Check your .env file exists and has the right key
cat .env
# Should show: OPENAI_API_KEY=sk-...
```

#### **"Selenium not installed"**
```bash
pip install selenium
```

#### **"No search results found"**
- Try a broader query
- Check your internet connection
- Ensure NCSU website is accessible

#### **"PDF extraction failed"**
```bash
# Install PDF support
pip install markitdown[pdf]
```

### **Performance Tips**

#### **For Faster Results:**
```python
'top_k': 10,           # Fewer search results
'max_pages': 5,        # Fewer pages to process
'llm_model': 'gpt-3.5-turbo',  # Faster model
```

#### **For Better Quality:**
```python
'top_k': 30,           # More search results
'max_pages': 20,       # More content
'relevance_threshold': 0.7,  # Higher quality filter
'llm_model': 'gpt-4o', # Better model
```

---

## 🎯 Advanced Usage

### **Multiple Queries**
Edit the config and run multiple times:
```python
# Research session 1
'query': 'Computer science admission requirements'
# Run: python ncsu_advanced_config.py

# Research session 2  
'query': 'Engineering graduate programs'
# Run: python ncsu_advanced_config.py
```

### **Custom Output Directory**
```python
'output_dir': 'my_research_results',  # Custom folder
```

### **Debug Mode**
```python
'log_level': 'DEBUG',     # Detailed logging
'save_config': True,      # Save configuration
```

---

## 📚 What This Tool Does

1. **🔍 Searches** NC State's official website using their search engine
2. **📄 Extracts** full content from relevant pages (not just snippets)
3. **🎯 Grades** content relevance using AI (removes irrelevant pages)
4. **📊 Filters** results by your quality threshold
5. **🤖 Generates** comprehensive, cited answers using advanced LLMs
6. **💾 Saves** everything for future reference

---

## 🛡️ Features

- ✅ **100% NCSU Content** - Only searches ncsu.edu domain
- ✅ **Full Content Extraction** - Gets complete page content, not just snippets
- ✅ **AI-Powered Relevance** - LLM grades content quality
- ✅ **Comprehensive Answers** - Detailed, well-structured responses
- ✅ **Source Citations** - All answers include source URLs
- ✅ **Configurable** - Adjust search depth, quality thresholds, LLM settings
- ✅ **Persistent Results** - All research saved for future reference

---

## 🎉 Ready to Research!

1. **Set your API key** in `.env`
2. **Edit your query** in `ncsu_advanced_config.py`
3. **Run** `python ncsu_advanced_config.py`
4. **Check results** in `results/` folder

**Happy researching!** 🚀

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your API key is correctly set
3. Ensure all dependencies are installed
4. Try with a simpler query first

The tool provides detailed logging to help diagnose any problems.