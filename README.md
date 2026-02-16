# 🐺 NCSU Research Assistant

A powerful AI-powered research assistant that searches NC State University's website, extracts content, grades relevance, and generates comprehensive answers using advanced LLMs.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

### 🎯 Core Capabilities
- **🔍 Smart Search**: Searches NC State's official website with configurable top-k results
- **📄 Full Content Extraction**: 100% content extraction from web pages using MarkItDown
- **🤖 AI-Powered Grading**: LLM-based content relevance scoring (0-1 scale)
- **🎚️ Configurable Filtering**: Adjustable relevance threshold for quality control
- **💬 Multiple LLM Providers**: OpenAI, Anthropic Claude, or Mock provider
- **📊 Comprehensive Results**: Detailed answers with source citations

### 🚀 Enhanced Features
- **🔗 URL Deduplication**: Automatically removes duplicate search results
- **📝 Smart Content Truncation**: Intelligently handles large content (>200k chars)
- **🔗 Rich Hyperlinks**: Clickable inline citations in markdown format
- **🌐 Windows Support**: UTF-8 encoding fix for proper emoji display
- **💾 Result Persistence**: All research saved for future reference

### 🎨 User Experience
- **🖥️ Advanced Web UI**: Full-featured Streamlit interface
- **🎨 NC State Branding**: Professional red theme with university logos
- **⚙️ Interactive Configuration**: Sidebar controls for all settings
- **📈 Progress Tracking**: Real-time status updates during research
- **🔍 Error Handling**: Detailed error messages and troubleshooting

---

## 🚀 Quick Start

### Option 1: Deploy to Streamlit Community (Recommended)

1. **Fork this repository** to your GitHub account

2. **Go to [Streamlit Community Cloud](https://share.streamlit.io)**

3. **Click "New app"** and connect your repository

4. **Configure the app:**
   - Main file: `user_interface.py`
   - Python version: 3.10+

5. **Add your API key** in the Streamlit dashboard:
   - Go to App Settings → Secrets
   - Add your OpenAI API key:
     ```toml
     [openai]
     api_key = "sk-your-openai-api-key-here"
     ```

6. **Deploy!** Your app will be live in minutes

### Option 2: GitHub Codespaces (One-Click Setup)

1. **Click "Code" → "Create codespace on main"** in your GitHub repository

2. **Wait for automatic setup** (DevContainer installs everything)

3. **Streamlit auto-starts** on port 8501

4. **Add API key** through the web UI sidebar or create `.env` file

5. **Start researching!**

### Option 3: Local Development

#### Prerequisites
- Python 3.10 or higher
- Chrome browser (for Selenium web scraping)
- OpenAI API key (recommended) or Anthropic API key

#### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd chatbot-deploy

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "OPENAI_API_KEY=sk-your-openai-api-key-here" > .env

# Run the Streamlit app
streamlit run user_interface.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📋 Configuration

### Web Interface Configuration

Use the **sidebar** in the Streamlit app to configure:

#### 🔑 API Settings
- **API Key**: Enter your OpenAI or Anthropic API key
- **Test Connection**: Verify your API key works

#### 🤖 LLM Settings
- **Provider**: OpenAI, Anthropic, or Mock
- **Model**: gpt-4o-mini, gpt-4o, claude-3-sonnet, etc.
- **Temperature**: 0.0 (deterministic) to 1.0 (creative)
- **Max Tokens**: Response length (1000-8000)

#### 🔍 Search Settings
- **Top-K Results**: Number of search results (5-50)
- **Max Pages**: Pages to extract content from (5-30)
- **Relevance Threshold**: Minimum relevance score (0.0-1.0)

#### ⚙️ Advanced Settings
- **Content Grading**: Enable/disable LLM-based grading
- **Selenium**: Enable for JavaScript-heavy pages
- **Enhanced Extraction**: More comprehensive content extraction

### Command-Line Configuration

Edit the `config` dictionary in `ncsu_advanced_config_base.py`:

```python
config = {
    'query': 'Your research question here',
    'llm_provider': 'openai',
    'llm_model': 'gpt-4o-mini',
    'llm_temperature': 0.3,
    'llm_max_tokens': 8000,
    'top_k': 20,
    'max_pages': 20,
    'relevance_threshold': 0.1,
    'enable_grading': True,
    'selenium_enabled': True,
}
```

Then run:
```bash
python ncsu_advanced_config_base.py
```

---

## 🎯 Example Queries

### Academic Programs
```
What are the computer science graduate programs at NCSU?
```

### Faculty Research
```
Which faculty are doing research on artificial intelligence?
```

### Course Information
```
What are the prerequisites for CSC 316?
```

### Admissions & Financial Aid
```
What kinds of scholarships are available for students in the College of Textiles?
```

### Student Services
```
How can I get reimbursement for my travel expenses as a student?
```

---

## 📊 Understanding the Output

### Console Output (Command-Line)
```
🔍 ADVANCED NCSU RESEARCH
📋 Query: 'Your question'
🔍 Top-K Results: 20
📊 Relevance Threshold: 0.1

📋 STEP 1: Searching NCSU website...
✅ Found 15 unique search results
🔄 Removed 2 duplicate URLs

📋 STEP 2: Extracting content from 15 pages...
✅ Extracted content from 12 pages (25,430 words)

📋 STEP 3: Grading content relevance...
✅ Graded 12 pages using LLM

📋 STEP 4: Filtering by relevance...
✅ 8 pages meet relevance threshold

📋 STEP 5: Generating LLM answer...
✅ Generated comprehensive answer
```

### Web Interface Output
- **Metrics Dashboard**: Search results, pages extracted, filtered pages, total words
- **AI-Generated Answer**: Formatted with rich hyperlinks and citations
- **Sources**: List of sources with relevance scores
- **Download Option**: Save answer as text file
- **Detailed Data**: View complete research data in JSON format

### Result Files (Saved Automatically)
- **`answer_*.txt`**: Clean, formatted answer ready to read
- **`data_*.json`**: Complete data including all URLs, content, and scores
- **`config_*.yaml`**: Settings used for this research session

---

## 🔧 Troubleshooting

### Common Issues

#### "API key not found"
```bash
# Check your .env file exists and has the right key
cat .env
# Should show: OPENAI_API_KEY=sk-...
```

**For Streamlit Cloud:**
- Go to App Settings → Secrets
- Add your API key in TOML format
- Restart the app

#### "Selenium not installed"
```bash
pip install selenium webdriver-manager
```

#### "No search results found"
- Try a broader query
- Check your internet connection
- Ensure NCSU website is accessible

#### "PDF extraction failed"
```bash
# Install PDF support
pip install markitdown[pdf]
```

#### Streamlit App Won't Start
```bash
# Check if port 8501 is in use
netstat -an | grep 8501

# Try a different port
streamlit run user_interface.py --server.port 8502
```

### Performance Tips

#### For Faster Results:
```python
'top_k': 10,           # Fewer search results
'max_pages': 5,        # Fewer pages to process
'llm_model': 'gpt-4o-mini',  # Faster model
```

#### For Better Quality:
```python
'top_k': 30,           # More search results
'max_pages': 20,       # More content
'relevance_threshold': 0.7,  # Higher quality filter
'llm_model': 'gpt-4o', # Better model
```

---

## 🏗️ Architecture

### System Flow

```
User Query → NCSU Search → Content Extraction → LLM Grading → Filtering → Answer Generation
     ↓            ↓              ↓                    ↓            ↓             ↓
  Web UI    Top-K Results   MarkItDown         Relevance     Threshold    Rich Hyperlinks
                            (100% content)      Score 0-1     Filter       + Citations
```

### Key Components

1. **NCSUScraper**: Handles search and content extraction
   - Selenium for JavaScript-rendered pages
   - MarkItDown for comprehensive content extraction
   - URL deduplication and normalization

2. **NCSUAdvancedResearcher**: Core research logic
   - LLM provider management (OpenAI, Anthropic, Mock)
   - Content grading and filtering
   - Smart content truncation
   - Answer generation with rich formatting

3. **Web Interface**: Streamlit-based UI
   - Interactive configuration
   - Progress tracking
   - Result visualization
   - NC State branding

---

## 📁 Project Structure

```
chatbot-deploy/
├── ncsu_advanced_config_base.py    # Core research engine
├── user_interface.py               # Advanced Streamlit UI
├── app.py                          # Simple Streamlit UI
├── ncsu_search_extractor.py        # Standalone search tool
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages (Chromium)
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
│
├── src/                            # Core modules
│   ├── scraper/
│   │   ├── ncsu_scraper.py        # Main scraper
│   │   ├── content_aggregator.py  # Content processing
│   │   ├── models.py              # Data models
│   │   └── ...
│   └── utils/
│       └── logger.py              # Logging utilities
│
├── .devcontainer/                  # GitHub Codespaces config
│   └── devcontainer.json
│
├── .streamlit/                     # Streamlit configuration
│   └── secrets.toml.example       # API key template
│
├── results/                        # Research results
│   ├── answer_*.txt
│   ├── data_*.json
│   └── config_*.yaml
│
└── logs/                           # Application logs
```

---

## 🔐 Security & Privacy

- **API Keys**: Never commit API keys to Git
- **Secrets Management**: Use `.env` files locally, Streamlit Secrets in cloud
- **Data Privacy**: All research data stays in your environment
- **HTTPS**: All connections to NCSU website use HTTPS
- **No Tracking**: No analytics or tracking code

---

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
flake8 .
mypy .
```

### Adding New LLM Providers

1. Create a new provider class in `ncsu_advanced_config_base.py`:
```python
class NewLLMProvider(LLMProvider):
    def __init__(self, model: str = "default-model"):
        super().__init__("new_provider", model)
        # Initialize your LLM client
    
    def generate_response(self, prompt: str) -> str:
        # Implement response generation
        pass
```

2. Add to `_setup_llm_provider()` method:
```python
elif provider_name == 'new_provider':
    return NewLLMProvider(...)
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **NC State University** for providing comprehensive web resources
- **Streamlit** for the amazing web framework
- **OpenAI & Anthropic** for powerful LLM APIs
- **MarkItDown** for excellent content extraction

---

## 📞 Support

### Having Issues?

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the [Configuration](#-configuration) guide
3. Verify your API key is correctly set
4. Ensure all dependencies are installed
5. Try with a simpler query first

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 🎉 Ready to Research!

1. **Set your API key** (`.env` file, Streamlit Secrets, or web UI)
2. **Run the app** (`streamlit run user_interface.py`)
3. **Enter your query** in the web interface
4. **Click "Start Research"**
5. **Get comprehensive answers** with sources!

**Happy researching! 🚀 Go Pack! 🐺**

---

<div align="center">
  <p><strong>Built with ❤️ for the Wolfpack</strong></p>
  <p>© 2026 NC State University Research Assistant</p>
</div>
