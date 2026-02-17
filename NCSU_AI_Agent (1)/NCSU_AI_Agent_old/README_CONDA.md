# 🎯 NCSU Research Assistant - Conda Setup Guide

A powerful AI-powered research assistant for NC State University using **Conda** environment management.

## 🚀 Quick Start with Conda

### 1. **Prerequisites**
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Chrome browser (for web scraping)
- OpenAI API key (recommended) or Anthropic API key

### 2. **Create Conda Environment**

```bash
# Navigate to project directory
cd NCSU

# Create a new conda environment with Python 3.10
conda create -n ncsuagent_env1 python=3.10 -y

# Activate the environment
conda activate ncsuagent_env1
```

### 3. **Install Dependencies**

```bash
# Install requirements from requirements.txt
pip install -r requirements.txt

# Install additional PDF support (optional but recommended)
pip install markitdown[pdf]

# Verify installation
python -c "import requests, selenium, openai; print('✅ All packages installed successfully!')"
```

### 4. **Setup API Key**

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env

# Edit the file (use your preferred editor)
nano .env
```

Add your API key to `.env`:
```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Or** for Anthropic:
```
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

Save and exit the editor.

### 5. **Configure Your Research Query**

Edit the configuration in `ncsu_advanced_config.py`:

```bash
# Open the config file
nano ncsu_advanced_config.py
```

Find the configuration section (around line 613) and edit your query:

```python
config = {
    # 📋 Query Configuration - EDIT THIS LINE
    'query': 'Your research question about NCSU here',
    
    # 🤖 LLM Configuration
    'llm_provider': 'openai',  # or 'anthropic'
    'llm_model': 'gpt-4o',
    
    # 🔍 Search Configuration
    'top_k': 20,        # Number of search results
    'max_pages': 20,    # Number of pages to analyze
    
    # 📊 Content Processing
    'relevance_threshold': 0.1,  # Quality filter (0.0-1.0)
}
```

Save and exit.

### 6. **Run the Research Assistant**

```bash
# Make sure conda environment is activated
conda activate ncsu-research

# Run the script
python ncsu_advanced_config.py
```

### 7. **Check Results**

Results are automatically saved in the `results/` directory:

```bash
# View the generated answer
ls results/
cat results/answer_*.txt
```

---

## 🔄 Daily Usage Workflow

### **Starting a Research Session**
```bash
# 1. Navigate to project
cd NCSU

# 2. Activate conda environment
conda activate ncsu-research

# 3. Edit your query (optional)
nano ncsu_advanced_config.py

# 4. Run research
python ncsu_advanced_config.py

# 5. View results
ls results/
```

### **Switching Between Projects**
```bash
# Deactivate current environment
conda deactivate

# Activate different environment
conda activate other-project

# Or reactivate NCSU research
conda activate ncsu-research
```

---

## 🛠️ Environment Management

### **List Conda Environments**
```bash
conda env list
```

### **Update Dependencies**
```bash
# Activate environment first
conda activate ncsu-research

# Update packages
pip install --upgrade -r requirements.txt
```

### **Export Environment (for sharing)**
```bash
# Export current environment
conda activate ncsu-research
conda env export > environment.yml

# Others can recreate with:
# conda env create -f environment.yml
```

### **Remove Environment (if needed)**
```bash
# Deactivate first
conda deactivate

# Remove environment
conda env remove -n ncsu-research
```

---

## 🎯 Example Research Queries

Edit the `'query'` line in `ncsu_advanced_config.py`:

### **Academic Programs**
```python
'query': 'What graduate programs are available in computer science at NCSU?'
```

### **Faculty Research**
```python
'query': 'Which NCSU faculty are researching artificial intelligence and machine learning?'
```

### **Course Prerequisites**
```python
'query': 'What are the prerequisites for CSC 316 Data Structures and Algorithms?'
```

### **Campus Resources**
```python
'query': 'What research facilities are available for engineering students?'
```

### **Admissions Information**
```python
'query': 'What are the admission requirements for the PhD program in Computer Science?'
```

---

## 📊 Understanding the Output

### **Console Progress**
```
🎯 NCSU Advanced Research Assistant
📋 Query: 'Your research question'
🔍 Top-K Results: 20
📊 Relevance Threshold: 0.1

📋 STEP 1: Searching NCSU website...
✅ Found 15 search results

🔗 SEARCH RESULT URLs:
  1. Computer Science Department
     🌐 https://csc.ncsu.edu/
  2. Graduate Programs
     🌐 https://csc.ncsu.edu/graduate/

📋 STEP 2: Extracting content from 15 pages...
✅ Extracted content from 12 pages (25,430 words)

📋 STEP 3: Grading content relevance...
✅ Graded 12 pages using LLM

📋 STEP 4: Filtering by relevance...
✅ 8 pages meet relevance threshold

📋 STEP 5: Generating LLM answer...
✅ Generated comprehensive answer

🎉 RESEARCH COMPLETE!
📄 Answer: results/answer_*.txt
```

### **Generated Files**
```bash
results/
├── answer_your_query_20240101_120000.txt    # Final AI answer
├── data_your_query_20240101_120000.json     # Complete research data  
└── config_your_query_20240101_120000.yaml   # Settings used
```

---

## 🔧 Troubleshooting

### **Environment Issues**

#### **"conda: command not found"**
```bash
# Install Miniconda or Anaconda first
# Download from: https://docs.conda.io/en/latest/miniconda.html
```

#### **"Environment not found"**
```bash
# Recreate the environment
conda create -n ncsu-research python=3.10 -y
conda activate ncsu-research
pip install -r requirements.txt
```

#### **"Package not found"**
```bash
# Make sure environment is activated
conda activate ncsu-research

# Reinstall requirements
pip install -r requirements.txt
```

### **API Key Issues**

#### **"API key not found"**
```bash
# Check .env file exists and has correct format
cat .env
# Should show: OPENAI_API_KEY=sk-...

# Make sure no extra spaces or quotes
```

#### **"Invalid API key"**
```bash
# Verify your API key at:
# OpenAI: https://platform.openai.com/api-keys
# Anthropic: https://console.anthropic.com/
```

### **Selenium Issues**

#### **"Chrome driver not found"**
```bash
# Install/update Chrome browser
# Selenium will auto-download the driver
```

#### **"Selenium timeout"**
```bash
# Check internet connection
# Try running again (sometimes NCSU website is slow)
```

---

## ⚙️ Configuration Options

### **Performance Settings**

#### **Fast Research (Quick Results)**
```python
'top_k': 10,                    # Fewer search results
'max_pages': 5,                 # Fewer pages to analyze
'llm_model': 'gpt-3.5-turbo',   # Faster model
'relevance_threshold': 0.3,     # Lower quality filter
```

#### **Comprehensive Research (Best Quality)**
```python
'top_k': 30,                    # More search results
'max_pages': 25,                # More pages to analyze
'llm_model': 'gpt-4o',          # Best model
'relevance_threshold': 0.7,     # Higher quality filter
```

#### **Debug Mode**
```python
'log_level': 'DEBUG',           # Detailed logging
'save_config': True,            # Save configuration
'save_data': True,              # Save all data
```

---

## 🚀 Advanced Conda Tips

### **Create Environment with Specific Packages**
```bash
# Create environment with common packages pre-installed
conda create -n ncsu-research python=3.10 requests beautifulsoup4 -y
conda activate ncsu-research
pip install -r requirements.txt
```

### **Use Conda-Forge Channel**
```bash
# Install packages from conda-forge (often more up-to-date)
conda install -c conda-forge python-dotenv pydantic -y
```

### **Environment Variables in Conda**
```bash
# Set environment variables permanently in conda environment
conda activate ncsu-research
conda env config vars set OPENAI_API_KEY=your-key-here
conda deactivate
conda activate ncsu-research  # Reactivate to load variables
```

---

## 📋 Complete Setup Checklist

- [ ] Install Anaconda/Miniconda
- [ ] Create conda environment: `conda create -n ncsu-research python=3.10 -y`
- [ ] Activate environment: `conda activate ncsu-research`
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Create `.env` file with API key
- [ ] Edit query in `ncsu_advanced_config.py`
- [ ] Run script: `python ncsu_advanced_config.py`
- [ ] Check results in `results/` folder

---

## 🎉 You're Ready to Research!

Your conda environment is set up and ready. The NCSU Research Assistant will:

1. **🔍 Search** NC State's website comprehensively
2. **📄 Extract** full content from relevant pages
3. **🎯 Grade** content quality using AI
4. **📊 Filter** results by relevance
5. **🤖 Generate** detailed, cited answers
6. **💾 Save** everything for future reference

**Happy researching with conda!** 🚀

---

## 📞 Need Help?

1. **Check this troubleshooting guide** first
2. **Verify conda environment** is activated
3. **Confirm API key** is set correctly
4. **Try a simple query** first to test setup
5. **Check internet connection** and NCSU website access

The tool provides detailed logging to help diagnose issues.
