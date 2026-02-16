# 🚀 Deployment Guide

This guide walks you through deploying the NCSU Research Assistant to various platforms.

---

## ✅ Pre-Deployment Checklist

Before deploying, ensure you have:

- [ ] **API Key**: OpenAI or Anthropic API key ready
- [ ] **GitHub Account**: For repository hosting
- [ ] **Streamlit Account**: For Streamlit Community Cloud (optional)
- [ ] **All Files**: Verify all required files are present

### Required Files Checklist

```
chatbot-deploy/
├── ✅ ncsu_advanced_config_base.py
├── ✅ user_interface.py
├── ✅ app.py
├── ✅ ncsu_search_extractor.py
├── ✅ requirements.txt
├── ✅ packages.txt
├── ✅ README.md
├── ✅ .gitignore
├── ✅ NC-State-University-Logo.png
├── ✅ NC_State_Wolfpack_logo.svg.png
├── ✅ .devcontainer/devcontainer.json
├── ✅ .streamlit/secrets.toml.example
└── ✅ src/ (complete folder structure)
```

---

## 🌐 Option 1: Streamlit Community Cloud (Recommended)

### Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done):
   ```bash
   cd chatbot-deploy
   git init
   git add .
   git commit -m "Initial commit: NCSU Research Assistant"
   ```

2. **Create GitHub Repository**:
   - Go to https://github.com/new
   - Name: `ncsu-research-assistant` (or your choice)
   - Description: "AI-powered research assistant for NC State University"
   - Public or Private (your choice)
   - Don't initialize with README (we have one)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/ncsu-research-assistant.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Community

1. **Go to Streamlit Community Cloud**:
   - Visit: https://share.streamlit.io
   - Sign in with GitHub

2. **Create New App**:
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/ncsu-research-assistant`
   - Branch: `main`
   - Main file path: `user_interface.py`
   - App URL: Choose a custom URL (e.g., `ncsu-research-assistant`)

3. **Configure Secrets**:
   - Click "Advanced settings" → "Secrets"
   - Add your API key:
     ```toml
     [openai]
     api_key = "sk-your-actual-openai-api-key-here"
     ```
   - Or for Anthropic:
     ```toml
     [anthropic]
     api_key = "sk-ant-your-actual-anthropic-key-here"
     ```

4. **Deploy**:
   - Click "Deploy!"
   - Wait 2-5 minutes for deployment
   - Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

### Step 3: Test Your Deployment

1. **Open your app URL**
2. **Enter a test query**: "What are the computer science programs at NCSU?"
3. **Verify**:
   - API key is working
   - Search results appear
   - Answer is generated
   - Sources are displayed

### Troubleshooting Streamlit Deployment

**Issue: "ModuleNotFoundError"**
- Check `requirements.txt` is in the repository root
- Verify all package names are correct
- Restart the app from Streamlit dashboard

**Issue: "API key not found"**
- Go to App Settings → Secrets
- Verify the TOML format is correct
- Restart the app

**Issue: "Selenium/ChromeDriver error"**
- Verify `packages.txt` is in repository root
- Check it contains:
  ```
  chromium
  chromium-driver
  ```
- Restart the app

---

## 💻 Option 2: GitHub Codespaces

### Step 1: Push to GitHub

Follow the same steps as Option 1, Step 1 to push your code to GitHub.

### Step 2: Open in Codespaces

1. **Go to your GitHub repository**
2. **Click "Code" button** (green button)
3. **Select "Codespaces" tab**
4. **Click "Create codespace on main"**

### Step 3: Wait for Setup

The DevContainer will automatically:
- Install Python 3.11
- Install system packages (chromium, chromium-driver)
- Install Python requirements
- Start Streamlit on port 8501

This takes about 3-5 minutes.

### Step 4: Add API Key

**Option A: Through Web UI**
1. Wait for Streamlit to start
2. Click on the forwarded port 8501
3. Enter API key in the sidebar

**Option B: Create .env file**
1. In Codespaces terminal:
   ```bash
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   ```
2. Restart Streamlit:
   ```bash
   streamlit run user_interface.py
   ```

### Step 5: Test

1. Open the forwarded port (click the popup or Ports tab)
2. Test with a query
3. Verify everything works

### Troubleshooting Codespaces

**Issue: "Port 8501 not forwarded"**
- Go to "Ports" tab
- Click "Forward a Port"
- Enter: 8501
- Click the globe icon to open

**Issue: "Streamlit not starting"**
- Check terminal output for errors
- Manually start: `streamlit run user_interface.py`

---

## 🏠 Option 3: Local Development

### Step 1: Install Dependencies

```bash
cd chatbot-deploy

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Configure API Key

**Option A: .env file (recommended)**
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

**Option B: Environment variable**
```bash
# Windows PowerShell:
$env:OPENAI_API_KEY="sk-your-key-here"

# Mac/Linux:
export OPENAI_API_KEY="sk-your-key-here"
```

**Option C: Enter in Web UI**
- Leave .env empty
- Enter key in sidebar when app starts

### Step 3: Run Streamlit

```bash
streamlit run user_interface.py
```

The app will open at: http://localhost:8501

### Step 4: Test

1. Enter a test query
2. Verify results
3. Check saved files in `results/` folder

### Troubleshooting Local Development

**Issue: "Selenium WebDriver not found"**
```bash
pip install selenium webdriver-manager
```

**Issue: "Port 8501 already in use"**
```bash
# Use different port
streamlit run user_interface.py --server.port 8502
```

**Issue: "ModuleNotFoundError"**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

---

## 🐳 Option 4: Docker (Advanced)

### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "user_interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t ncsu-research-assistant .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=sk-your-key-here \
  ncsu-research-assistant
```

Access at: http://localhost:8501

---

## 📊 Post-Deployment Verification

### Test Checklist

- [ ] **App loads** without errors
- [ ] **API key** is recognized
- [ ] **Search** returns results
- [ ] **Content extraction** works
- [ ] **LLM grading** completes
- [ ] **Answer generation** produces output
- [ ] **Sources** are displayed
- [ ] **Download** button works
- [ ] **Error handling** shows helpful messages

### Performance Monitoring

**Streamlit Community Cloud:**
- Check app logs in dashboard
- Monitor resource usage
- Set up alerts for errors

**Local/Codespaces:**
- Check `logs/` folder for detailed logs
- Monitor terminal output
- Check `results/` for saved research

---

## 🔄 Updating Your Deployment

### Streamlit Community Cloud

1. **Push updates to GitHub**:
   ```bash
   git add .
   git commit -m "Update: description of changes"
   git push
   ```

2. **Automatic redeployment**:
   - Streamlit detects changes
   - Automatically redeploys
   - Takes 1-2 minutes

### GitHub Codespaces

1. **Pull latest changes**:
   ```bash
   git pull origin main
   ```

2. **Restart Streamlit**:
   - Stop current process (Ctrl+C)
   - Run: `streamlit run user_interface.py`

### Local Development

1. **Pull latest changes**:
   ```bash
   git pull origin main
   ```

2. **Update dependencies** (if requirements.txt changed):
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Restart Streamlit**

---

## 🔐 Security Best Practices

### API Keys

- ✅ **DO**: Use environment variables or secrets management
- ✅ **DO**: Rotate keys regularly
- ✅ **DO**: Use different keys for dev/prod
- ❌ **DON'T**: Commit keys to Git
- ❌ **DON'T**: Share keys in screenshots
- ❌ **DON'T**: Hardcode keys in code

### .gitignore

Ensure these are in `.gitignore`:
```
.env
.streamlit/secrets.toml
results/
logs/
*.key
*.pem
```

### Streamlit Secrets

- Only visible to app owner
- Encrypted at rest
- Not exposed in logs
- Can be updated without redeployment

---

## 📈 Scaling Considerations

### Streamlit Community Cloud Limits

- **Free tier**:
  - 1 GB RAM
  - 1 CPU core
  - Unlimited public apps
  - Sleep after inactivity

- **Recommendations**:
  - Reduce `top_k` for faster responses
  - Lower `max_pages` to save resources
  - Use `gpt-4o-mini` instead of `gpt-4o`

### Performance Optimization

**For faster responses:**
```python
config = {
    'top_k': 10,
    'max_pages': 5,
    'llm_model': 'gpt-4o-mini',
    'relevance_threshold': 0.3,
}
```

**For better quality:**
```python
config = {
    'top_k': 30,
    'max_pages': 20,
    'llm_model': 'gpt-4o',
    'relevance_threshold': 0.7,
}
```

---

## 🆘 Getting Help

### Common Issues

1. **Check logs**: Look for error messages
2. **Verify API key**: Test with a simple query
3. **Check dependencies**: Ensure all packages installed
4. **Review configuration**: Verify settings are correct

### Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **GitHub Docs**: https://docs.github.com
- **OpenAI API**: https://platform.openai.com/docs

---

## ✅ Deployment Complete!

Your NCSU Research Assistant is now live! 🎉

**Next Steps:**
1. Share the URL with users
2. Monitor usage and performance
3. Collect feedback
4. Iterate and improve

**Happy researching! 🚀 Go Pack! 🐺**
