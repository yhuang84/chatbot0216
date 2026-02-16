# ⚡ Quick Start Guide

Get your NCSU Research Assistant running in 5 minutes!

---

## 🚀 Fastest Way: Streamlit Community Cloud

### 1. Fork & Push to GitHub (2 minutes)

```bash
cd chatbot-deploy
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/ncsu-research-assistant.git
git push -u origin main
```

### 2. Deploy to Streamlit (2 minutes)

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your repo → `user_interface.py`
4. Add API key in Secrets:
   ```toml
   [openai]
   api_key = "sk-your-key-here"
   ```
5. Click "Deploy!"

### 3. Start Researching! (1 minute)

- Open your app URL
- Enter a query: "What are the CS programs at NCSU?"
- Click "Start Research"
- Get comprehensive answers!

---

## 💻 Local Development

### Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Run the app
streamlit run user_interface.py
```

Open http://localhost:8501 and start researching!

---

## 🎯 Example Queries to Try

1. **"What are the computer science graduate programs at NCSU?"**
2. **"How can I get reimbursement for travel expenses as a student?"**
3. **"What scholarships are available for textile students?"**
4. **"Which faculty are doing AI research?"**
5. **"What are the prerequisites for CSC 316?"**

---

## 🔧 Configuration Tips

### For Faster Results:
- Top-K: 10
- Max Pages: 5
- Model: gpt-4o-mini

### For Better Quality:
- Top-K: 30
- Max Pages: 20
- Model: gpt-4o
- Threshold: 0.7

---

## 📚 Need More Help?

- **Full Guide**: See [README.md](README.md)
- **Deployment**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Troubleshooting**: Check README troubleshooting section

---

**That's it! You're ready to research! 🎉**
