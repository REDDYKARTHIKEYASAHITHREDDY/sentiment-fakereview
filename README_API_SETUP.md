# API Keys Setup Guide

This application requires API keys for full functionality. Here are two ways to configure them:

## Option 1: Using Streamlit Secrets (Recommended)

1. Create a `.streamlit` folder in your project directory (if it doesn't exist)
2. Create a file named `secrets.toml` inside `.streamlit/`
3. Add your API keys to the file:

```toml
HUGGINGFACE_API_KEY = "your_actual_huggingface_token_here"
NEWS_API_KEY = "your_actual_newsapi_key_here"
```

**Important:** Never commit `secrets.toml` to version control! It's already in `.gitignore`.

## Option 2: Using Environment Variables

### Windows (PowerShell):
```powershell
$env:HUGGINGFACE_API_KEY="your_actual_huggingface_token_here"
$env:NEWS_API_KEY="your_actual_newsapi_key_here"
```

### Windows (Command Prompt):
```cmd
set HUGGINGFACE_API_KEY=your_actual_huggingface_token_here
set NEWS_API_KEY=your_actual_newsapi_key_here
```

### Linux/Mac:
```bash
export HUGGINGFACE_API_KEY="your_actual_huggingface_token_here"
export NEWS_API_KEY="your_actual_newsapi_key_here"
```

## Getting Your API Keys

### HuggingFace API Key:
1. Go to https://huggingface.co/settings/tokens
2. Sign in or create an account
3. Click "New token"
4. Give it a name (e.g., "News Verification")
5. Select "Read" permissions
6. Copy the token

### NewsAPI Key:
1. Go to https://newsapi.org/register
2. Sign up for a free account
3. Verify your email
4. Copy your API key from the dashboard

## Testing Your Setup

After configuring your API keys, restart the Streamlit app. The warning messages should disappear if the keys are correctly configured.
