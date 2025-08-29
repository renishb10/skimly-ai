# 🤖 Skimly — AI that skims so you don’t have to

Skimly is a lightweight Streamlit app that **summarizes YouTube videos or web pages** into a concise ~400-word brief.  
It uses **LangChain** + **Hugging Face Inference** (Gemma 3 Instruct) and handles loading, splitting, and summarizing content with simple error handling.

---

## ✨ Features

- Paste a **YouTube** or **website** URL and get a summary
- Uses `YoutubeLoader` or `UnstructuredURLLoader` under the hood
- Chunking via `RecursiveCharacterTextSplitter`
- Map-Reduce summarization chain with a custom prompt
- Streamlit UI with sidebar token input and friendly error logs

---

## 🧰 Tech Stack

- **Streamlit** (UI)
- **LangChain** (`langchain`, `langchain_community`, `langchain_huggingface`)
- **Hugging Face Inference API** (LLM: `google/gemma-3-270m-it`)
- **youtube-transcript-api** (for YouTube transcripts)
- **unstructured** (via `UnstructuredURLLoader`) for web page extraction
- **python-dotenv** for environment variables

---

## 📦 Requirements

Create a `requirements.txt` like this (versions are suggestions; pin to what you use):

```txt
streamlit>=1.33.0
python-dotenv>=1.0.1
validators>=0.22.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-huggingface>=0.1.0
youtube-transcript-api>=0.6.2
unstructured>=0.14.0
```
