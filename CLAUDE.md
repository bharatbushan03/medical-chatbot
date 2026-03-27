# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG-based medical chatbot using FastAPI, Pinecone vector store, and HuggingFace models for question-answering over medical PDF documents.

## Tech Stack

- **Backend**: FastAPI with Jinja2 templates
- **Vector Store**: Pinecone (index name: `medical-chatbot`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `google/flan-t5-small` via HuggingFace Pipeline
- **LangChain**: RAG chain with conversation history support

## Architecture

```
app.py              # Main FastAPI app: RAG chain setup, chat endpoint, session management
src/
  helper.py         # PDF loading, text splitting, embeddings download
  prompt.py         # System prompt for QA tasks
templates/
  index.html        # Frontend UI
```

**RAG Flow**: User question → History-aware retriever (reformulates question using chat history) → Pinecone similarity search → Context injection → LLM generates answer.

**Session Management**: In-memory `session_store` dict maintains `ChatMessageHistory` per `session_id` for multi-turn conversations.

## Commands

```bash
# Run the application
python app.py
# Or: uvicorn app:app --host 0.0.0.0 --port 8080

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Required in `.env`:
- `PINECONE_API_KEY` - Pinecone API key
- `HUGGINGFACE_API_TOKEN` (or `HUGGINGFACEHUB_API_TOKEN`) - HuggingFace token

## API Endpoints

- `GET /` - Serves the chat UI
- `POST /chat` - Accepts `{question, session_id}` and returns `{answer}`
