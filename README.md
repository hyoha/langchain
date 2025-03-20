# local window onpremise CPU chatbot hf.co/Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M

# LangGraph Django Chatbot with Ollama & FAISS

This project is a Django-based chatbot system that integrates LangChain, LangGraph, FAISS, and AutoGen. It uses a local Ollama model to generate responses based on user questions and leverages a FAISS vector database for efficient similarity search and caching.

## Overview

The system workflow is as follows:
- **User Query**: A user submits a question via a web interface.
- **Vector Search**: The question is vectorized, and a FAISS vector database is queried to see if a similar question exists. If found, the cached answer is returned.
- **LLM Invocation**: If no similar question is found, the system loads additional context from a `data.txt` file and constructs a prompt. The local Ollama model (via LangChain) then generates a response.
- **Caching & Memory**: The new question-response pair is stored in the FAISS vector database and conversation memory for future reference.
- **Agent Framework**: AutoGen is used to initialize agent components that help manage the conversation flow, though here the focus is on direct invocation.

## Features

- **Vector Database (FAISS)**: Fast similarity search over high-dimensional vectors.
- **Local LLM Integration**: Uses a local Ollama model via LangChain.
- **Conversation Memory**: Maintains context using LangChain’s ConversationBufferMemory.
- **AutoGen Agents**: Provides an agent-based structure for managing conversation flow.
- **Caching**: Reduces latency by caching responses for repeated queries.
- **Detailed Debug Logging**: Print statements track key steps in data loading, vector search, and LLM invocation.

## Requirements

- Python 3.9
- Django 4.2.20
- langchain-ollama
- faiss-cpu
- numpy
- autogen
- (Other dependencies as specified in `requirements.txt`)

## Installation

ollama run hf.co/Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M
