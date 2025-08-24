# Agentic RAG Chatbot

This project implements an AI chatbot that combines Retrieval-Augmented Generation (RAG) with an agentic approach using LangChain, Chroma DB and Streamlit. The chatbot can retrieve relevant information from a knowledge base and provide accurate responses based on the retrieved content.

## Features

- RAG-based information retrieval system
- Integration with Google's Generative AI (Gemini)
- Interactive Streamlit web interface
- Document processing and chunking
- Vector-based similarity search
- Persistent vector store using Chroma

## Project Structure

```
├── RAG with Langgraph.ipynb    # Jupyter notebook with RAG implementation
├── rag_streamlit.py           # Main Streamlit application
├── requirements.txt           # Project dependencies
└── README.md                 # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/mehalr/agentic-rag-chatbot.git
cd agentic-rag-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your Google AI API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Running the Application

To run the Streamlit application:

```bash
streamlit run rag_streamlit.py
```

The application will be available at `http://localhost:8501`

## Features Implementation

### Vector Store
- Uses Chroma as the vector store
- Documents are embedded using Google's Generative AI embeddings model
- Persistent storage in `./chroma_langchain_db`

### RAG Implementation
- Documents are split using RecursiveCharacterTextSplitter
- Similarity search is performed to retrieve relevant context
- Retrieved information is used to enhance the AI's responses

### Agent System
- Uses LangChain's agent system for tool-based interactions
- Custom retrieval tool for accessing the knowledge base
- Maintains chat history for context-aware responses

## Dependencies

- streamlit
- python-dotenv
- langchain-google-genai
- langchain-core
- bs4 (BeautifulSoup4)
- langchain-community
- langchain-text-splitters
- chromadb
- nest-asyncio
