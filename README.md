# 🔬 PubMed RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that provides research-based answers to medical questions by searching through PubMed abstracts. Built with LangChain, FAISS, and Gradio for an intuitive chat interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

## 🌟 Features

- **Scientific Accuracy**: Answers backed by peer-reviewed PubMed research
- **Real-time RAG**: Retrieves relevant abstracts and generates contextual responses
- **Interactive Interface**: Clean Gradio web interface for easy interaction
- **Efficient Search**: FAISS vector database for fast similarity search
- **Customizable**: Easily adaptable for different medical domains or datasets

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (or compatible LLM API)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SHAH-MEER/PubMed__RAG.git
   cd pubmed-rag-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

## 📋 Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
PUBMED_API_KEY=your_ncbi_api_key_here  # Optional but recommended
HUGGINGFACE_API_TOKEN=your_hf_token_here  # If using HF embeddings
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Vector Search  │───▶│   LLM Response  │
│                 │    │     (FAISS)      │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ PubMed Abstracts │
                       │   (Embeddings)   │
                       └──────────────────┘
```

### Key Components

- **Document Loader**: Fetches and processes PubMed abstracts
- **Embedding Model**: Converts text to vector representations
- **Vector Store**: FAISS index for efficient similarity search
- **LLM Chain**: Generates responses using retrieved context
- **Web Interface**: Gradio-based chat interface

## 📖 Usage Examples

### Basic Medical Query
```
User: "What are the latest treatments for Type 2 diabetes?"
Bot: Based on recent PubMed research, current treatments for Type 2 diabetes include...
```

### Drug Information
```
User: "What are the side effects of metformin?"
Bot: According to clinical studies in PubMed, metformin's side effects include...
```

### Research Synthesis
```
User: "Compare effectiveness of different COVID-19 vaccines"
Bot: Research from multiple PubMed studies shows that vaccine effectiveness varies...
```

## 🛠️ Development

### Project Structure

```
pubmed-rag-chatbot/
├── app.py                 # Main application file
├── PubMedBot              # Notebook for Traininng and testing
├── vector_db/             # Vector Database with embedings   
│   ├── index.pkl/        # Processed abstracts
│   └── index.faiss/          # FAISS indices
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Performance Metrics

- **Response Time**: < 3 seconds average
- **Accuracy**: Based on peer-reviewed sources
- **Database Size**: Configurable (default: 10K abstracts)