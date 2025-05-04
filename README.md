# self-corrective-reranker-rag

A robust Retrieval-Augmented Generation system with self-correction capabilities powered by Voyage AI embeddings and Llama 3.

## Features

- **🔍 Smart Retrieval**: Voyage-3 embeddings for accurate semantic search
- **🔄 Intelligent Reranking**: Rerank-2-Lite for optimal document selection
- **🧠 Advanced Generation**: Llama3-8B-8192 for coherent answers
- **🔁 Self-Correction**: Auto-rephrases queries when information isn't found
- **⚠️ Error Handling**: Gracefully manages out-of-context questions

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Retrieve  │───>│   Rerank    │───>│   Generate  │───>│    Judge    │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                │
                     ┌────────────┐         ┌─────┐            │
                     │  Retrieve  │<────────┤Rephrase│<────────┘
                     └────────────┘         └─────┘
```

## Setup
Installation
bash# Clone the repository
git clone https://github.com/yourusername/self-corrective-rag.git
cd self-corrective-rag

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
Create a .env file with your API keys:

VOYAGE_API_KEY=your_voyage_api_key
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=your_langfuse_host  # Optional
```

## Usage

```python
from self_corrective_rag import invoke_rag

# Basic usage
result = invoke_rag("What is the Science of Computing?")
print(result)

# Detailed results
from self_corrective_rag import process_query
result = process_query("What is the Science of Computing?")
print(f"Relevant: {result['is_relevant']}, Retries: {result['retry_count']}")
```

## How It Works

1. **Retrieval**: Embeds query with Voyage-3 and searches Qdrant
2. **Reranking**: Prioritizes relevant content with Rerank-2-Lite
3. **Generation**: Creates structured answers with Llama3-8B-8192
4. **Validation**: Checks if answer is relevant to query
5. **Self-Correction**: Rephrases query and retries if needed

When information isn't found even after rephrasing, the system provides a clear message rather than returning potentially irrelevant content.

## Performance Tips

- Adjust `limit` in retrieval for recall vs. speed
- Modify `top_k` in reranking to control precision
- Change LLM temperature to balance creativity and factuality
