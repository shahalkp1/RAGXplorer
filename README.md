# RAGXplorer

**RAGXplorer** is an advanced Retrieval-Augmented Generation (RAG) system designed for dynamic knowledge retrieval and intelligent response generation. It combines local document retrieval with real-time web search and integrates a self-grading mechanism to ensure response accuracy, relevance, and quality.

<img width="1236" alt="image" src="https://github.com/user-attachments/assets/a982344a-456d-4ad1-abe8-8dc72b925811" />

### Key Features
- Local Document Retrieval: Use ChromaDB and HuggingFace embeddings for efficient knowledge retrieval from uploaded documents (PDFs, text files, etc.).
- Dynamic Web Search: Leverage Tavily to enhance responses with real-time, web-based information.
- AI Agent Framework: Powered by Pydantic.ai for type-safe agent management and structured response validation.
- LLM Integration: Utilizes Groq’s llama-3.3-70b-versatile model for natural language understanding and generation.
- Self-Grading Responses: Automatically evaluate and grade responses for relevancy, faithfulness, and context quality.
- This project is ideal for building reliable AI applications that demand a seamless combination of static document analysis and real-time dynamic data.
