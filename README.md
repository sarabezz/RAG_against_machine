*This project has been created as part of the 42 curriculum by saraberm.*

# RAG_AGAINST_MACHINE

RAG against the machine is a Retrieval-Augmented Generation (RAG) system that answers natural language questions about a real codebase (vLLM), by retrieving relevant fragments of code and documentation and using a language model to generate grounded answers from them.


## About

RAG against the machine is a Retrieval-Augmented Generation (RAG) system that answers natural language questions about a real codebase (vLLM), by retrieving relevant fragments of code and documentation and using a language model to generate grounded answers from them.

The goal of this project was to give a language model access to external context — by splitting a large set of documents into small, searchable pieces, retrieving the most relevant ones for a given question, and using them to generate an accurate answer, without needing to retrain the model itself.

The pipeline works in two phases. First, the source repository is split into meaningful chunks (functions, classes, documentation sections) and indexed with BM25 for fast lookup. Then, at query time, the system takes a question, retrieves the chunks that best match it, and passes them as context to a language model, which generates the answer word by word based on that context.
