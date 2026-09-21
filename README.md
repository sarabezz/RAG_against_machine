*This project has been created as part of the 42 curriculum by saraberm.*

# RAG_AGAINST_MACHINE

RAG against the machine is a Retrieval-Augmented Generation (RAG) system that answers natural language questions about a real codebase (vLLM), by retrieving relevant fragments of code and documentation and using a language model to generate grounded answers from them.


## About

RAG against the machine is a Retrieval-Augmented Generation (RAG) system that answers natural language questions about a real codebase (vLLM), by retrieving relevant fragments of code and documentation and using a language model to generate grounded answers from them.

The goal of this project was to give a language model access to external context — by splitting a large set of documents into small, searchable pieces, retrieving the most relevant ones for a given question, and using them to generate an accurate answer, without needing to retrain the model itself.

The pipeline works in two phases. First, the source repository is split into meaningful chunks (functions, classes, documentation sections) and indexed with BM25 for fast lookup. Then, at query time, the system takes a question, retrieves the chunks that best match it, and passes them as context to a language model, which generates the answer word by word based on that context.


## Instructions

1. Clone the repository and enter the project folder.
2. Install dependencies:
   ```bash
   make install
   ```
3. Place the vLLM repository at `data/raw/vllm-0.10.1/` and the provided datasets at `data/datasets/` (required — the ground-truth questions reference these exact paths).
4. Build the chunk + BM25 index:
   ```bash
   uv run python -m src index
   ```
5. Search for relevant sources for a single question:
   ```bash
   uv run python -m src search "your question"
   ```
6. Get a full answer (retrieval + generation) for a single question:
   ```bash
   uv run python -m src answer "your question"
   ```
7. Run the whole dataset at once:
   ```bash
   uv run python -m src search_dataset <dataset.json>
   uv run python -m src answer_dataset <search_results.json>
   ```
8. Evaluate recall against ground truth:
   ```bash
   uv run python -m src evaluate <search_results.json> <ground_truth.json>
   ```
9. Lint the code:
    ```bash
    make lint
    ```
   (or `make lint-strict` for stricter mypy checks)
   










   
