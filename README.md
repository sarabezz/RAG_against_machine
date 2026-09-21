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


## Resources
### References:

- The 42 subject PDF (en.subject.pdf) — project requirements
- Hugging Face's official Qwen3-0.6B model card — to confirm how to disable "thinking mode" (enable_thinking=False)
bm25s library documentation — for indexing and searching (index, save, load, retrieve)
- Pydantic documentation — for data validation (BaseModel, Field)
- Python Fire documentation — for building the CLI
- The vLLM repository (github.com/vllm-project/vllm) — the source data being indexed and queried

### AI usage:

AI (Claude) was used throughout this project as a learning tool and pair-programming partner, since I started with no prior knowledge of RAG theory. Specifically, it was used to:

- Explain core concepts (RAG, chunking, BM25 vs TF-IDF, Pydantic validation, greedy decoding, quantization) step by step with analogies, since I had zero background in the theory.
- Write and iterate on the code together, section by section, rather than generating the whole project at once — I reviewed and tested every part against the real vLLM repository and real datasets before accepting it.
- Debug real errors encountered during development (dependency issues, indentation bugs, dtype mismatches, memory crashes from quantization on a low-RAM machine).
- Give an honest technical opinion when I had to decide between further optimizing the LLM (switching to GGUF/llama.cpp) or accepting current performance and documenting the hardware limitation.
- Check my understanding through mock interview questions, which revealed gaps I then had to re-learn correctly.

All code was tested and validated by me against the real vLLM 0.10.1 repository and the provided datasets before being included in the final project.


## System architecture

The pipeline is organized into independent modules, each handling one stage, all tied together by a CLI (`__main__.py`) built with Python Fire:

1. `chunking.py` — splits raw source files (Python and Markdown) into meaningful chunks: functions, classes, and Markdown sections, respecting a maximum chunk size.
2. `ingestion.py` — walks the entire repository, reads every `.py`/`.md` file, and calls `chunking.py` on each one to produce the full list of chunks.
3. `retrieval.py` — takes those chunks, builds a BM25 index from them (`build_index`), and later loads that index to search it (`search_index`) given a question.
4. `models.py` — defines the Pydantic schemas (`MinimalSource`, `MinimalSearchResults`, `MinimalAnswer`, etc.) that structure the data passed between every stage, and validate it.
5. `generation.py` — given a question and the retrieved chunks, builds a prompt (with the chunk text as context) and calls the Qwen3-0.6B language model to generate the final answer.
6. `evaluation.py` — compares retrieved sources against the ground-truth dataset to compute recall@k metrics.
7. `__main__.py` — exposes all of the above as CLI commands (`index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`), orchestrating the full flow: raw files → chunks → BM25 index → retrieved sources → generated answer → evaluation.








   
