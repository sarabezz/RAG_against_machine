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


## Chunking strategy

Python files are parsed with the `ast` module. Each top-level function, async function, or class definition becomes its own chunk, since these are the most semantically meaningful units in code. Any "loose" code between definitions (imports, constants, module-level statements) is grouped into its own chunk so nothing gets lost. Markdown files are split by heading boundaries (any line starting with `#` through `######`), so each section stays together. Both chunk types respect a configurable `max_chunk_size` (2000 characters by default); if a single block is too large, it's split further, falling back to raw character-level splitting for individual oversized lines. If a Python file fails to parse (`SyntaxError`), it's chunked with the Markdown chunker as a fallback instead of being skipped entirely.


### Retrieval method

Chunks are indexed using BM25 (via the `bm25s` library), a ranking algorithm that improves on plain TF-IDF by accounting for term-frequency saturation (repeating a word many times gives diminishing returns) and document-length normalization (long documents aren't unfairly favored just for containing more words). At query time, the question is tokenized the same way and matched against the index, returning the top-`k` chunks ranked by relevance score. The index (with the full corpus) is persisted to disk so it doesn't need to be rebuilt on every run.


## Performance analysis

Retrieval performance was measured using recall@k (with a 5% character-overlap threshold to count a retrieved chunk as a match), evaluated against the provided ground-truth datasets:

Documentation questions: 82.00% recall@k (target: 75%)
Code questions: 57.58% recall@k (target: 50%)

Both results comfortably exceed the subject's targets, confirming the chunking + BM25 approach retrieves relevant context reliably for both natural-language documentation and source code.

Generation performance, however, did not meet the intended latency target. On the available hardware (8GB RAM, no GPU), answer generation with Qwen3-0.6B took roughly 12–28 seconds per question, well above the ~2 second target. This is a hardware constraint rather than a pipeline design flaw: the model's own accuracy was verified to be correct (e.g., correctly answering "/v1/load_lora_adapter" for a LoRA-loading question), and an optimized inference engine (like GGUF/llama.cpp) would very likely resolve the latency gap — a trade-off documented further in "Design decisions" below.


## Design decisions

**BM25 over dense embeddings/vector search**: BM25 was chosen because it's fast, needs no GPU or extra model to encode chunks, and works well on both natural language (docs) and source code identifiers, which matched the project's constraints well.

**AST-based chunking over fixed-size chunking**: splitting by function/class/section boundaries (instead of just every N characters) keeps each chunk semantically complete, which matters both for retrieval relevance and for giving the LLM coherent context to reason over.

**Pydantic for data validation**: used to enforce the structure of data passed between pipeline stages (questions, sources, answers), catching malformed data at runtime rather than letting errors surface silently deep in the pipeline.

**Greedy decoding (`do_sample=False`) over sampling**: chosen because this is a factual QA task with one correct answer expected, not creative generation — determinism was preferred over variety.

**Transformers + PyTorch over GGUF/llama.cpp**: I was aware that a quantized GGUF model running on llama.cpp would likely perform faster on my hardware, since it's a factory-optimized format built for CPU inference. I chose to complete the project with `transformers` + raw PyTorch instead, because it's the standard, reference implementation most commonly used for research, learning, and prototyping with new models, and I wanted to fully understand how the underlying model-loading and generation process works before optimizing further.

**Reduced context size and generation length**: `max_chunk_size`, the context window passed to the LLM, and `max_new_tokens` were tuned down (e.g., context capped at 1500 characters, answers capped at ~60-80 tokens) specifically to reduce generation latency on constrained hardware, since the subject doesn't specify these values.


## Challenges faced

- **Low-RAM hardware (8GB total)**: initial attempts to use `device_map="auto"` caused the model to be offloaded to disk, making generation extremely slow (~196s per answer). Removing it and manually selecting `"cuda"`/`"cpu"` fixed that. A later attempt to speed things up further with dynamic int8 quantization gave a real 3x speedup in isolation (0.518s/token → 0.173s/token), but caused the full system to freeze when used in the actual pipeline, because holding both a float32 and int8 copy of the model spiked memory past what was available. The quantization was reverted, and the hardware limitation was documented instead of further engineered around.
- **Environment instability**: WSL crashed outright mid-project (`Wsl/Service/E_UNEXPECTED`) and its own update tool failed too, requiring a manual MSI installer download and eventually a full machine restart to recover.
- **File path mismatches**: the ground-truth dataset referenced file paths like `data/raw/vllm-0.10.1/docs/features/lora.md`, so the repo and datasets had to be placed at those exact relative paths for evaluation to work correctly.
- **Environment pollution**: extracting files on Windows left behind `:Zone.Identifier` companion files throughout the repository and datasets, which had to be found and deleted.
- **Tooling configuration**: `flake8` initially scanned third-party code inside `.venv`, fixed by excluding it via a `.flake8` config file.
- **Indentation bugs**: incorrect indentation in `__main__.py` twice caused methods to be nested inside each other instead of being sibling CLI commands, diagnosed by inspecting the raw file with `sed` and fixed by rewriting the affected method with correct spacing.


## Example usage

Indexing the repository:
```
$ uv run python -m src index
Indexado 20822 chunks en 4.66s -> data/output/index/chunks.json
Indice BM25 guardado en data/output/index/bm25
```

Asking a single question directly:
```
$ uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"
{
  "question_id": "ad-hoc",
  "question": "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?",
  "retrieved_sources": [...],
  "answer": "/v1/load_lora_adapter"
}
```

Running the full evaluation against the ground-truth dataset:
```
$ uv run python -m src evaluate data/output/search_results/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json
Preguntas evaluadas: 100
Recall@k global: 82.00%
  - easy: 91.30%
  - medium: 78.50%
  - hard: 65.00%
```


## Project Info
| | |
|---|---|
| Language | Python |
| School | 42 Madrid |
| Status | ✅ Completed |
