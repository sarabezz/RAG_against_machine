import fire


class Rag:
    """CLI del proyecto RAG against the machine."""

    def index(self, max_chunk_size: int = 2000,
              repo_path: str = "data/raw/vllm-0.10.1",
              output_path: str = "data/output/index/chunks.json") -> None:
        """Ingesta y crea el indice del repositorio."""
        import json
        import time
        from pathlib import Path

        from src.ingestion import build_chunks

        start = time.time()
        chunks = build_chunks(repo_path, max_chunk_size)
        elapsed = time.time() - start

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([c._asdict() for c in chunks], f)

        print(f"Indexado {len(chunks)} chunks en {elapsed:.2f}s "
              f"-> {output_path}")

        from src.retrieval import build_index

        chunk_dicts = [c._asdict() for c in chunks]
        build_index(chunk_dicts, "data/output/index/bm25")
        print(f"Indice BM25 guardado en data/output/index/bm25")

    def search(self, query: str, k: int = 10) -> None:
        """Busca una unica pregunta."""
        from src.models import MinimalSearchResults, MinimalSource
        from src.retrieval import load_index, search_index

        retriever = load_index("data/output/index/bm25")
        results = search_index(retriever, query, k)

        sources = [
            MinimalSource(
                file_path=r["file_path"],
                first_character_index=r["first_character_index"],
                last_character_index=r["last_character_index"],
            )
            for r in results
        ]
        result = MinimalSearchResults(
            question_id="ad-hoc",
            question=query,
            retrieved_sources=sources,
        )
        print(result.model_dump_json(indent=2))

    def search_dataset(self, dataset_path: str, k: int = 10,
                        save_directory: str = "data/output/search_results"
                        ) -> None:
        """Busca todas las preguntas de un dataset JSON."""
        import json
        from pathlib import Path

        from tqdm import tqdm

        from src.models import (MinimalSearchResults, MinimalSource,
                                 StudentSearchResults)
        from src.retrieval import load_index, search_index

        with open(dataset_path, encoding="utf-8") as f:
            raw = json.load(f)

        retriever = load_index("data/output/index/bm25")

        search_results = []
        for q in tqdm(raw["rag_questions"], desc="Buscando preguntas"):
            results = search_index(retriever, q["question"], k)
            sources = [
                MinimalSource(
                    file_path=r["file_path"],
                    first_character_index=r["first_character_index"],
                    last_character_index=r["last_character_index"],
                )
                for r in results
            ]
            search_results.append(MinimalSearchResults(
                question_id=q["question_id"],
                question=q["question"],
                retrieved_sources=sources,
            ))

        output = StudentSearchResults(search_results=search_results, k=k)

        Path(save_directory).mkdir(parents=True, exist_ok=True)
        output_path = Path(save_directory) / Path(dataset_path).name
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output.model_dump_json(indent=2))

        print(f"Guardado en {output_path}")

    def answer_dataset(
        self,
        search_results_path: str,
        save_directory: str = "data/output/answers",
    ) -> None:
        """Genera respuestas a partir de resultados de busqueda."""
        print(f"TODO: answer_dataset {search_results_path} "
              f"-> {save_directory}")

    def evaluate(self, search_results_path: str,
                 ground_truth_path: str) -> None:
        """Evalua resultados de busqueda contra el ground truth."""
        import json

        from src.evaluation import evaluate_recall

        with open(search_results_path, encoding="utf-8") as f:
            search_results = json.load(f)
        with open(ground_truth_path, encoding="utf-8") as f:
            ground_truth = json.load(f)

        metrics = evaluate_recall(search_results, ground_truth)

        print(f"Preguntas evaluadas: {metrics['num_questions']}")
        print(f"Recall@k global: {metrics['recall_at_k'] * 100:.2f}%")
        for diff, val in metrics["recall_by_difficulty"].items():
            print(f"  - {diff}: {val * 100:.2f}%")

    def answer(self, query: str, k: int = 10) -> None:
        """Responde una pregunta unica con contexto recuperado."""
        from src.generation import generate_answer
        from src.models import MinimalAnswer, MinimalSource
        from src.retrieval import load_index, search_index

        retriever = load_index("data/output/index/bm25")
        results = search_index(retriever, query, k)

        sources = [
            MinimalSource(
                file_path=r["file_path"],
                first_character_index=r["first_character_index"],
                last_character_index=r["last_character_index"],
            )
            for r in results
        ]
        answer_text = generate_answer(query, results)

        result = MinimalAnswer(
            question_id="ad-hoc",
            question=query,
            retrieved_sources=sources,
            answer=answer_text,
        )
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    fire.Fire(Rag)
