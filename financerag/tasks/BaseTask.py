import csv
import json
import os
import logging
from typing import Optional, List, Dict, Callable, Tuple, Any

from pydantic import BaseModel

from financerag.common import HFDataLoader, Retrieval, Reranker, Generator
from financerag.tasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)


# Adapted from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/AbsTask.py
class BaseTask(BaseModel):
    metadata: TaskMetadata
    queries: Optional[Dict[str, str]] = None
    corpus: Optional[Dict[str, Dict[str, str]]] = None
    data_loaded: bool = False
    retrieve_results: Optional[Dict] = None
    rerank_results: Optional[Dict] = None
    generate_results: Optional[Dict] = None

    def model_post_init(self, __context: Any) -> None:
        self.load_data()

    @property
    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries = {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]
        subset = self.metadata_dict["dataset"]["subset"]

        corpus, queries = HFDataLoader(
            hf_repo=dataset_path,
            subset=subset,
            streaming=False,
            keep_in_memory=False,
        ).load()
        # Conversion from DataSet
        self.queries = {query["id"]: query["text"] for query in queries}
        self.corpus = {
            doc["id"]: {"title": doc["title"], "text": doc["text"]}
            for doc in corpus
        }

        self.data_loaded = True

    def retrieve(
            self,
            retriever: Retrieval,
            top_k: Optional[int] = 100,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        if not isinstance(retriever, Retrieval):
            raise AttributeError("model must implement the `Retrieval` protocol")
        self.retrieve_results = retriever.retrieve(queries=self.queries, corpus=self.corpus, top_k=top_k, **kwargs)

        return self.results

    def rerank(
            self,
            reranker: Reranker,
            results: Optional[Dict] = None,
            top_k: Optional[int] = 100,
            batch_size: Optional[int] = None,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:

        if not isinstance(reranker, Reranker):
            raise AttributeError("model must implement the `Reranker` protocol")

        self.rerank_results = reranker.rerank(
            queries=self.queries,
            corpus=self.corpus,
            results=results,
            top_k=top_k,
            batch_size=batch_size,
            **kwargs)

        return self.results

    def generate(
            self,
            model: Generator,
            results: Optional[Dict] = None,
            prepare_messages: Optional[Callable] = None,
            **kwargs
    ) -> Dict[str, str]:
        if not isinstance(model, Generator):
            raise AttributeError("model must implement the `Generator` protocol")

        if prepare_messages is None:
            logger.info(
                "No prepare_messages function provided. "
                "Using default message preparation function, which selects the highest scored document for each query."
            )

            def default_messages(query: str, documents: List[Tuple[str, float]]) -> List[Dict]:
                first_document = max(documents, key=lambda x: x[1])[0]
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Document: {first_document}"
                                                f"\nGenerate an answer to the question from the document."
                                                f"\nQuestion: {query}"},
                ]
                return messages

            prepare_messages = default_messages

        if results is None:
            results = self.rerank_results if self.rerank_results is None else self.retrieve_results
            assert results is not None, ("Neither rerank_results nor retrieve_results are available. "
                                         "One of them must be provided.")
        messages_dict = self.prepare_generation_inputs(results, prepare_messages)
        self.generate_results = model.generation(messages_dict, **kwargs)

        return self.generate_results

    def prepare_generation_inputs(self, results, prepare_messages) -> Dict[str, List[dict]]:
        messages_dict: Dict[str, List[Dict[str, str]]] = {}
        logger.info("Preparing generation inputs for %d queries.", len(results))
        for query_id, result in results.items():
            query = self.queries[query_id]
            documents = [(self.corpus[doc_id], score) for doc_id, score in result.items()]
            messages = prepare_messages(query, documents)
            messages_dict[query_id] = messages

        logger.info("Successfully prepared generation inputs for all queries.")
        return messages_dict

    def save_results(self, top_k: int = 10, output_dir: Optional[str] = None) -> None:
        # If no output directory is provided, stop saving.
        if output_dir is None:
            return
        # Create the output directory if it does not exist
        output_dir = os.path.join(output_dir, self.metadata.name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Output directory set to: {output_dir}")

        # Path to save the CSV file
        csv_file_path = os.path.join(output_dir, "results.csv")
        logger.info(f"Saving top {top_k} results to CSV file: {csv_file_path}")

        # Determine whether to use rerank results or retrieve results
        final_result = self.rerank_results if self.rerank_results is not None else self.retrieve_results

        # Process the final result if it's not None
        if final_result is not None:
            with open(csv_file_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                # Write the header to the CSV file
                writer.writerow(['query_id', 'corpus_id'])
                logger.info("Writing header ['query_id', 'corpus_id'] to CSV.")

                # For each query_id, save the top_k corpus_ids sorted by score
                for q_id, doc_scores in final_result.items():
                    # Sort doc_scores by score and select top_k documents
                    sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
                    logger.info(f"Saving top {top_k} corpus_id for query_id: {q_id}")

                    # Write the query_id and corpus_id to the CSV
                    for doc_id, _ in sorted_docs:
                        writer.writerow([q_id, doc_id])
                        logger.debug(f"Written query_id: {q_id}, corpus_id: {doc_id} to CSV")

            logger.info(f"Top {top_k} results saved successfully to {csv_file_path}")

        # Save generate_results to JSON Lines format
        if self.generate_results is not None:
            jsonl_file_path = os.path.join(output_dir, 'output.jsonl')
            logger.info(f"Saving generate_results to JSONL file: {jsonl_file_path}")

            with open(jsonl_file_path, 'w') as f:
                f.writelines(
                    json.dumps(
                        {'query_id': q_id, 'answer': answer}
                    ) + '\n' for q_id, answer in self.generate_results.items()
                )

            logger.info(f"generate_results saved successfully to {jsonl_file_path}")
