import logging
from typing import Optional, Union, List, Dict, Callable, Tuple

from datasets import Dataset, IterableDataset
from pydantic import BaseModel, ConfigDict

from financerag.common.protocols import Retrieval, Reranker, Generator
from financerag.tasks.TaskMetadata import TaskMetadata

DatasetType = Union[Dataset, IterableDataset]

logger = logging.getLogger(__name__)


class BaseTask(BaseModel):
    metadata: TaskMetadata
    dataset: Optional[DatasetType] = None
    data_loaded: bool = False
    retrieve_results: Optional[Dict] = None
    rerank_results: Optional[Dict] = None
    generate_results: Optional[Dict] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_data(self) -> DatasetType:
        raise NotImplementedError

    def retrieve(
            self,
            retriever: Retrieval,
            top_k: Optional[int] = 100,
            **kwargs
    ):
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
    ):
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
