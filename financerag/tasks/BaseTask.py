from typing import Optional, Union, List, Dict

from datasets import Dataset, IterableDataset
from pydantic import BaseModel, ConfigDict

from financerag.tasks.TaskMetadata import TaskMetadata
from financerag.common.protocols import Retrieval, Reranker, Generator

DatasetType = Union[Dataset, IterableDataset]

class BaseTask(BaseModel):
    metadata: TaskMetadata
    dataset: Optional[DatasetType] = None
    data_loaded: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_data(self) -> DatasetType:
        raise NotImplementedError

    def retrieve(self, model: Retrieval, **kwargs):
        if not isinstance(model, Retrieval):
            raise AttributeError("Model must be have an attributes of `Retrieval`")
        results = model.retrieve(queries=self.queries, corpus=self.corpus, **kwargs)
        return results

    def rerank(self, model: Reranker, **kwargs):
        results = model.rerank(queries=self.queries, corpus=self.corpus, **kwargs)
        return results

    def generate(self, model, results, prepare_messages, **kwargs):
        messages_dict = self.prepare_generation_inputs(results, prepare_messages)
        answers = model.generation(messages_dict, **kwargs)

        return answers

    def prepare_generation_inputs(self, results, prepare_messages) -> Dict[str, List[dict]]:
        messages_dict: Dict[str, List[Dict[str, str]]] = {}
        for query_id, result in results.items():
            query = self.queries[query_id]
            documents = [(self.corpus[doc_id], score) for doc_id, score in result.items()]
            messages = prepare_messages(query, documents)
            messages_dict[query_id] = messages

        return messages_dict
