from typing import Union, List, Literal, Dict

import numpy as np
from torch import Tensor

from .base import BaseEncoder


# Adopted by https://github.com/beir-cellar/beir/blob/main/beir/retrieval/models/sentence_bert.py
class SentenceTransformerEncoder(BaseEncoder):

    def encode_queries(
            self,
            queries: List[str],
            batch_size: int = 16,
            **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        if self.query_prompt is not None:
            queries = [self.query_prompt + query for query in queries]
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(
            self,
            corpus: Union[List[Dict[Literal['title', 'text'], str]], Dict[Literal['title', 'text'], List]],
            batch_size: int = 8,
            **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        if type(corpus) is dict:
            sentences = [
                (
                        corpus["title"][i] + ' ' + corpus["text"][i]
                ).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))
            ]
        else:
            sentences = [
                (
                        doc["title"] + ' ' + doc["text"]
                ).strip() if "title" in doc else doc["text"].strip() for doc in corpus
            ]
        if self.doc_prompt is not None:
            sentences = [self.doc_prompt + s for s in sentences]
        return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs)
