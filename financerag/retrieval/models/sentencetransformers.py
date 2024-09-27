from typing import Union, Tuple, List, Literal, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from torch import Tensor


# Adopted by https://github.com/beir-cellar/beir/blob/main/beir/retrieval/models/sentence_bert.py
class SentenceTransformersEncoder:

    def __init__(self, model_name_or_path: Union[str, Tuple] = None, **kwargs):
        if isinstance(model_name_or_path, str):
            self.q_model = SentenceTransformer(model_name_or_path, **kwargs)
            self.doc_model = self.q_model

        elif isinstance(model_name_or_path, tuple):
            self.q_model = SentenceTransformer(model_name_or_path[0], **kwargs)
            self.doc_model = SentenceTransformer(model_name_or_path[1], **kwargs)

    def encode_queries(
            self,
            queries: List[str],
            batch_size: int = 16,
            **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
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
        return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs)
