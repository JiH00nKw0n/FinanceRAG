import logging
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
from nltk.tokenize import word_tokenize
from pydantic import Field, model_validator

from financerag.common import Lexical

from .base import BaseRetriever

logger = logging.getLogger(__name__)


def tokenize_list(input_list: List[str]) -> List[List[str]]:
    return list(map(word_tokenize, input_list))

class BM25Retriever(BaseRetriever):
    tokenizer: Callable[[List[str]], List[List[str]]] = Field(default=tokenize_list)

    @model_validator(mode="after")
    def check_model(self):
        """
        Validates that the model implements the Encoder protocol.
        """
        if not isinstance(self.model, Lexical):
            raise TypeError("model must implement the `Lexical` protocol")
        return self

    def retrieve(
        self,
        corpus: Dict[str, Dict[Literal["id", "title", "text"], str]],
        queries: Dict[Literal["id", "text"], str],
        top_k: Optional[int] = None,
        return_sorted: bool = False,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:

        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}

        logger.info("Tokenizing queries with lower cases")
        query_lower_tokens = self.tokenizer([queries[qid].lower() for qid in queries])

        corpus_ids = list(corpus.keys())

        corpus_lower = [
            (
                (corpus[cid]["title"] + " " + corpus[cid]["text"]).strip()
                if "title" in corpus
                else corpus[cid]["text"].strip()
            )
            for cid in corpus_ids
        ]

        logger.info("Prepare Lexical model with ...")
        model = self.model(corpus=corpus_lower, tokenizer=self.tokenizer, **kwargs)

        logger.info("...")
        for qid, query in zip(query_ids, query_lower_tokens):
            scores = model.get_scores(query)
            top_k_result = np.argsort(scores)[::-1][:top_k]
            for idx in top_k_result:
                self.results[qid][corpus_ids[idx]] = scores[idx]

        return self.results
