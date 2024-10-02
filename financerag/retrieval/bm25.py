import logging
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
from nltk.tokenize import word_tokenize

from financerag.common import Lexical, Retrieval

logger = logging.getLogger(__name__)


def tokenize_list(input_list: List[str]) -> List[List[str]]:
    return list(map(word_tokenize, input_list))


class BM25Retriever(Retrieval):

    def __init__(self, model: Lexical, tokenizer: Callable[[List[str]], List[List[str]]] = tokenize_list):
        self.model: Lexical = model
        self.tokenizer: Callable[[List[str]], List[List[str]]] = tokenizer
        self.results: Optional[Dict[str, Any]] = {}

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

        for qid, query in zip(query_ids, query_lower_tokens):
            scores = self.model.get_scores(query)
            top_k_result = np.argsort(scores)[::-1][:top_k]
            for idx in top_k_result:
                self.results[qid][corpus_ids[idx]] = scores[idx]

        return self.results
