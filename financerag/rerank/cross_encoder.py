import logging
from typing import Dict, Optional

from pydantic import model_validator

from financerag.common import CrossEncoder

from .base import BaseReranker

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/main/beir/reranking/rerank.py
class CrossEncoderReranker(BaseReranker):
    """
    A reranker class that utilizes a cross-encoder model from the `sentence-transformers` library
    to rerank search results based on query-document pairs. This class implements a reranking
    mechanism using cross-attention, where each query-document pair is passed through the
    cross-encoder model to compute relevance scores.

    The cross-encoder model expects two inputs (query and document) and directly computes a
    score indicating the relevance of the document to the query. The model follows the
    `CrossEncoder` protocol, ensuring it is compatible with `sentence-transformers` cross-encoder models.

    Methods:
        - rerank: Takes in a corpus, queries, and initial retrieval results, and reranks
                  the top-k documents using the cross-encoder model.
    """

    @model_validator(mode="after")
    def check_model(self):
        """
        Validates that the model implements the CrossEncoder protocol.
        """
        if not isinstance(self.model, CrossEncoder):
            raise AttributeError("model must implement the `CrossEncoder` protocol")
        return self

    def rerank(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        results: Dict[str, Dict[str, float]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:

        sentence_pairs, pair_ids = [], []

        for query_id in results:
            if len(results[query_id]) > top_k:
                for doc_id, _ in sorted(
                    results[query_id].items(), key=lambda item: item[1], reverse=True
                )[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (
                        corpus[doc_id].get("title", "")
                        + " "
                        + corpus[doc_id].get("text", "")
                    ).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])

            else:
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (
                        corpus[doc_id].get("title", "")
                        + " "
                        + corpus[doc_id].get("text", "")
                    ).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])

        #### Starting to Rerank using cross-attention
        logging.info("Starting To Rerank Top-{}....".format(top_k))
        rerank_scores = [
            float(score)
            for score in self.model.predict(
                sentences=sentence_pairs, batch_size=batch_size, **kwargs
            )
        ]

        #### Reranker results
        self.results = {query_id: {} for query_id in results}
        for pair, score in zip(pair_ids, rerank_scores):
            query_id, doc_id = pair[0], pair[1]
            self.results[query_id][doc_id] = score

        return self.results
