import heapq
import logging
from typing import Literal, Dict, Optional, Any

import torch
from pydantic import model_validator

from financerag.common.protocols import Encoder
from .base import BaseRetriever

logger = logging.getLogger(__name__)


# Copied from https://github.com/beir-cellar/beir/blob/main/beir/retrieval/search/dense/util.py
@torch.no_grad()
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        a (torch.Tensor): Tensor representing query embeddings.
        b (torch.Tensor): Tensor representing corpus embeddings.

    Returns:
        torch.Tensor: Cosine similarity scores for all pairs.
    """
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    return torch.mm(torch.nn.functional.normalize(a, p=2, dim=1),
                    torch.nn.functional.normalize(b, p=2, dim=1).transpose(0, 1))


@torch.no_grad()
def dot_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the dot-product score between two tensors.

    Args:
        a (torch.Tensor): Tensor representing query embeddings.
        b (torch.Tensor): Tensor representing corpus embeddings.

    Returns:
        torch.Tensor: Dot-product scores for all pairs.
    """
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    return torch.mm(a, b.transpose(0, 1))


def _ensure_tensor(x: Any) -> torch.Tensor:
    """
    Ensures the input is a torch.Tensor, converting if necessary.

    Args:
        x (Any): Input to be checked.

    Returns:
        torch.Tensor: Converted tensor.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    return x


# Adapted from https://github.com/beir-cellar/beir/blob/main/beir/retrieval/search/dense/exact_search.py
class DenseRetriever(BaseRetriever):
    """
    Encoder Retrieval that performs similarity-based search over a corpus.
    """
    model: Encoder
    batch_size: int = 64
    score_functions: Optional[Dict[str, Any]] = None
    corpus_chunk_size: int = 50000

    @model_validator(mode='after')
    def check_model(self):
        """
        Validates that the model implements the Encoder protocol.
        """
        if not isinstance(self.model, Encoder):
            raise AttributeError("model must implement the `Encoder` protocol")
        return self

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes score functions if not provided.
        """
        super().model_post_init(__context)
        if self.score_functions is None:
            self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}

    def retrieve(
            self,
            corpus: Dict[str, Dict[Literal["title", "text"], str]],
            queries: Dict[str, str],
            top_k: Optional[int] = None,
            return_sorted: bool = False,
            score_function: Optional[str] = 'cos_sim',
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieves the top-k most relevant documents from the corpus based on the given queries.

        Args:
            corpus (Dict): A dictionary where each key is a document ID and value contains the document metadata.
            queries (Dict): A dictionary where each key is a query ID and value is the query text.
            top_k (Optional[int]): Number of top results to return. If None, returns all.
            return_sorted (bool): Whether to return sorted results.
            score_function (Optional[str]): Scoring function to use ('cos_sim' or 'dot').
            **kwargs: Additional arguments for the retrieve function.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary with query IDs as keys and another dictionary containing
            document IDs and their scores as values.
        """
        if score_function not in self.score_functions:
            raise ValueError(
                f"Score function: {score_function} must be either 'cos_sim' for cosine similarity or 'dot' for dot product."
            )

        logger.info("Encoding queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        query_texts = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            query_texts, batch_size=self.batch_size, **kwargs
        )

        logger.info("Sorting corpus by document length...")
        sorted_corpus_ids = sorted(
            corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)

        logger.info("Encoding corpus in batches... This may take a while.")
        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        corpus = [corpus[cid] for cid in sorted_corpus_ids]

        for batch_num, start_idx in enumerate(range(0, len(corpus), self.corpus_chunk_size)):
            logger.info(f"Encoding batch {batch_num + 1}/{len(range(0, len(corpus), self.corpus_chunk_size))}...")
            end_idx = min(start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[start_idx:end_idx],
                batch_size=self.batch_size,
                **kwargs
            )

            # Compute similarities using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores, min(top_k + 1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)

            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = sorted_corpus_ids[start_idx + sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results
