from typing import Any, Dict, Optional

from pydantic import BaseModel


class BaseReranker(BaseModel):
    """
    Base class for reranking models. This class provides the structure for reranking retrieval
    results based on a corpus and queries. It is meant to be extended by specific reranking models.

    Attributes:
        model (`Any`):
            The underlying reranking model used to process the results.
        results (`Optional[Dict[str, Any]]`, defaults to `None`):
            Stores the results after reranking, if available.

    Methods:
        model_post_init(__context: Any):
            Ensures the `results` attribute is initialized to an empty dictionary if not already set.
        rerank(corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], results: Dict[str, Dict[str, float]],
               top_k: Optional[int] = None, batch_size: Optional[int] = None, **kwargs):
            Abstract method to perform reranking on retrieval results. Must be implemented by subclasses.
    """

    model: Any
    results: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization method to ensure the `results` attribute is initialized.

        Args:
            __context (`Any`):
                Additional context or configuration for initialization (typically unused).

        If the `results` attribute is `None`, this method initializes it as an empty dictionary.
        """
        if self.results is None:
            self.results = {}

    def rerank(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        results: Dict[str, Dict[str, float]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Abstract method to perform reranking of retrieval results. This method should be
        overridden in subclasses with the specific reranking logic.

        Args:
            corpus (`Dict[str, Dict[str, str]]`):
                A dictionary where keys are document IDs and values are dictionaries containing
                the document text and other attributes (e.g., title, content).
            queries (`Dict[str, str]`):
                A dictionary where keys are query IDs and values are the query text.
            results (`Dict[str, Dict[str, float]]`):
                A dictionary where keys are query IDs and values are dictionaries mapping
                document IDs to their retrieval scores.
            top_k (`Optional[int]`, defaults to `None`):
                The number of top results to return after reranking. If `None`, returns all results.
            batch_size (`Optional[int]`, defaults to `None`):
                The size of batches to process during reranking.
            **kwargs:
                Additional keyword arguments to pass to the reranking model.

        Returns:
            `Dict[str, Dict[str, float]]`:
                A dictionary where the keys are query IDs and the values are dictionaries
                mapping document IDs to reranked scores.

        Raises:
            `NotImplementedError`:
                This method is abstract and must be implemented by any subclass.
        """
        raise NotImplementedError
