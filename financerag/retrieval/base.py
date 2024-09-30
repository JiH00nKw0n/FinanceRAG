from typing import Any, Optional, Dict, Literal, List, Union

import numpy as np
from pydantic import BaseModel
from torch import Tensor


class BaseRetriever(BaseModel):
    """
    Base class for retrievers. This class provides a structure for retrieving relevant documents
    from a corpus based on a set of queries. It is meant to be extended by specific retriever implementations.

    Attributes:
        model (`Any`):
            The underlying model used for retrieval.
        results (`Optional[Dict[str, Any]]`, defaults to `None`):
            Stores the results from the retrieval process, if available.

    Methods:
        model_post_init(__context: Any):
            Ensures the `results` attribute is initialized to an empty dictionary if not already set.
        retrieve(corpus: Dict[str, Dict[Literal["id", "title", "text"], str]], queries: Dict[Literal["id", "text"], str],
                 top_k: Optional[int] = None, return_sorted: bool = False, **kwargs):
            Abstract method to retrieve documents based on the input queries. Must be implemented by subclasses.
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

    def retrieve(
            self,
            corpus: Dict[str, Dict[Literal["id", "title", "text"], str]],
            queries: Dict[Literal["id", "text"], str],
            top_k: Optional[int] = None,
            return_sorted: bool = False,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Abstract method to retrieve relevant documents from the corpus based on input queries.

        Args:
            corpus (`Dict[str, Dict[Literal["id", "title", "text"], str]]`):
                The corpus to retrieve from, where the key is the document ID and the value is a dictionary
                containing the document title and text.
            queries (`Dict[Literal["id", "text"], str]`):
                The queries used to retrieve documents, where the key is the query ID and the value is the query text.
            top_k (`Optional[int]`, defaults to `None`):
                The number of top documents to return for each query.
            return_sorted (`bool`, defaults to `False`):
                Whether to return the results sorted by score.
            **kwargs:
                Additional keyword arguments to pass to the retriever model.

        Returns:
            `Dict[str, Dict[str, float]]`:
                A dictionary where the key is the query ID and the value is another dictionary
                mapping document IDs to their retrieval scores.

        Raises:
            `NotImplementedError`:
                This method is abstract and must be implemented by any subclass.
        """
        raise NotImplementedError


class BaseEncoder(BaseModel):
    """
    Base class for encoders. This class provides the structure for encoding queries and corpus documents
    into embeddings for use in retrieval or other tasks. It is intended to be extended by specific encoder implementations.

    Attributes:
        q_model (`Any`):
            The model used for encoding queries.
        doc_model (`Any`):
            The model used for encoding corpus documents.
        query_prompt (`Optional[str]`, defaults to `None`):
            An optional prompt to be used when encoding queries.
        doc_prompt (`Optional[str]`, defaults to `None`):
            An optional prompt to be used when encoding documents.

    Methods:
        encode_queries(queries: List[str], batch_size: int = 16, **kwargs):
            Abstract method to encode a list of queries into embeddings. Must be implemented by subclasses.
        encode_corpus(corpus: Union[List[Dict[Literal['title', 'text'], str]], Dict[Literal['title', 'text'], List]],
                      batch_size: int = 16, **kwargs):
            Abstract method to encode a corpus of documents into embeddings. Must be implemented by subclasses.
    """

    q_model: Any
    doc_model: Any
    query_prompt: Optional[str] = None
    doc_prompt: Optional[str] = None

    def encode_queries(
            self,
            queries: List[str],
            batch_size: int = 16,
            **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        """
        Abstract method to encode a list of queries into embeddings. This method should be overridden by subclasses.

        Args:
            queries (`List[str]`):
                A list of query texts to be encoded.
            batch_size (`int`, defaults to `16`):
                The batch size for encoding the queries.
            **kwargs:
                Additional keyword arguments to pass to the query encoding process.

        Returns:
            `Union[List[Tensor], np.ndarray, Tensor]`:
                The encoded queries as either a list of tensors, a numpy array, or a single tensor.

        Raises:
            `NotImplementedError`:
                This method is abstract and must be implemented by any subclass.
        """
        raise NotImplementedError

    def encode_corpus(
            self,
            corpus: Union[List[Dict[Literal['title', 'text'], str]], Dict[Literal['title', 'text'], List]],
            batch_size: int = 16,
            **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        """
        Abstract method to encode a corpus of documents into embeddings. This method should be overridden by subclasses.

        Args:
            corpus (`Union[List[Dict[Literal['title', 'text'], str]], Dict[Literal['title', 'text'], List]]`):
                The corpus of documents to encode, either as a list of dictionaries with "title" and "text"
                fields, or as a dictionary where "title" and "text" are keys with lists of corresponding values.
            batch_size (`int`, defaults to `16`):
                The batch size for encoding the corpus.
            **kwargs:
                Additional keyword arguments to pass to the document encoding process.

        Returns:
            `Union[List[Tensor], np.ndarray, Tensor]`:
                The encoded corpus as either a list of tensors, a numpy array, or a single tensor.

        Raises:
            `NotImplementedError`:
                This method is abstract and must be implemented by any subclass.
        """
        raise NotImplementedError
