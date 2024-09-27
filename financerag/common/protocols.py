from typing import List, Protocol, runtime_checkable, Union, Tuple, Literal, Dict, Optional
import numpy as np
import torch

__all__ = [
    "Encoder",
    "Retrieval",
    "Reranker",
    "Generator",
]


@runtime_checkable
class Encoder(Protocol):
    """
    Protocol for encoders, providing methods to encode texts, queries, and corpora into dense vectors.
    """

    def encode_queries(
            self,
            queries: List[str],
            **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encodes a list of queries into dense vector representations.

        Args:
            queries (List[str]): A list of query strings to encode.
            **kwargs: Additional arguments passed to the encoder.

        Returns:
            Union[torch.Tensor, np.ndarray]: Encoded queries as a tensor or numpy array.
        """
        ...

    def encode_corpus(
            self,
            corpus: Union[List[Dict[Literal['title', 'text'], str]], Dict[Literal['title', 'text'], List]],
            **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encodes a list of corpus documents into dense vector representations.

        Args:
            corpus (List[str]): A list of corpus documents to encode.
            **kwargs: Additional arguments passed to the encoder.

        Returns:
            Union[torch.Tensor, np.ndarray]: Encoded corpus documents as a tensor or numpy array.
        """
        ...


@runtime_checkable
class Retrieval(Protocol):
    """
    Protocol for retrievers, providing a method to search for the most relevant documents based on queries.
    """

    def retrieve(
            self,
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str],
            top_k: Optional[int] = None,
            score_function: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Searches the corpus for the most relevant documents to the given queries.

        Args:
            corpus (Dict[str, Dict[str, str]]):
                A dictionary where each key is a document ID and each value is another dictionary containing document fields
                (e.g., {'text': str, 'title': str}).
            queries (Dict[str, str]):
                A dictionary where each key is a query ID and each value is the query text.
            top_k (Optional[int], optional):
                The number of top documents to return for each query. If None, return all documents. Defaults to None.
            score_function (Optional[str], optional):
                The scoring function to use when ranking the documents (e.g., 'cosine', 'dot', etc.). Defaults to None.
            **kwargs:
                Additional arguments passed to the search method.

        Returns:
            Dict[str, Dict[str, float]]:
                A dictionary where each key is a query ID, and each value is another dictionary mapping document IDs to
                relevance scores (e.g., {'doc1': 0.9, 'doc2': 0.8}).
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """
    Protocol for rerankers, providing methods to predict sentence similarity and rank documents based on queries.
    """

    def predict(
            self,
            sentences: Union[List[Tuple[str, str]], List[List[str]], Tuple[str, str], List[str]],
            batch_size: Optional[int] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Predicts similarity or relevance scores for pairs of sentences or lists of sentences.

        Args:
            sentences (Union[List[Tuple[str, str]], List[List[str]], Tuple[str, str], List[str]]):
                Sentences to predict similarity scores for. Can be a list of sentence pairs, list of sentence lists,
                a single sentence pair, or a list of sentences.
            batch_size (Optional[int], optional): Batch size for prediction. Defaults to None.

        Returns:
            Union[torch.Tensor, np.ndarray]: Predicted similarity or relevance scores as a tensor or numpy array.
        """
        ...

    def rank(
            self,
            query: str,
            documents: List[str],
            top_k: Optional[int] = None,
            return_documents: Optional[bool] = None,
            batch_size: Optional[int] = None,
    ) -> List[Dict[Literal["corpus_id", "score", "text"], Union[int, float, str]]]:
        """
        Ranks a list of documents based on the relevance to a given query.

        Args:
            query (str): The query string to rank documents for.
            documents (List[str]): A list of document strings to rank.
            top_k (Optional[int], optional): Number of top documents to return. Defaults to None.
            return_documents (Optional[bool], optional): Whether to return full document texts. Defaults to None.
            batch_size (Optional[int], optional): Batch size for processing. Defaults to None.

        Returns:
            List[Dict[Literal["corpus_id", "score", "text"], Union[int, float, str]]]:
                A list of dictionaries, where each dictionary contains:
                    - 'corpus_id' (int): The document ID.
                    - 'score' (float): The relevance score.
                    - 'text' (str): The document text (if return_documents is True).
        """
        ...


@runtime_checkable
class Generator(Protocol):
    """
    Protocol for text generators, providing methods for generating text completions in a chat-like interface.
    """

    def generation(
            self,
            messages: Dict[str, List[Dict[str, str]]],
            **kwargs
    ) -> Dict[str, str]:
        """
        Generates a chat completion based on a sequence of messages.

        Args:
            messages (Dict[str, List[Dict[str, str]]]): A list of message dictionaries per `query_id`.
            Each dictionary in list must contain:
                - 'role' (str): The role of the speaker (e.g., 'user' or 'system').
                - 'content' (str): The content of the message.
            **kwargs: Additional arguments passed to the generator.

        Returns:
            Dict[str, str]: A dictionary containing the generated response:
                - 'query_id' (str): The query ID as a key.
                - 'answer' (str): The generated text content as a value.
        """
        ...
