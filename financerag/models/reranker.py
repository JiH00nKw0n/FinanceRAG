# Adapted from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/encoder_interface.py
from __future__ import annotations

from typing import Any, Dict, List, Protocol, Sequence, Union, runtime_checkable

import numpy as np
import torch

Corpus = Union[List[Dict[str, str]], Dict[str, List[str]]]


@runtime_checkable
class Encoder(Protocol):
    """The interface for an encoder in MTEB.

    Besides the required functions specified below, the encoder can additionally specify the the following signatures seen below.
    In general the interface is kept aligned with sentence-transformers interface. In cases where exceptions occurs these are handled within MTEB.
    """

    def __init__(self, device: str | None = None) -> None:
        """The initialization function for the encoder. Used when calling it from the mteb run CLI.

        Args:
            device: The device to use for encoding. Can be ignored if the encoder is not using a device (e.g. for API)
        """

    def encode(
        self, sentences: Sequence[str], *, prompt_name: str | None = None, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            prompt_name: The name of the prompt. This will just be the name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        ...


@runtime_checkable
class EncoderWithQueryCorpusEncode(Encoder, Protocol):
    """The optional interface for an encoder that supports encoding queries and a corpus."""

    def encode_queries(
        self, queries: Sequence[str], *, prompt_name: str | None = None, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given queries using the encoder.

        Args:
            queries: The queries to encode.
            prompt_name: The name of the prompt. This will just be the name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded queries.
        """
        ...

    def encode_corpus(
        self, corpus: Corpus, *, prompt_name: str | None = None, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given corpus using the encoder.

        Args:
            corpus: The corpus to encode.
            prompt_name: The name of the prompt. This will just be the name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded corpus.
        """
        ...