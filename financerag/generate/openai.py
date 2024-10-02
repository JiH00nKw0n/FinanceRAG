import logging
import multiprocessing
import os
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple, cast

import openai
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import Field

from .base import BaseGenerator

openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)


class OpenAIGenerator(BaseGenerator):
    name: str = Field(None, description="The model name of the OpenAI model to be used")

    def _process_query(
        self, args: Tuple[str, List[ChatCompletionMessageParam], Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Internal method to process a single query with the OpenAI model.
        Args:
            args (tuple): Contains query ID, messages, model name, and kwargs.
        Returns:
            tuple: Contains the query ID and the generated response.
        """
        q_id, messages, kwargs = args
        temperature = kwargs.pop("temperature", 1.0)
        top_p = kwargs.pop("top_p", 1.0)
        stream = kwargs.pop("stream", False)
        max_tokens = kwargs.pop("max_tokens", 10000)
        presence_penalty = kwargs.pop("presence_penalty", 0.0)
        frequency_penalty = kwargs.pop("frequency_penalty", 0.0)

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        return q_id, response.choices[0].message.content

    def generation(
        self,
        messages: Dict[str, List[Dict[str, str]]],
        num_processes: int = multiprocessing.cpu_count(),  # Number of parallel processes
        **kwargs,
    ) -> Dict[str, str]:
        """
        Generate responses for the given messages using the OpenAI model.
        Args:
            messages (Dict[str, List[Dict[str, str]]]): A dictionary with query IDs and associated messages.
            num_processes (int): Number of processes to use for parallel processing.
            **kwargs: Additional arguments for OpenAI model generation.
        Returns:
            Dict[str, str]: A dictionary containing the query IDs and generated responses.
        """
        logger.info(
            f"Starting generation for {len(messages)} queries using {num_processes} processes..."
        )

        # Prepare arguments for multiprocessing
        query_args = [
            (q_id, cast(list[ChatCompletionMessageParam], msg), kwargs.copy())
            for q_id, msg in messages.items()
        ]

        # Use multiprocessing Pool for parallel generation
        with Pool(processes=num_processes) as pool:
            results = pool.map(self._process_query, query_args)

        # Collect results
        self.results = {q_id: content for q_id, content in results}

        logger.info(
            f"Generation completed for all queries. Collected {len(self.results)} results."
        )

        return self.results
