from typing import Any, Optional, Dict

from pydantic import BaseModel


class BaseReranker(BaseModel):
    model: Any
    results: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
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
        raise NotImplementedError
