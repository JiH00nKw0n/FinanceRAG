from pydantic import BaseModel
from typing import Any, Optional, Dict, Literal


class BaseRetriever(BaseModel):
    model: Any
    results: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
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
        raise NotImplementedError

