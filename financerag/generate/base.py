from typing import Any, Optional, Dict, List

from pydantic import BaseModel


class BaseGenerator(BaseModel):
    model: Any
    results: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.results is None:
            self.results = {}

    def generation(
            self,
            messages: Dict[str, List[Dict[str, str]]],
            **kwargs
    ) -> Dict[str, str]:
        raise NotImplementedError
