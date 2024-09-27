from typing import Optional, Union

from datasets import Dataset, IterableDataset
from pydantic import BaseModel, ConfigDict

from financerag.tasks.TaskMetadata import TaskMetadata

DatasetType = Union[Dataset, IterableDataset]

class BaseTask(BaseModel):
    metadata: TaskMetadata
    dataset: Optional[DatasetType] = None
    data_loaded: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_data(self):
        raise NotImplementedError

    def run(self):
        pass

