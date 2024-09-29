from .BaseTask import BaseTask
from .TaskMetadata import TaskMetadata


class FinQABench(BaseTask):
    metadata: TaskMetadata = TaskMetadata(
        name="FinQABench",
        description="FinQABench: A New QA Benchmark for Finance applications",
        reference="https://huggingface.co/datasets/lighthouzai/finqabench",
        dataset={
            "path": "Linq-AI-Research/FinanceRAG",
            "subset": "FinQABench",
        },
        type="RAG",
        category="s2p",
        modalities=["text"],
        date=None,
        domains=["Report"],
        task_subtypes=[
            "Financial retrieval",
            "Question answering",
        ],
        license="apache-2.0",
        annotations_creators='LM-generated and reviewed',
        dialect=[],
        sample_creation='LM-generated and verified',
        bibtex_citation=None,
    )
