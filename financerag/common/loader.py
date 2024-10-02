import logging
from pathlib import Path
from typing import Optional, Tuple, cast

from datasets import Dataset, Value, load_dataset

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class HFDataLoader:
    """
    A Hugging Face Dataset loader for corpus and query data. Supports loading datasets from local files
    (in JSONL format) or directly from a Hugging Face repository.

    Args:
        hf_repo (`str`, *optional*):
            The Hugging Face repository containing the dataset. If provided, it overrides the
            data folder, prefix, and *_file arguments.
        data_folder (`str`, *optional*):
            Path to the folder containing the dataset files when loading from local files.
        subset (`str`, *optional*):
            The subset of the dataset to load (e.g., "FinDER", "FinQA"). Used in both local and HF repo loading.
        prefix (`str`, *optional*):
            A prefix to add to the file names (e.g., "train_", "test_").
        corpus_file (`str`, defaults to `"corpus.jsonl"`):
            The filename for the corpus when loading locally.
        query_file (`str`, defaults to `"queries.jsonl"`):
            The filename for the queries when loading locally.
        streaming (`bool`, defaults to `False`):
            Whether to stream the dataset. Useful for very large datasets.
        keep_in_memory (`bool`, defaults to `False`):
            Whether to keep the dataset in memory.
    """

    def __init__(
        self,
        hf_repo: Optional[str] = None,
        data_folder: Optional[str] = None,
        subset: Optional[str] = None,
        prefix: Optional[str] = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        streaming: bool = False,
        keep_in_memory: bool = False,
    ):
        """
        Initializes the HFDataLoader class.

        Args:
            See class-level docstring for description of arguments.
        """
        self.corpus: Optional[Dataset] = None
        self.queries: Optional[Dataset] = None
        self.hf_repo = hf_repo
        self.subset = subset
        if hf_repo:
            logger.warning(
                "A Hugging Face repository is provided. This will override the data_folder, prefix and *_file arguments."
            )
        else:
            if (data_folder is None) or (subset is None):
                raise ValueError(
                    "A Hugging Face repository or local directory required."
                )

            # Set up local file paths
            if prefix:
                query_file = prefix + "_" + query_file

            self.corpus_file = (
                (Path(data_folder) / subset / corpus_file).as_posix()
                if data_folder
                else corpus_file
            )
            self.query_file = (
                (Path(data_folder) / subset / query_file).as_posix()
                if data_folder
                else query_file
            )
        self.streaming = streaming
        self.keep_in_memory = keep_in_memory

    @staticmethod
    def check(file_in: str, ext: str):
        """
        Check if the given file exists and has the correct extension.

        Args:
            file_in (`str`): The path of the file to check.
            ext (`str`): The expected file extension.

        Raises:
            `ValueError`: If the file does not exist or if the extension does not match.
        """
        if not Path(file_in).exists():
            raise ValueError(
                "File {} not present! Please provide accurate file.".format(file_in)
            )

        if not file_in.endswith(ext):
            raise ValueError(
                "File {} must be present with extension {}".format(file_in, ext)
            )

    def load(self) -> Tuple[Dataset, Dataset]:
        """
        Loads both the corpus and query datasets. If the datasets are not already loaded,
        they are loaded from the specified source (either local files or Hugging Face repository).

        Returns:
            `Tuple[Dataset, Dataset]`: A tuple containing the loaded corpus and queries datasets.
        """
        if not self.hf_repo:
            self.check(file_in=self.corpus_file, ext="jsonl")
            self.check(file_in=self.query_file, ext="jsonl")

        if self.corpus is None:
            logger.info("Loading Corpus...")
            self._load_corpus()
            self.corpus = cast(Dataset, self.corpus)
            logger.info("Loaded %d TEST Documents.", len(self.corpus))
            logger.info("Doc Example: %s", self.corpus[0])

        if self.queries is None:
            logger.info("Loading Queries...")
            self._load_queries()
            self.queries = cast(Dataset, self.queries)

        logger.info("Loaded %d TEST Queries.", len(self.queries))
        logger.info("Query Example: %s", self.queries[0])

        return self.corpus, self.queries

    def load_corpus(self) -> Dataset:
        """
        Loads the corpus dataset. If the corpus is already loaded, returns the existing dataset.

        Returns:
            `Dataset`: The loaded corpus dataset.
        """
        if not self.hf_repo:
            self.check(file_in=self.corpus_file, ext="jsonl")

        if (self.corpus is None) or (not len(self.corpus)):
            logger.info("Loading Corpus...")
            self._load_corpus()
            self.corpus = cast(Dataset, self.corpus)
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", self.corpus[0])

        return self.corpus

    def _load_corpus(self):
        """
        Internal method to load the corpus dataset from either local files or Hugging Face repository.
        The dataset is processed by renaming and removing unnecessary columns.
        """
        if self.hf_repo:
            corpus_ds = load_dataset(
                path=self.hf_repo,
                name=self.subset,
                split="corpus",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )
        else:
            corpus_ds = load_dataset(
                "json",
                data_files=self.corpus_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )

        corpus_ds = cast(Dataset, corpus_ds)
        corpus_ds = corpus_ds.cast_column("_id", Value("string"))
        corpus_ds = corpus_ds.rename_column("_id", "id")
        corpus_ds = corpus_ds.remove_columns(
            [
                col
                for col in corpus_ds.column_names
                if col not in ["id", "text", "title"]
            ]
        )
        self.corpus = corpus_ds

    def _load_queries(self):
        """
        Internal method to load the queries dataset from either local files or Hugging Face repository.
        The dataset is processed by renaming and removing unnecessary columns.
        """
        if self.hf_repo:
            queries_ds = load_dataset(
                path=self.hf_repo,
                name=self.subset,
                split="queries",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )
        else:
            queries_ds = load_dataset(
                "json",
                data_files=self.query_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        queries_ds = cast(Dataset, queries_ds)
        queries_ds = queries_ds.cast_column("_id", Value("string"))
        queries_ds = queries_ds.rename_column("_id", "id")
        queries_ds = queries_ds.remove_columns(
            [col for col in queries_ds.column_names if col not in ["id", "text"]]
        )
        self.queries = queries_ds
