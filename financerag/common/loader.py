import logging
import os
from typing import Dict, Tuple

from datasets import load_dataset, Value

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class HFDataLoader:

    def __init__(
            self,
            hf_repo: str = None,
            data_folder: str = None,
            subset: str = None,
            prefix: str = None,
            corpus_file: str = "corpus.jsonl",
            query_file: str = "queries.jsonl",
            streaming: bool = False,
            keep_in_memory: bool = False):
        self.corpus = {}
        self.queries = {}
        self.hf_repo = hf_repo
        self.subset = subset
        if hf_repo:
            logger.warning(
                "A huggingface repository is provided. This will override the data_folder, prefix and *_file arguments.")
        else:
            # data folder would contain these files:
            # (1) FinDER/corpus.jsonl  (format: jsonlines)
            # (2) FinDER/queries.jsonl (format: jsonlines)
            if prefix:
                query_file = prefix + "_" + query_file

            self.corpus_file = os.path.join(data_folder, subset, corpus_file) if data_folder else corpus_file
            self.query_file = os.path.join(data_folder, subset, query_file) if data_folder else query_file
        self.streaming = streaming
        self.keep_in_memory = keep_in_memory

    @staticmethod
    def check(file_in: str, ext: str):
        if not os.path.exists(file_in):
            raise ValueError("File {} not present! Please provide accurate file.".format(file_in))

        if not file_in.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(file_in, ext))

    def load(self) -> Tuple[Dict[str, str], Dict[str, str]]:

        if not self.hf_repo:
            self.check(file_in=self.corpus_file, ext="jsonl")
            self.check(file_in=self.query_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d TEST Documents.", len(self.corpus))
            logger.info("Doc Example: %s", self.corpus[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        logger.info("Loaded %d TEST Queries.", len(self.queries))
        logger.info("Query Example: %s", self.queries[0])

        return self.corpus, self.queries

    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        if not self.hf_repo:
            self.check(file_in=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus))
            logger.info("Doc Example: %s", self.corpus[0])

        return self.corpus

    def _load_corpus(self):
        if self.hf_repo:
            corpus_ds = load_dataset(
                path=self.hf_repo,
                name=self.subset,
                split='corpus',
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming
            )
        else:
            corpus_ds = load_dataset(
                'json',
                data_files=self.corpus_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory
            )
        corpus_ds = corpus_ds.cast_column('_id', Value('string'))
        corpus_ds = corpus_ds.rename_column('_id', 'id')
        corpus_ds = corpus_ds.remove_columns(
            [col for col in corpus_ds.column_names if col not in ['id', 'text', 'title']])
        self.corpus = corpus_ds

    def _load_queries(self):
        if self.hf_repo:
            queries_ds = load_dataset(
                path=self.hf_repo,
                name=self.subset,
                split='queries',
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming
            )
        else:
            queries_ds = load_dataset(
                'json',
                data_files=self.query_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory
            )
        queries_ds = queries_ds.cast_column('_id', Value('string'))
        queries_ds = queries_ds.rename_column('_id', 'id')
        queries_ds = queries_ds.remove_columns([col for col in queries_ds.column_names if col not in ['id', 'text']])
        self.queries = queries_ds
