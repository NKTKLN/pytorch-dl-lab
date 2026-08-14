"""Chapter 10 — transformer summarizer on the gazeta corpus."""

from dl_roadmap.chapters.summarization.data import prepare_gazeta
from dl_roadmap.chapters.summarization.dataset import (
    SummarizationBatch,
    SummarizationDataset,
)
from dl_roadmap.chapters.summarization.model import Summarizer

__all__ = [
    "SummarizationBatch",
    "SummarizationDataset",
    "Summarizer",
    "prepare_gazeta",
]
