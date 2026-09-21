"""Chapter 10 — transformer summarizer on the gazeta corpus."""

from dl_roadmap.chapters.summarization.data import deduplicate_split, prepare_gazeta
from dl_roadmap.chapters.summarization.dataset import (
    SummarizationBatch,
    SummarizationDataset,
)
from dl_roadmap.chapters.summarization.model import Summarizer
from dl_roadmap.chapters.summarization.trainer import SummarizationTrainer

__all__ = [
    "SummarizationBatch",
    "SummarizationDataset",
    "SummarizationTrainer",
    "Summarizer",
    "deduplicate_split",
    "prepare_gazeta",
]
