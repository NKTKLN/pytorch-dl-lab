"""Model architectures used throughout dl_roadmap."""

from dl_roadmap.models.char_rnn import CharRNN
from dl_roadmap.models.cnn_fashion_mnist import CnnBlock, CnnFashionMNIST
from dl_roadmap.models.imdb_lstm import ImdbLSTM
from dl_roadmap.models.mlp_mnist import MLP_MNIST
from dl_roadmap.models.reverser_seq2seq import (
    DecoderRNN as ReverserDecoderRNN,
)
from dl_roadmap.models.reverser_seq2seq import (
    EncoderRNN as ReverserEncoderRNN,
)
from dl_roadmap.models.reverser_seq2seq import ReverserSeq2Seq
from dl_roadmap.models.transformer_summarizer import (
    FFN,
    MultiHeadAttention,
    PositionalEncoding,
    Summarizer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from dl_roadmap.models.translator_seq2seq import (
    Attention as BahdanauAttention,
)
from dl_roadmap.models.translator_seq2seq import (
    DecoderRNN as TranslatorDecoderRNN,
)
from dl_roadmap.models.translator_seq2seq import (
    EncoderRNN as TranslatorEncoderRNN,
)
from dl_roadmap.models.translator_seq2seq import Translator

__all__ = [
    "FFN",
    "MLP_MNIST",
    "BahdanauAttention",
    "CharRNN",
    "CnnBlock",
    "CnnFashionMNIST",
    "ImdbLSTM",
    "MultiHeadAttention",
    "PositionalEncoding",
    "ReverserDecoderRNN",
    "ReverserEncoderRNN",
    "ReverserSeq2Seq",
    "Summarizer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "Translator",
    "TranslatorDecoderRNN",
    "TranslatorEncoderRNN",
]
