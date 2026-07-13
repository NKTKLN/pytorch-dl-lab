"""Word-level vocabulary for sequence models."""

from typing import ClassVar


class Lang:
    """Vocabulary built from a corpus of tokenized sentences.

    Maps words to ids and back, assigning ids in first-seen order on top
    of the reserved special tokens (``<BOS>``, ``<EOS>``, ``<UNK>``, ``<PAD>``).
    """

    BASE_TOKENS: ClassVar[list[str]] = ["<BOS>", "<EOS>", "<UNK>", "<PAD>"]

    def __init__(self, name: str) -> None:
        """Initializes a vocabulary seeded with the base special tokens.

        Args:
            name: Human-readable name of the language (e.g. ``"en"``).
        """
        self.name = name
        self.word2index = {token: i for i, token in enumerate(self.BASE_TOKENS)}
        self.index2word = dict(enumerate(self.BASE_TOKENS))
        self.n_words = len(self.index2word)

    def add_sentence(self, sentence: list[str] | str) -> None:
        """Adds every word of a sentence to the vocabulary.

        Args:
            sentence: A tokenized sentence, or a space-separated string.
        """
        if isinstance(sentence, str):
            sentence = sentence.split(" ")

        for word in sentence:
            self.add_word(word)

    def add_word(self, word: str) -> None:
        """Adds a single word to the vocabulary if not already present.

        Args:
            word: The word to register.
        """
        if word in self.word2index:
            return

        self.word2index[word] = self.n_words
        self.index2word[self.n_words] = word
        self.n_words += 1
