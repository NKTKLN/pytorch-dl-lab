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
        self.word2token = {token: i for i, token in enumerate(self.BASE_TOKENS)}
        self.token2word = dict(enumerate(self.BASE_TOKENS))
        self.n_words = len(self.token2word)

    def __len__(self) -> int:
        """Returns the number of words in the vocabulary."""
        return self.n_words

    def get_by_token(self, token: int, default: str | None = None) -> str | None:
        """Looks up the word for a token id.

        Args:
            token: The token id to look up.
            default: Value returned if `token` is not in the vocabulary.

        Returns:
            The word mapped to `token`, or `default` if absent.
        """
        return self.token2word.get(token, default)

    def get_by_word(self, word: str, default: int | None = None) -> int | None:
        """Looks up the token id for a word.

        Args:
            word: The word to look up.
            default: Value returned if `word` is not in the vocabulary.

        Returns:
            The token id mapped to `word`, or `default` if absent.
        """
        return self.word2token.get(word, default)

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
        if word in self.word2token:
            return

        self.word2token[word] = self.n_words
        self.token2word[self.n_words] = word
        self.n_words += 1
