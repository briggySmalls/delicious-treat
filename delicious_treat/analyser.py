"""Analysis of a set of text"""
from typing import Optional, Sequence

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import pandas as pd


class Analyser:
    """Class for analysing a bulk of text"""

    def __init__(self, text: str, lemmatise: bool = False) -> None:
        # Tokenise the text
        self.tokens = self._tokenise(text)
        # Lemmatise the text
        if lemmatise:
            self.tokens = self._lemmatise(self.tokens)

    def frequency_distribution(self, pos: Optional[str] = None) -> FreqDist:
        """Get a frequency distribution of tokens

        Args:
            pos (Optional[str], optional): Part of speech to filter by

        Returns:
            FreqDist: Frequency distribution
        """
        # Pick tokens based on pos variable
        if pos is None:
            tokens = self.tokens
        else:
            # Interpret the tokens in terms of POS
            pos_tokens = pd.DataFrame(pos_tag(self.tokens),
                               columns=["token", "type"])
            # Filter out the specified type
            tokens = pos_tokens[pos_tokens['type'] == pos]['token']

        # Return frequency distribution
        return FreqDist(tokens)

    @staticmethod
    def _tokenise(text: str) -> Sequence[str]:
        return [
            token.lower() for token in word_tokenize(text)
        ]

    @staticmethod
    def _lemmatise(tokens: Sequence[str]) -> Sequence[str]:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
