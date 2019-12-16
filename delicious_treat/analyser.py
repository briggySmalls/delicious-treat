"""Analysis of a set of text"""
from typing import Optional

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pandas as pd


class Analyser:
    """Class for analysing a bulk of text"""

    def __init__(self, text: str):
        # Tokenise the text
        self.tokens = [
            token.lower() for token in word_tokenize(text)
        ]

    def frequency_distribution(self, pos: Optional[str] = None) -> FreqDist:
        """Get a frequency distribution of tokens

        Args:
            pos (Optional[str], optional): Part of speech to filter by

        Returns:
            TYPE: Frequency distribution
        """
        # Pick tokens based on pos variable
        if pos is None:
            tokens = self.tokens
        else:
            # Interpret the tokens in terms of POS
            pos = pd.DataFrame(pos_tag(self.tokens),
                               columns=["token", "type"])
            # Filter out the specified type
            tokens = pos.loc[pos['type'] == pos, 'token']

        # Return frequency distribution
        return FreqDist(tokens)
