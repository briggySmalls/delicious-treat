"""Analysis of a set of text"""
from string import punctuation

import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.probability import ConditionalFreqDist, FreqDist, defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


def _penn2morphy(tag: str) -> str:
    if tag.startswith('NN'):
        return str(wn.NOUN)
    if tag.startswith('VB'):
        return str(wn.VERB)
    if tag.startswith('JJ'):
        return str(wn.ADJ)
    if tag.startswith('RB'):
        return str(wn.ADV)
    # Default to noun
    return str(wn.NOUN)


def _analyse(messages: pd.DataFrame, lemmatise: bool = False) -> pd.DataFrame:
    # Convert to tokens
    tokens = _tokenise(messages['message'])
    # Get part of speech
    part_of_speech = _part_of_speech(tokens)
    # Lemmatise, if required
    if lemmatise:
        part_of_speech = _lemmatise(part_of_speech)
    # Build a list of tokens we wish to ignore
    to_ignore = set(stopwords.words("english")).union(set(punctuation))
    # Filter the tokens
    valid_tokens = ~part_of_speech['token'].isin(to_ignore)
    return part_of_speech[valid_tokens]


def _tokenise(messages: pd.Series) -> pd.DataFrame:
    tokens = []
    for i, message in enumerate(messages):
        # Convert message to tokens
        new_tokens = [
            token.lower() for sentence in sent_tokenize(message)
            for token in word_tokenize(sentence)
        ]
        # Record token with corresponding message index
        tokens.extend([(token, i) for token in new_tokens])

    return pd.DataFrame(tokens, columns=['token', 'message_idx'])


def _part_of_speech(tokens: pd.DataFrame) -> pd.DataFrame:
    # Process text to identify part of speech
    pos = pd.DataFrame(pos_tag(tokens['token']), columns=["token", "tag"])
    # Keep record of message index
    pos['message_idx'] = tokens['message_idx']
    return pos


def _lemmatise(part_of_speech: pd.DataFrame) -> pd.DataFrame:
    # Create a lemmatiser
    lemmer = WordNetLemmatizer()

    def lemmatise_row(row: pd.Series) -> pd.Series:
        (token, tag, message_idx) = row
        new_tag = _penn2morphy(tag)
        new_token = lemmer.lemmatize(token, pos=new_tag)
        return pd.Series([new_token, new_tag, message_idx],
                         index=["token", "tag", "message_idx"])

    # Lemmatise tokens
    return part_of_speech.apply(lemmatise_row, axis=1)


def _freq_dist(part_of_speech: pd.DataFrame) -> FreqDist:
    dist = ConditionalFreqDist()
    for _, (token, tag, _) in part_of_speech.iterrows():
        dist[tag][token] += 1
    return dist


class Analyser:
    """Class to analyse messages"""
    def __init__(self, messages: pd.Series) -> None:
        self.messages = messages
        self.tokens = None

    def analyse(self) -> None:
        """Run analysis (slow)"""
        self.tokens = _analyse(self.messages, lemmatise=True)

    def freq_dist(self, pos: bool = False) -> defaultdict:
        """Obtain an nltk frequency distribution

        Args:
            pos (bool, optional): Whether to subdivide frequency distrubtion
                                  by part of speech

        Returns:
            defaultdict: Frequency distribution
        """
        assert self.tokens is not None
        if not pos:
            return FreqDist(self.tokens['token'])

        dist = ConditionalFreqDist()
        for _, (token, tag, _) in self.tokens.iterrows():
            dist[tag][token] += 1
        return dist

    def filter_messages(self, token: str) -> pd.DataFrame:
        """Find messages with the given token

        Args:
            token (str): Token to filter for

        Returns:
            pd.DataFrame: Messages that contain the token
        """
        assert self.tokens is not None
        message_idxs = self.tokens[self.tokens['token'] == token].message_idx
        return self.messages.iloc[message_idxs]
