"""Analysis of a set of text"""
from string import punctuation
from typing import Sequence

import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


def _penn2morphy(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    if tag.startswith('VB'):
        return wn.VERB
    if tag.startswith('JJ'):
        return wn.ADJ
    if tag.startswith('RB'):
        return wn.ADV
    # Default to noun
    return wn.NOUN


def _analyse(text: str, lemmatise: bool = False) -> pd.DataFrame:
    # Convert to tokens
    tokens = _tokenise(text)
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


def _tokenise(text: str) -> Sequence[str]:
    # First convert into tokens
    return [
        token.lower() for sentence in sent_tokenize(text)
        for token in word_tokenize(sentence)
    ]


def _part_of_speech(tokens: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(pos_tag(tokens), columns=["token", "tag"])


def _lemmatise(part_of_speech: pd.DataFrame) -> pd.DataFrame:
    # Create a lemmatiser
    lemmer = WordNetLemmatizer()

    def lemmatise_row(row: pd.Series) -> pd.Series:
        (token, tag) = row
        new_tag = _penn2morphy(tag)
        new_token = lemmer.lemmatize(token, pos=new_tag)
        return pd.Series([new_token, new_tag], index=["token", "tag"])

    # Lemmatise tokens
    return part_of_speech.apply(lemmatise_row, axis=1)


def _freq_dist(part_of_speech: pd.DataFrame) -> FreqDist:
    dist = ConditionalFreqDist()
    for _, (token, tag) in part_of_speech.iterrows():
        dist[tag][token] += 1
    return dist
