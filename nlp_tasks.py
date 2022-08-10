import numpy as np
import spacy
from spacymoji import Emoji
from transformers import pipeline

nlp = spacy.load("en_core_web_md")
emoji = Emoji(nlp)
nlp.add_pipe("emoji")


def create_corpus(text):
    """
    Use spacy to create a corpus from each tweet (using cleaned text)
    - A corpus is a list of tokens that are:
    - Not stop words
    - Not spaces
    - Don not look like a URL
    - Not numbers
    - Does not start with "@" or "."
    - Also, use the token lemma (i.e. base form of the token, with no inflectional suffixes)
    - Include Emojis

    Use a trained spacy model (https://spacy.io/models/en#en_core_web_md) to recognize
    the following entities (using cleaned text):
    - "GPE": 'Countries, cities, states'
    - "NORP": 'Nationalities or religious or political groups'
    - "ORG": Companies, agencies, institutions, etc.'
    - "PERSON": 'People, including fictional'
    - "PRODUCT": 'Objects, vehicles, foods, etc. (not services)'
    - "EVENT": 'Named hurricanes, battles, wars, sports events, etc.'
    - "LOC": 'Non-GPE locations, mountain ranges, bodies of water'
    """
    ner_dict = {}
    target_ner = ["GPE", "NORP", "ORG", "PERSON", "PRODUCT", "EVENT", "LOC"]

    # For tokenization casing doesn't matter, for sentiment it matters
    # lowercase all text
    doc = nlp(text.lower())
    corpus = list()

    for token in doc:
        if len(token.lemma_) > 1 and not (
            token.is_stop
            or token.is_punct
            or token.is_space
            or token.like_url
            or token.is_digit
            or token.prefix_ == "@"
            or token.prefix_ == "#"
            or token.prefix_ == "$"
            or token.prefix_ == "."
        ):
            corpus.append(token.lemma_)

    for ent in doc.ents:

        if ent.label_ in target_ner:

            if ent.label_ not in ner_dict:
                ner_dict[ent.label_] = [ent.text]

            else:
                ner_dict[ent.label_].append(ent.text)

    if len(ner_dict) == 0:
        return corpus, np.nan
    else:
        return corpus, ner_dict


def sentiment_extraction(df):
    # Expects the df to have col 'text'
    checkpoint = "pysentimiento/robertuito-sentiment-analysis"

    pipe = pipeline(
        model=checkpoint,
        truncation=True,
        max_length=100,
    )

    so_dict = df["text"].apply(pipe)

    labels = []
    scores = []
    for so in so_dict:
        labels.append(so[0]["label"])
        scores.append(so[0]["score"])

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    return df
