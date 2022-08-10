# %%
import ast
from pathlib import Path

import pandas as pd
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import bigrams, trigrams

# %%
df = pd.read_csv(
    Path("Bitcoin_tweets.csv"),
    usecols=[
        "user_name",
        "user_location",
        "date",
        "text",
        "hashtags",
        "source",
    ],
    dtype={
        "user_location": str,
        "user_description": str,
        "text": str,
        "hashtags": str,
    },
)

# %%


def clean_text(s):
    """
    Remove:
    - any hashtags mention and the "#" char
    - any urls
    - any user mentions and the "@" char
    - any "RT"
    """
    text = s["text"]
    hashtags = s["hashtags"]
    # urls = s["urls"]
    # mentions = s["mentions"]

    # remove mentions
    if not pd.isnull(hashtags):
        hashtags = ast.literal_eval(
            hashtags
        )  # Evaluate literally takes the object and reads it as a string
        for hashtag in hashtags:
            text = text.replace(
                hashtag, ""
            ).strip()  # removes leading and trailling spaces

    # # remove urls
    # if not pd.isnull(urls):
    #     urls = ast.literal_eval(urls)
    #     for url in urls:
    #         text = text.replace(url["url"], "").strip()

    # # remove usernames
    # if not pd.isnull(mentions):
    #     mentions = ast.literal_eval(mentions)
    #     for mention in mentions:
    #         text = text.replace(mention["username"], "").strip()

    text = text.replace("#", "").replace("@", "").replace("RT", "").strip()

    return text


# %%


def vader_compound(text):
    """
    Calculate the tweet compound score using VADER (using cleaned text)
    - Compound score ranges from -1 (very negative) to +1 (very positive)
    - Scores equal to or less than -0.5 consider negative
    - Scores equal or more than +0.5 consider positive
    - Rest of scores consider neutral
    https://www.nltk.org/howto/sentiment.html
    """
    sid = SentimentIntensityAnalyzer()
    compound = sid.polarity_scores(text)["compound"]

    # compound ranges from -1 (most neg) to +1 (most pos)
    if compound >= 0.5:
        return "positive"
    elif compound <= -0.5:
        return "negative"
    else:
        return "neutral"


# %%
nlp = spacy.load("en_core_web_md")


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
    """

    doc = nlp(text)
    corpus = list()

    for token in doc:
        if len(token.lemma_) > 1 and not (
            token.is_stop
            or token.is_punct
            or token.is_space
            or token.like_url
            or token.is_digit
            or token.prefix_ == "@"
            or token.prefix_ == "."
        ):
            corpus.append(token.lemma_.lower())

    return corpus


# %%


def get_ner(text):
    """
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
    doc = nlp(text)

    target_ner = ["GPE", "NORP", "ORG", "PERSON", "PRODUCT", "EVENT", "LOC"]

    for ent in doc.ents:

        if ent.label_ in target_ner:

            if ent.label_ not in ner_dict:
                ner_dict[ent.label_] = [ent.text]

            else:
                ner_dict[ent.label_].append(ent.text)

    return ner_dict


# %%

#df["text_clean"] = df.apply(clean_text, axis=1)
df["sentiment"] = df["text"].apply(vader_compound)
df["ner"] = df["text"].apply(get_ner)

# %%

df["text_clean"] = df.apply(clean_text, axis=1)
df["sentiment"] = df["text_clean"].apply(vader_compound)

df["corpus_unigrams"] = df["text_clean"].apply(create_corpus)
df["corpus_bigrams"] = df["corpus_unigrams"].apply(
    lambda c: ["_".join(item) for item in bigrams(c)]
)
df["corpus_trigrams"] = df["corpus_unigrams"].apply(
    lambda c: ["_".join(item) for item in trigrams(c)]
)

df["ner"] = df["text_clean"].apply(get_ner)

# %%
