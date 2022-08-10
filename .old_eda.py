# # %%
# import random
# import re
# import string
# from collections import OrderedDict
# from datetime import datetime
# from operator import itemgetter
# from pathlib import Path

# import matplotlib.pyplot as plt
# import mplfinance as fplt
# import nltk
# import numpy as py
# import pandas as pd
# import spacy
# from PIL import Image
# from spacymoji import Emoji
# from transformers import AutoConfig, AutoTokenizer, pipeline
# from wordcloud import STOPWORDS, WordCloud, get_single_color_func

# %pylab inline

# # %% Missing data summary
# df_tweets.isna().sum()

# # %% Create column of mentions in tweets
# df_tweets["mention"] = df_tweets.text.str.findall(r"(?<![@\w])@(\w{1,25})").apply(
#     ",".join
# )

# #%% General text cleaning


# def clean_text(text):
#     """
#     Use regex to remove unwanted symbols and text
#     """
#     text = re.sub(r"@\w+", "", text)  # Remove mentions
#     text = re.sub(r"@+", "", text)  # Remove @
#     text = re.sub(r"\n+", "", text)  # Remove new line notation
#     text = re.sub(r"#+", "", text)  # Remove #
#     text = re.sub(r"https?:\/\/\S+", "", text)  # Remove links
#     text = re.sub("\s+", " ", text)  # Remove whitespace
#     text = text.strip()

#     return text


# # %% Run cleaning
# df_tweets["text_clean"] = df_tweets["text"].apply(clean_text)

# # %% Load spacy Cleaning
# nlp = spacy.load("en_core_web_md", disable=["ner"])
# emoji = Emoji(nlp)
# nlp.add_pipe("emoji")

# # %% Spacy cleaning/tokenize etc...


# def create_corpus(text):
#     """
#     Use spacy to create a corpus from each tweet (using cleaned text)
#     - A corpus is a list of tokens that are:
#     - Not stop words
#     - Not spaces
#     - Don not look like a URL
#     - Not numbers
#     - Does not start with "@" or "."
#     - Also, use the token lemma (i.e. base form of the token, with no inflectional suffixes)
#     - Include Emojis
#     """
#     doc = nlp(
#         text.lower()
#     )  # For tokenization casing doesn't matter, for sentiment it matters.
#     corpus = list()

#     for token in doc:
#         # if token._.is_emoji is True:
#         #     corpus.append(token._.emoji_desc)
#         if len(token.lemma_) > 1 and not (
#             token.is_stop
#             or token.is_punct
#             or token.is_space
#             or token.like_url
#             or token.is_digit
#             or token.prefix_ == "@"
#             or token.prefix_ == "#"
#             or token.prefix_ == "$"
#             or token.prefix_ == "."
#         ):
#             corpus.append(token.lemma_)

#     return corpus


# # %% Run Spacy Cleaning
# df_tweets["text_spacy"] = df_tweets["text_clean"].apply(create_corpus)

# # %% Sentiment Analysis
# checkpoint = "pysentimiento/robertuito-sentiment-analysis"

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# # batch_sentences = 'df_tweets_clean["text"].iloc[3635]'
# # tokenizer(
# #     batch_sentences, padding=True, truncation=True, return_tensors="pt", max_length=100
# # )

# pipe = pipeline(model=checkpoint, truncation=True, max_length=100)

# so_dict = df_tweets["text_clean"].apply(pipe)

# labels = []
# scores = []
# for so in so_dict:
#     labels.append(so[0]["label"])
#     scores.append(so[0]["score"])

# df_tweets["sentiment_label"] = labels
# df_tweets["sentiment_score"] = scores
# df_tweets.to_csv("Bitcoin_tweets_processed_recent.csv", encoding="utf-8")
