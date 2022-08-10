# %%
import mplfinance as fplt
import pandas as pd

from load import load_stocks, load_tweets
from nlp_tasks import create_corpus, sentiment_extraction
from preprocess import cleaning
from visualize import plot_sd_stock, sentiment_worldcloud

# from nltk.util import bigrams, trigrams


# %%
# Load data
df_tweets = load_tweets("bitcoin_tweets.csv")
df_stocks = load_stocks("bitcoin_history.csv")


# %%
# Preprocessing

# Clean tweets
df_tweets = cleaning(
    df_tweets,
    cols=["text"],
)

# Ensure all text contains 'btc' or 'Bitcoin'
# As per extraction guidelines
# 41,182 rows dropped
df_tweets = df_tweets[
    df_tweets.text.str.contains(
        "btc|bitcoin",
        case=False,
        regex=True,
    )
]


# %%
# EDA

# Stock data

# Set datetime index
df_stocks["Date"] = pd.to_datetime(df_stocks["Date"])
df_stocks = df_stocks.set_index("Date")

# Candlestick plot
fplt.plot(
    df_stocks,
    type="candle",
    style="yahoo",
    title="Bitcoin (USD) 2021",
    ylabel="Price ($)",
)

# Standard deviation plot
plot_sd_stock(df_stocks.reset_index())


# %%
# NLP Tasks

# Create column of mentions
df_tweets["mention"] = df_tweets.text.str.findall(
    r"(?<![@\w])@(\w{1,25})",
).apply(",".join)

# Create corpus and named entity columns
df_tweets[["corpus", "ner"]] = df_tweets.apply(
    lambda x: create_corpus(x["text"]),
    axis=1,
    result_type="expand",
)

# Calculate sentiment of tweets
df_tweets = sentiment_extraction(df_tweets)

# Save output
df_tweets.to_csv("bitcoin_tweets_processed.csv")


# %%
# EDA

# Twitter sentiment

# df_tweets["corpus_bigrams"] = df_tweets["corpus_unigrams"].apply(
#     lambda c: ["_".join(item) for item in bigrams(c)]
# )
# df_tweets["corpus_trigrams"] = df_tweets["corpus_unigrams"].apply(
#     lambda c: ["_".join(item) for item in trigrams(c)]
# )

sentiment_worldcloud(df_tweets)
