# %%
from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as fplt
import numpy as np
import pandas as pd
import seaborn as sns
from load import load_processed_tweets, load_stocks, load_tweets
from nlp_tasks import create_corpus, sentiment_extraction
from nltk import FreqDist
from preprocess import cleaning
from visualize import plot_sd_stock, sentiment_worldcloud

# %%
# Load data
df_tweets = load_tweets(Path("data", "raw", "bitcoin_tweets.csv"))
df_stocks = load_stocks(Path("data", "raw", "bitcoin_history.csv"))

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
# df_tweets.to_csv("bitcoin_tweets_processed.csv")


# %%
# Load processed data
df = load_processed_tweets("bitcoin_tweets_processed.csv")

# Create datetime format column
df["date"] = pd.to_datetime(df["date"])

# EDA
# Unigram
sentiment_worldcloud(
    df,
    1,
)

# Bigram
sentiment_worldcloud(
    df,
    2,
)

# Trigram
sentiment_worldcloud(
    df,
    3,
)


# %%
# CORRELLATION SENTIMENT vs CLOSE PRICE ####

# Count of sentiment grouped by week
df_weeks = (
    df.groupby(
        [
            "sentiment_label",
            pd.Grouper(
                key="date",
                freq="W-MON",
            ),
        ]
    )["text"]
    .count()
    .reset_index()
    .sort_values("date")
)

# Dataframe to fill empty values
df_weeks_join = pd.DataFrame(
    data={
        "text": np.zeros(44),
        "sentiment_label": ["NAN" for s in range(0, 44)],
        "date": pd.date_range(
            start="2021-02-03",
            end="2021-12-06",
            freq="W-MON",
        ),
    }
)

# Count of sentiment grouped by week with empty dates filled
df_weeks = pd.concat(
    [df_weeks, df_weeks_join],
)

# Turn count into whole numbers
# df_weeks["text"] = df_weeks["text"].astype(int)
df_weeks = df_weeks[df_weeks.sentiment_label != "NEU"]

# Create total count of sentiment by week
total = (
    df.groupby(
        [
            pd.Grouper(
                key="date",
                freq="W-MON",
            ),
        ]
    )["text"]
    .count()
    .reset_index()
    .sort_values("date")
)

# Create datetime column
# df_stocks = df_stocks.reset_index()
df_stocks["Date"] = pd.to_datetime(df_stocks["Date"])

# Average stock price per week
df_stocks_weeks = (
    df_stocks.groupby(
        [
            pd.Grouper(
                key="Date",
                freq="W-MON",
            ),
        ]
    )
    .mean()
    .reset_index()
    .sort_values("Date")
)

# Sentiment per week vs Stock price
sns.set_theme(style="whitegrid")

# Initialize the matplotlib figure
fig, ax = plt.subplots(
    figsize=(
        12,
        6,
    )
)

# Initialize lineplot
sns.lineplot(
    data=df_stocks_weeks[["Date", "Close"]],
    color="y",
    marker="o",
    ci=None,
    ax=ax,
)

sns.set_color_codes("pastel")
ax1 = ax.twinx()

# Initialize barplot
sns.barplot(
    y="text",
    x=df_weeks["date"].dt.date,
    data=df_weeks,
    hue="sentiment_label",
    hue_order=[
        "POS",
        "NEG",
    ],
    alpha=0.8,
    ci=None,
    ax=ax1,
)

# Set legends for axis
ax.legend(
    ncol=2,
    loc="upper right",
    frameon=True,
)

ax1.legend(
    ncol=2,
    loc="upper left",
    frameon=True,
)

ax.set(
    ylabel="Close Price",
    xlabel="Date",
)

ax1.set(
    ylabel="Count",
)

sns.despine(
    left=True,
    bottom=True,
)

plt.title("Count of tweets with sentiment ")
plt.setp(ax.get_xticklabels(), rotation=35)

# Total tweets per week
sns.set_theme(style="whitegrid")
f1, ax11 = plt.subplots(
    figsize=(
        12,
        6,
    )
)

sns.barplot(
    y="text",
    x=total["date"].dt.date,
    data=total,
    ci=None,
    ax=ax11,
    color="b",
)
ax11.set(ylabel="Count")
plt.setp(ax11.get_xticklabels(), rotation=55)
plt.title("Total tweets per week")


# CORRELLATION SENTIMENT vs CLOSE PRICE ####
# %%

df = load_processed_tweets("bitcoin_tweets_processed.csv")

df["date"] = pd.to_datetime(df["date"])

# %%
# NUMBER OF MENTIONS PER WEEK
# Group mentions by week, concatenate all strings into a list
df_test = df.groupby([pd.Grouper(key="date", freq="M",)])[
    "mention"
].apply(list)

# Split strings e.g. 'elonmusk, youtube' into 'elonmusk' , 'youtube'
temp1 = df_test.apply(
    lambda x: [x[i].split(",") for i in range(len(x)) if type(x[i]) == str]
)
temp2 = temp1.apply(lambda l: [item for sublist in l for item in sublist])

# Create dciontaries, keys as mentions and value as count
df_test1 = temp2.apply(lambda x: FreqDist((x)))


def deldict(x):
    try:
        x.pop(np.nan)
    except KeyError:
        pass
    return x


eyal = df_test1.to_frame()

# Remove nan from mentions
df_test11 = eyal.apply(
    lambda x: deldict(x["mention"]),
    axis=1,
)

# Sort my most common and convert to list
y1 = df_test11.apply(lambda x: x.most_common()[0:5])

y1 = y1.to_frame()

y3 = pd.DataFrame(
    y1[0].tolist(),
    index=y1.index,
    columns=[
        "1",
        "2",
        "3",
        "4",
        "5",
    ],
)  # "6", "7", "8", "9", "10",

y4 = pd.melt(y3.reset_index(), id_vars="date",).drop(
    "variable",
    axis=1,
)
y4[["mention", "count"]] = pd.DataFrame(y4["value"].tolist())
y4 = y4.drop(
    "value",
    axis=1,
).sort_values("date")


y4["count_norm"] = (y4["count"] - y4["count"].min()) / (
    y4["count"].max() - y4["count"].min()
)

y4["date"] = y4["date"].dt.month_name()
g = sns.FacetGrid(
    y4,
    col="date",
    height=3.5,
    col_wrap=4,
    palette=sns.color_palette(
        "husl",
        len(y4["mention"]),
    ),
)
g.map_dataframe(
    sns.barplot,
    data=y4,
    y="count_norm",
    x="mention",
    hue="mention",
    dodge=True,
    ci=False,
)

g.set_titles("Normalised count of mentions per month")
# NUMBER OF MENTIONS PER WEEK
# %%

# SENTIMENT OF TWEETS PER MENTION
df_mention_sent = df[
    df["mention"].str.contains(
        "|".join(y4["mention"].tolist()),
        na=False,
    )
]

# Turn mentions into lists
df_mention_sent["mention"] = df_mention_sent["mention"].str.split(",")

# Turn list into duplicated rows
df_mention_sent = df_mention_sent.explode("mention")

# Group by mention and count of each sentiment label
df_mention_sent = df_mention_sent.groupby(
    [
        "mention",
        "sentiment_label",
    ]
)["text"].count()
df_mention_sent = df_mention_sent.to_frame()

# Remove low mentions

# df_mention_sent.describe()
# count	57522.000000
# mean	5.717273
# std	107.264415
# min	1.000000
# 25%	1.000000
# 50%	1.000000
# 75%	2.000000
# max	17332.000000

# Remove neutral mentions
df_mention_sent = df_mention_sent[df_mention_sent["text"] > 115]
df_mention_sent = df_mention_sent.reset_index()
# df_mention_sent = df_mention_sent[df_mention_sent["sentiment_label"]  "NEU"]
df_mention_sent["text_norm"] = (
    df_mention_sent["text"] / df_mention_sent["text"].abs().max()
)
df_mention_sent = df_mention_sent.sort_values(["text_norm"])
# %%
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(
    figsize=(
        12,
        30,
    )
)
# Draw a nested barplot by species and sex
sns.barplot(
    data=df_mention_sent,
    y="mention",
    x="text_norm",
    hue="sentiment_label",
    ci="sd",
    palette="dark",
    alpha=0.6,
    orient="h",
    ax=ax,
)
plt.setp(ax.get_xticklabels(), rotation=35)

# %%
