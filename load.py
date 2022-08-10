# %%
from pathlib import Path

import pandas as pd


def load_tweets(path):
    # Parse date columns, drop nans in text
    df = pd.read_csv(
        Path(path),
        usecols=[
            "user_name",
            "user_location",
            "user_followers",
            "date",
            "text",
        ],
        dtype={
            "user_name": str,
            "user_location": str,
            "user_followers": float,
            "user_description": str,
            "text": str,
        },
    ).dropna(
        subset=["text"],
    )

    # Call dt.date to only get year, month, date instead of time
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.date

    # Drop rows that could not be parsed into datetime
    df = df.dropna(
        subset=["date"],
    )

    return df


def load_stocks(path):
    # Parse date columns
    df = pd.read_csv(
        Path(path),
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    ).dt.date

    return df


# %%
