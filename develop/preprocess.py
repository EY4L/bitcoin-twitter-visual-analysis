import numpy as np


def cleaning(df, cols):
    # Removing leading, trailing, 2+ whitespaces
    # Replace empty strings with NaN
    df[cols] = (
        df[cols]
        .applymap(lambda cell: cell.strip())
        .replace(
            to_replace=r"\s+",
            value=" ",
            regex=False,
        )
        .replace(
            to_replace="",
            value=np.nan,
            regex=False,
        )
    )

    return df
