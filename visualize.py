import random
from ast import literal_eval
from operator import itemgetter

import matplotlib.pyplot as plt
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, get_single_color_func


def plot_sd_stock(df):
    # Standard deviation is an estimate for volatility
    mean = df["Close"].mean()
    std = np.std(df["Close"])

    plt.figure(figsize=(15, 15))
    plt.title("Bitcoin Standard Deviation USD")
    plt.xlabel("Days")
    plt.ylabel("Price USD")

    plt.scatter(x=df["Date"], y=df["Close"])

    plt.hlines(
        y=mean,
        xmin=df["Date"].min(),
        xmax=df["Date"].max(),
        color="y",
    )

    plt.hlines(
        y=mean - std,
        xmin=df["Date"].min(),
        xmax=df["Date"].max(),
        color="r",
    )

    plt.hlines(
        y=mean + std,
        xmin=df["Date"].min(),
        xmax=df["Date"].max(),
        color="r",
    )

    plt.hlines(
        y=mean - 2 * std,
        xmin=df["Date"].min(),
        xmax=df["Date"].max(),
        color="g",
    )

    plt.hlines(
        y=mean + 2 * std,
        xmin=df["Date"].min(),
        xmax=df["Date"].max(),
        color="g",
    )

    plt.show()


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    words_freq[:n]

    common_words = get_top_n_words(df["Review Text"], 25)
    df2 = pd.DataFrame(common_words, columns=["ReviewText", "count"])
    df2.groupby("ReviewText").sum()["count"].sort_values(ascending=False).iplot(
        kind="bar",
        yTitle="Count",
        linecolor="black",
        title="Top 25 words in reviews after removing stop words",
    )


def sentiment_worldcloud(df):
    neg_text = df[df["sentiment_label"] == "NEG"]
    pos_text = df[df["sentiment_label"] == "POS"]
    # neut_text = df[df["sentiment_label"] == "NEU"]

    # N-Gram Functions

    def get_ngram_freq(n_grams):
        """
        Function returns the frequncy of occurance of each ngram, given the list of ngrams
        """
        fdist = nltk.FreqDist(n_grams)
        for k, v in fdist.items():
            fdist[k] = v
        return fdist

    def get_ngram_dist(text, n_ngram=1):
        """
        Function gets the text and the desired lenght of ngrams
        Returns, frequency of each ngram
        """
        if n_ngram not in [1, 2, 3]:
            raise ValueError("Invalid ngram value. Expected one of: %s" % n_ngram)
        if n_ngram == 1:
            gs = nltk.word_tokenize(text)

        elif n_ngram == 2:
            gs = nltk.bigrams(text)

        elif n_ngram == 3:
            gs = nltk.trigrams(text)

        fdist = get_ngram_freq(gs)

        return fdist

    # Calulate n-grams
    neg_text = neg_text["text_spacy"].apply(literal_eval)
    pos_text = pos_text["text_spacy"].apply(literal_eval)
    # neut_text = neut_text["text_spacy"].apply(literal_eval)

    flat_text_neg = [item for sublist in neg_text for item in sublist]
    flat_text_pos = [item for sublist in pos_text for item in sublist]
    # flat_text_neut = [item for sublist in neut_text for item in sublist]

    neg_fdist = get_ngram_dist(flat_text_neg, 3)
    pos_fdist = get_ngram_dist(flat_text_pos, 3)
    # neut_fdist = get_ngram_dist(flat_text_neut, 3)

    # Most frequent ngrams

    def get_sub_most_frequnt(fdist, top_n):
        """
        Returns the top_n frequented terms
        """
        sorted_dist = sorted(fdist.items(), key=itemgetter(1), reverse=True)
        sub_sort = dict(sorted_dist[:top_n])

        sub_sort2 = {" ".join(k): v for k, v in sub_sort.items()}
        return sub_sort2

    # Subset ngrams
    neg_sub_sort = get_sub_most_frequnt(neg_fdist, 40)
    pos_sub_sort = get_sub_most_frequnt(pos_fdist, 40)
    # neut_sub_sort = get_sub_most_frequnt(neut_fdist, 40)

    def get_sentiment_for_common_ngrams(neg_sorted, pos_sorted, com_ngrams):
        """
        For ngrams appearing both in negative and positive reviews,
        assign the sentiment with more frequency.
        """

        com_2_n = []
        com_2_p = []

        for w in com_ngrams:
            cnt_n = neg_sorted[w]
            cnt_p = pos_sorted[w]
            if cnt_p > cnt_n:
                com_2_p += [w]
            else:
                com_2_n += [w]

        return com_2_p, com_2_n

    def get_pos_neg_ngrams(neg_sorted, pos_sorted, com_ngrams):
        # Function gets, most common neg and pos ngrams
        # 1) Get sets of pos, neg ngrams with common ngram appearing in only one of them
        com_2_p, com_2_n = get_sentiment_for_common_ngrams(
            neg_sorted, pos_sorted, com_ngrams
        )

        # 2) Returning unique ngrams
        neg_uniqes = list(set(neg_sorted.keys()) - com_ngrams) + com_2_n
        pos_uniqes = list(set(pos_sorted.keys()) - com_ngrams) + com_2_p

        # 3) Remove common ngrams from the one with least freq and add it to the one with most freq
        neg_uniq_dict = {k: neg_sorted[k] for k in neg_uniqes}
        pos_uniq_dict = {k: pos_sorted[k] for k in pos_uniqes}

        return pos_uniq_dict, neg_uniq_dict

    def get_all_ngrams(uniq_pos_ngrams, uniq_neg_ngrams):
        # Combine dictionary of pos and neg ngrams to get the freq of all ngrams
        all_ngrams = uniq_pos_ngrams.copy()
        all_ngrams.update(uniq_neg_ngrams)
        return all_ngrams

    def get_uniq_pos_neg_all_ngrams(neg_sub_sort, pos_sub_sort):
        # 1. Get the common ngrams appearing in both pos and neg reviews

        com_ngrams = set(neg_sub_sort.keys()) & set(pos_sub_sort.keys())
        # 2. Remove common ngrams from the one with least freq and add it to the one with most freq
        uniq_pos_ngrams, uniq_neg_ngrams = get_pos_neg_ngrams(
            neg_sub_sort, pos_sub_sort, com_ngrams
        )
        all_ngrams = get_all_ngrams(uniq_pos_ngrams, uniq_neg_ngrams)

        return uniq_pos_ngrams, uniq_neg_ngrams, all_ngrams

    # Find unique ngrams
    uniq_pos_ngrs, uniq_neg_ngrs, all_ngrs = get_uniq_pos_neg_all_ngrams(
        neg_sub_sort, pos_sub_sort
    )

    def get_normalized_frequecies(init_freq):
        # Normalize the occurance of ngram with the most frequent one
        max_cnt = max(init_freq.values())
        norm_freqs = {k: float(init_freq[k]) / max_cnt for k in init_freq.keys()}
        return norm_freqs

    # Normalise
    norm_freqs_all = get_normalized_frequecies(all_ngrs)
    norm_freqs_neg = get_normalized_frequecies(uniq_neg_ngrs)
    norm_freqs_pos = get_normalized_frequecies(uniq_pos_ngrs)

    class GroupedColorFunc(object):
        """
        Uses different colors for different groups of words.
        """

        def __init__(self, color_to_words, default_color):
            self.color_func_to_words = [
                (get_single_color_func(color), set(words))
                for (color, words) in color_to_words.items()
            ]

            self.default_color_func = get_single_color_func(default_color)

        def get_color_func(self, word):
            """Returns a single_color_func associated with the word"""
            try:
                color_func = next(
                    color_func
                    for (color_func, words) in self.color_func_to_words
                    if word in words
                )
            except StopIteration:
                color_func = self.default_color_func

            return color_func

        def __call__(self, word, **kwargs):
            return self.get_color_func(word)(word, **kwargs)
            return self.get_color_func(word)(word, **kwargs)

        # Define functions to select a hue of colors arounf: grey, red and green

    def red_color_func(
        word, font_size, position, orientation, random_state=None, **kwargs
    ):
        return "hsl(0, 100%%, %d%%)" % random.randint(30, 50)

    def green_color_func(
        word, font_size, position, orientation, random_state=None, **kwargs
    ):
        return "hsl(100, 100%%, %d%%)" % random.randint(20, 40)

    def plot_pos_neg_wordclouds(neg_ngrams_sort, pos_ngrams_sort):
        fig = plt.figure(figsize=(16, 12))
        plt.subplot(121)

        wc1 = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=20,
            min_font_size=8,
        ).generate_from_frequencies(neg_ngrams_sort)

        plt.imshow(
            wc1.recolor(color_func=red_color_func, random_state=3),
            interpolation="bilinear",
        )
        axis("off")

        wc2 = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=20,
            min_font_size=8,
        ).generate_from_frequencies(pos_ngrams_sort)

        plt.subplot(122)

        plt.imshow(
            wc2.recolor(color_func=green_color_func, random_state=3),
            interpolation="bilinear",
        )
        axis("off")
        show()

    # Plot
    plot_pos_neg_wordclouds(norm_freqs_neg, norm_freqs_pos)

    def plot_allwords_wordclouds(norm_freqs_all):

        wc = WordCloud(
            width=1200,
            height=800,
            background_color="white",
            max_words=200,
            min_font_size=10,
        ).generate_from_frequencies(norm_freqs_all)

        color_to_words = {
            # words below will be colored with a green single color function
            "#00ff00": uniq_pos_ngrs.keys(),
            # will be colored with a red single color function
            "red": uniq_neg_ngrs.keys(),
        }

        # Words that are not in any of the color_to_words values
        # will be colored with a grey single color function
        default_color = "grey"

        # Create a color function with multiple tones
        grouped_color_func = GroupedColorFunc(color_to_words, default_color)

        # Apply our color function
        wc.recolor(color_func=grouped_color_func)

        plt.figure(figsize=(16, 12))

        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    # plot
    plot_allwords_wordclouds(norm_freqs_all)
