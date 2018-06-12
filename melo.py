#!/usr/bin/env python3

from collections import defaultdict
from datetime import datetime
from scipy.special import erf, erfinv

import numpy as np

import matplotlib.pyplot as plt


class MarginElo:
    """
    Margin-depedent Elo ratings and predictions inspired by the
    Bradley-Terry model.

    Described in https://arxiv.org/abs/1802.00527

    """
    def __init__(self, times, labels1, labels2, values, nlines=200, k=1e-4):
        """
        To do, document this.

        """
        self.comparisons = np.sort(
            np.rec.fromarrays(
                [times, labels1, labels2, values],
                dtype=[
                    ('time', 'M8[us]'),
                    ('label1',   'U8'),
                    ('label2',   'U8'),
                    ('value',    'f8'),
                ]
            ), axis=0
        )

        self.nlines = nlines

        self.k = k

        self.lines, self.ratings = self.rate(
            self.comparisons, self.nlines
        )

    def norm_cdf(self, x, loc=0, scale=1):
        """
        Normal cumulative probability distribution

        """
        return 0.5*(1 + erf((x - loc)/(np.sqrt(2)*scale)))

    def naive_prior(self, values, lines):
        """
        Initial ratings which respect the naive probability P(value > line),
        determined from the minimum bias value distribution.

        """
        prob = np.array(
            [(values > line).sum() / values.size for line in lines],
            dtype=float, ndmin=1
        )
        return np.sqrt(2)/2*erfinv(2*prob - 1)

    def query_rating(self, time, label):
        """
        Find the last rating preceeding the specified 'time'.

        """
        ratings = [r for t, r in self.ratings[label] if t < time]

        return ratings.pop()

    def rate(self, comparisons, nlines):
        """
        Apply the margin dependent Elo model to the list of binary comparisons.

        """
        median = np.median(comparisons.value)
        std = np.std(comparisons.value)

        # standardize the data
        comparisons.value -= median
        comparisons.value /= std

        # create standardized comparison lines
        extremum = np.percentile(np.abs(comparisons.value), 99.9)
        lines = np.linspace(-extremum, extremum, self.nlines)

        # determine naive prior
        default_rating = self.naive_prior(comparisons.value, lines)
        current_rating = defaultdict(lambda: default_rating)
        ratings = defaultdict(list)

        for (time, label1, label2, value) in comparisons:

            # lookup ratings
            rating1 = current_rating[label1]
            rating2 = current_rating[label2]

            # prior and posterior outcome probabilities
            prior = self.norm_cdf(rating1 - rating2[::-1])
            post = np.heaviside(value - lines, 0.5)

            # rating change
            rating_change = self.k * (post - prior)

            # save current rating
            current_rating[label1] = rating1 + rating_change
            current_rating[label2] = rating2 - rating_change

            # record ratings
            ratings[label1].append((time, rating1 + rating_change))
            ratings[label2].append((time, rating2 - rating_change))

        lines *= std
        lines += median

        return lines, ratings

    def prior_cdf(self, time, label1, label2):
        """
        Prior cumulative probability distribution for a comparison between
        label1 and label2 at specified 'time'.

        """
        # lookup ratings
        rating1 = self.query_rating(time, label1)
        rating2 = self.query_rating(time, label2)

        # prior and posterior outcome probabilities
        prior = self.norm_cdf(rating1 - rating2[::-1])

        return self.lines, prior

    def samples(self, time, label1, label2, size=10**6):
        """
        Draw random samples from the prior.

        """
        lines, prior = self.prior_cdf(time, label1, label2)
        assert np.all(np.diff(prior[::-1]) > 0)
        return np.interp(np.random.rand(size), prior[::-1], lines[::-1])

def main():

    size = 10**5
    today = datetime.today()
    time = datetime(today.year, today.month, today.day)
    times = [np.datetime64(time) - np.timedelta64(n, 's') for n in range(size)]
    labels1 = size*['CLE']
    labels2 = size*['MIA']
    values = np.random.normal(1, 2, size=size)

    elo = MarginElo(times, labels1, labels2, values)

    today = datetime.today()
    time = np.datetime64(datetime(today.year, today.month, today.day))

    true = np.random.normal(1, 2, size=10**6)
    pred = elo.samples(time, 'CLE', 'MIA')

    for samples in (true, pred):
        plt.hist(samples, histtype='step', bins=100, density=True)

    plt.show()


if __name__ == "__main__":
    main()
