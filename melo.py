#!/usr/bin/env python3

from collections import defaultdict
from datetime import datetime
import random
from scipy.special import erf, erfinv

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA


today = datetime.today()

class ToyModel:
    def __init__(self, nteams=32):
        self.nteams = nteams
        self.teams = ['team_{}'.format(n) for n in range(nteams)]
        self.means = [np.random.uniform(10, 50) for n in range(nteams)]

    def games(self, size=10**5):

        times = []
        labels1 = []
        labels2 = []
        values = []

        for n in range(size):
            i1, i2 = random.sample(range(self.nteams), 2)
            score1 = np.random.normal(self.means[i1], 5)
            score2 = np.random.normal(self.means[i2], 5)

            times.append(np.datetime64(today) - np.timedelta64(n, 's'))
            labels1.append(self.teams[i1])
            labels2.append(self.teams[i2])
            values.append(score1 - score2)

        return times, labels1, labels2, values


class MarginElo:
    """
    Margin-depedent Elo ratings and predictions inspired by the
    Bradley-Terry model.

    Described in https://arxiv.org/abs/1802.00527

    """
    def __init__(self, times, labels1, labels2, values, nlines=10, k=0.05):
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

        lmin, lmax = np.percentile(self.comparisons.value, [0.01, 99.99])
        self.lines = np.linspace(lmin, lmax, nlines)

        self.k = k

        self.rating_hcap = self.naive_prior(self.comparisons.value, self.lines)

        self.ratings = self.rate(self.comparisons, self.lines)

    def norm_cdf(self, x, loc=0, scale=1):
        """
        Normal cumulative probability distribution

        """
        return 0.5*(1 + erf((x - loc)/(np.sqrt(2)*scale)))

    def naive_prior(self, values, lines):
        """
        Rating difference, dR = P(values - lines > 0), that the observed values
        cover the specified lines.

        """
        prob = np.array(
            [(values > line).sum() / values.size for line in lines],
            dtype=float, ndmin=1
        )

        return np.sqrt(2)*erfinv(2*prob - 1)

    def query_rating(self, time, label):
        """
        Find the last rating preceeding the specified 'time'.

        """
        return list(r for t, r in self.ratings[label] if t < time).pop()

    def rate(self, comparisons, lines):
        """
        Apply the margin dependent Elo model to the list of binary comparisons.

        """
        current_rating = defaultdict(lambda: np.zeros_like(lines))
        ratings = defaultdict(list)

        for (time, label1, label2, value) in comparisons:

            # lookup team ratings
            rating1 = current_rating[label1]
            rating2 = current_rating[label2]

            # prior and posterior outcome probabilities
            prior = self.norm_cdf(rating1 - rating2 + self.rating_hcap)
            post = np.where(value - lines > 0, 1, 0)

            # rating change
            rating_change = self.k * (post - prior)

            # save current rating
            current_rating[label1] += rating_change
            current_rating[label2] -= rating_change

            # record ratings
            ratings[label1].append((time, current_rating[label1]))
            ratings[label2].append((time, current_rating[label2]))

        return ratings

    def prior_cdf(self, time, label1, label2):
        """
        Prior cumulative probability distribution for a comparison between
        label1 and label2 at specified 'time'.

        """
        rating1 = self.query_rating(time, label1)
        rating2 = self.query_rating(time, label2)

        prior = np.sort(1 - self.norm_cdf(rating1 - rating2 + self.rating_hcap))

        return self.lines, prior

    def samples(self, time, label1, label2, size=10**6):
        """
        Draw random samples from the prior.

        """
        lines, prior = self.prior_cdf(time, label1, label2)

        return np.interp(np.random.rand(size), prior, lines)


def main():
    tm = ToyModel(nteams=32)
    melo = MarginElo(*tm.games(10**5))
    minbias = melo.comparisons.value

    pred = melo.samples(np.datetime64(today), 'team_0', 'team_1', size=10**7)

    true = np.random.normal(tm.means[0], 5, 10**7) - np.random.normal(tm.means[1], 5, 10**7)

    plt.boxplot([minbias, pred, true], showfliers=False)
    plt.show()


if __name__ == "__main__":
    main()
