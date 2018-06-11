#!/usr/bin/env python3

from collections import defaultdict
from datetime import datetime
from scipy.special import erf, erfinv

import numpy as np

import matplotlib.pyplot as plt


today = datetime.today()
time = np.datetime64(datetime(today.year, today.month, today.day))

games = []

for rand in np.random.normal(1, 2, size=10**5):
    time -= np.timedelta64(1, 's')
    games.append((time, 'CLE', 'MIA', rand))

class MarginElo:
    def __init__(self, comparisons, nlines=100, k=0.0001):
        """
        MarginElo takes one required argument 'comparisons' which is a list of,

        (date, alpha, beta, value)

        where 'date' is a Python datetime object, 'alpha' and 'beta' are names
        (strings) of the entities being compared, and 'value' is the relative
        comparison value, e.g. a score difference.

        """
        self.comparisons = np.sort(
            np.array(
                comparisons, dtype=[
                    ('time', 'M8[us]'),
                    ('alpha',    'U8'),
                    ('beta',     'U8'),
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
        Initialize ratings that respect the naive probability P(value > line),
        determined from the minimum bias value distribution.

        """
        prob = np.array(
            [(values > line).sum() / values.size for line in lines],
            dtype=float, ndmin=1
        )
        return np.sqrt(2)/2*erfinv(2*prob - 1)

    def query_rating(self, time, name):
        """
        Find the last rating preceeding the specified 'time'.

        """
        ratings = [r for t, r in self.ratings[name] if t < time]

        return ratings.pop()

    def rate(self, comparisons, nlines):
        """
        Apply the margin dependent Elo model to the list of binary comparisons.

        """
        median = np.median(comparisons['value'])
        std = np.std(comparisons['value'])

        # standardize the data
        comparisons['value'] -= median
        comparisons['value'] /= std

        # create standardized comparison lines
        extremum = np.percentile(np.abs(comparisons['value']), 99.9)
        lines = np.linspace(-extremum, extremum, self.nlines)

        # determine naive prior
        default_rating = self.naive_prior(comparisons['value'], lines)
        current_rating = defaultdict(lambda: default_rating)
        ratings = defaultdict(list)

        for (time, alpha, beta, value) in comparisons:

            # lookup ratings
            alpha_rtg, beta_rtg = [
                current_rating[name] for name in (alpha, beta)
            ]

            # ratings difference
            rtg_diff = alpha_rtg - beta_rtg[::-1]

            # prior and posterior outcome probabilities
            prior = self.norm_cdf(rtg_diff)
            post = np.heaviside(value - lines, 0.5)

            # rating change
            rtg_change = self.k * (post - prior)

            # save current rating
            current_rating[alpha] = alpha_rtg + rtg_change
            current_rating[beta] = beta_rtg - rtg_change

            # record ratings
            ratings[alpha].append((time, alpha_rtg + rtg_change))
            ratings[beta].append((time, beta_rtg - rtg_change))

        lines *= std
        lines += median

        return lines, ratings

    def prior_cdf(self, time, alpha, beta):
        """
        Prior cumulative probability distribution for a comparison between
        alpha and beta at specified 'time'.

        """
        # lookup ratings
        alpha_rtg, beta_rtg = [
            self.query_rating(time, name) for name in (alpha, beta)
        ]

        # ratings difference
        rtg_diff = alpha_rtg - beta_rtg[::-1]

        # prior and posterior outcome probabilities
        prior = self.norm_cdf(rtg_diff)

        return self.lines, prior

    def samples(self, time, alpha, beta, size=10**6):
        """
        Draw random samples from the prior.

        """
        lines, prior = self.prior_cdf(time, alpha, beta)
        assert np.all(np.diff(prior[::-1]) > 0)
        return np.interp(np.random.rand(size), prior[::-1], lines[::-1])

def main():
    elo = MarginElo(games)

    today = datetime.today()
    time = np.datetime64(datetime(today.year, today.month, today.day))

    samples = elo.samples(time, 'CLE', 'MIA')
    plt.hist(samples, bins=100)
    plt.show()

    #l = np.linspace(-4, 8, 1000)
    #p = elo.norm_cdf(l, loc=1, scale=2)
    #plt.plot(l, 1 - p, color='k')

    #l, p = elo.prior_cdf(time, 'CLE', 'MIA')
    #plt.scatter(l[1:], -np.diff(p))
    #plt.ylim(1e-4, 1)
    #plt.yscale('log')


if __name__ == "__main__":
    main()
