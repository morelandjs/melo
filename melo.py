#!/usr/bin/env python3

from datetime import datetime
from itertools import chain
from scipy.special import erf, erfinv
import numpy as np

import matplotlib.pyplot as plt


today = datetime.today()
date = datetime(today.year, today.month, today.day)

games = []

for rand in np.random.normal(0, 1, size=10**5):
    games.append((date, 'CLE', 'MIA', rand))

class MarginElo:
    def __init__(self, comparisons, lines=0, k=0.001):
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
                    ('alpha',   'U10'),
                    ('beta',    'U10'),
                    ('value',    'f8'),
                ]
            ), axis=0
        )

        self.lines = np.array(lines, dtype=float, ndmin=1)

        self.k = k

        self.ratings = {}

        self.rate(self.comparisons, self.lines)

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
            dtype=float
        )
        return np.sqrt(2)/2*erfinv(2*prob - 1)

    def query_rating(self, date, name, default_rtg):
        """
        Look up the most recent margin dependent ratings preceeding 'date'.
        If no entry exists, return 'default_rtg'.

        """
        try:
            rtg = self.ratings[name]
            return rtg[rtg['date'] < date]['rating'][-1]
        except KeyError:
            self.ratings[name] = (date, default_rtg)
            return default_rtg

    def rate(self, comparisons, lines):
        """
        Apply the margin dependent Elo model to the list of binary comparisons.

        """
        mean = comparisons['value'].mean()
        std = comparisons['value'].std()

        comparisons['value'] -= mean
        comparisons['value'] /= std

        lines -= mean
        lines /= std

        default_rtg = self.naive_prior(comparisons['value'], lines)

        for (date, alpha, beta, value) in comparisons:

            # lookup ratings
            alpha_rtg, beta_rtg = [
                self.query_rating(date, name, default_rtg)
                for name in (alpha, beta)
            ]

            # ratings difference
            rtg_diff = alpha_rtg - beta_rtg[::-1]

            # prior and posterior outcome probabilities
            prior = self.norm_cdf(rtg_diff)
            post = np.heaviside(value - lines, .5)

            # rating change
            rtg_change = self.k * (post - prior)

            print(rtg_change)

            # update ratings
            #self.ratings['alpha'].append([date, alpha_rtg + rtg_change])
            #self.ratings['beta'].append([date, beta_rtg - rtg_change])

def main():
    elo = MarginElo(games)
    #hcap, prior = elo.prior('CLE', 'MIA')
    #plt.plot(hcap, prior)
    #plt.show()



if __name__ == "__main__":
    main()
