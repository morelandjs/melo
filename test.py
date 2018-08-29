#!/usr/bin/env python3

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

from melo import Elo


class PoissonLeague:
    """
    Create a toy-model league of Poisson random variables.

    """
    today = datetime.today()
    lambdas = [22, 12, 15, 18, 21, 24, 27, 30]

    def __init__(self, handicap=0, cycles=1000):
        self.times = []
        self.labels1 = []
        self.labels2 = []
        self.outcomes = []
        self.values = []
        self.handicap = handicap

        for cycle in range(int(cycles)):
            for lambda1, lambda2 in np.random.choice(
                    self.lambdas, size=(4, 2), replace=False):

                score1, score2 = [
                    np.random.poisson(lam=lam)
                    for lam in (lambda1, lambda2)
                ]
                tiebreaker = np.random.uniform(-.5, .5)
                outcome = score1 - score2 + tiebreaker > handicap

                self.times.append(self.today + timedelta(cycle))
                self.labels1.append(lambda1)
                self.labels2.append(lambda2)
                self.outcomes.append(outcome)
                self.values.append(score1 - score2 + tiebreaker)


def main():
    plt.figure(figsize=(6, 3.375))
    league = PoissonLeague(handicap=3, cycles=1e4)

    elo = Elo(
        league.times,
        league.labels1,
        league.labels2,
        league.values,
        lines=3,
        k=0.01
    )

    lambda1 = league.lambdas[0]
    times, ratings1 = [
        elo.ratings[str(lambda1)][k]
        for k in ('time', 'over')
    ]

    for lambda2 in league.lambdas[1:]:

        # elo predicted win probability
        ratings2 = elo.ratings[str(lambda2)]['under']
        pred = elo.norm_cdf(ratings1 - ratings2)
        plt.plot(times, pred)

        # true win probability
        size = int(1e5)
        scores1, scores2 = [
            np.random.poisson(lam, size=size)
            for lam in (lambda1, lambda2)
        ]
        tiebreaker = np.random.uniform(-.5, .5, size=size)
        outcomes = scores1 - scores2 + tiebreaker > league.handicap
        true = np.count_nonzero(outcomes) / np.size(outcomes)
        plt.axhline(true , ls='dashed', color='k')

    plt.xlim(min(times), max(times))
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    main()
