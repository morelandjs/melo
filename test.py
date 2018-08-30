#!/usr/bin/env python3

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

from melo import Melo

from scipy.stats import skellam


class PoissonLeague:
    """
    Create a toy-model league of Poisson random variables.

    """
    today = datetime.today()
    lambdas = [22, 12, 15, 18, 21, 24, 27, 30]

    def __init__(self, cycles=1000):
        self.times = []
        self.labels1 = []
        self.labels2 = []
        self.values = []

        for cycle in range(int(cycles)):
            for lambda1, lambda2 in np.random.choice(
                    self.lambdas, size=(4, 2), replace=False):

                score1, score2 = [
                    np.random.poisson(lam=lam)
                    for lam in (lambda1, lambda2)
                ]

                self.times.append(self.today - timedelta(hours=cycle))
                self.labels1.append(lambda1)
                self.labels2.append(lambda2)
                self.values.append(score1 - score2)


def main():
    plt.figure(figsize=(9, 5))
    league = PoissonLeague(cycles=1e5)

    lines = np.linspace(-29.5, 30.5, 61)

    melo = Melo(
        league.times,
        league.labels1,
        league.labels2,
        league.values,
        lines=lines,
        k=0.001
    )

    lambda1 = league.lambdas[0]

    for lambda2 in league.lambdas[1:]:

        # elo predicted win probability
        x, y = melo.prior(
            datetime.today(),
            str(lambda1),
            str(lambda2)
        )

        plt.plot(x, y, 'o')

        # true win probability
        x = np.linspace(-30, 30, 1000)
        y = 1 - skellam.cdf(x, lambda1, lambda2)

        plt.plot(x, y, color='k')

    plt.xlabel('Spread')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    main()
