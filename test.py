#!/usr/bin/env python3

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skellam

from melo import Melo


plot_functions = {}
plotdir = Path('figures')
plotdir.mkdir(exist_ok=True)

def plot(f):
    def wrapper(*args, **kwargs):
        print(f.__name__)
        f(*args, **kwargs)
        plt.savefig('{}/{}.pdf'.format(plotdir, f.__name__))
        plt.close()

    plot_functions[f.__name__] = wrapper

    return wrapper


def finish(despine=True, remove_ticks=False, pad=0.1, h_pad=None, w_pad=None,
           rect=[0, 0, 1, 1]):
    fig = plt.gcf()

    for ax in fig.axes:
        if despine:
            for spine in 'top', 'right':
                ax.spines[spine].set_visible(False)

        if remove_ticks:
            for ax_name in 'xaxis', 'yaxis':
                getattr(ax, ax_name).set_ticks_position('none')
        else:
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

    fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)


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


@plot
def validate_all():
    """
    Validate predictions at every value of the line.

    """
    # construct a fake Poisson league
    league = PoissonLeague(10**5)
    lines = np.linspace(-29.5, 30.5, 61)
    now = datetime.today()

    # calculate ratings
    melo = Melo(
        league.times,
        league.labels1,
        league.labels2,
        league.values,
        lines=lines,
        k=0.002
    )

    lambda1 = league.lambdas[0]

    for lambda2 in league.lambdas[1:]:

        # Elo predicted win probability
        x, y = melo.predict_prob(now, str(lambda1), str(lambda2))
        plt.plot(x, y, 'o', label='Skellam({}, {})'.format(lambda1, lambda2))

        # true win probability
        x = np.linspace(-29.5, 30.5, 601)
        y = 1 - skellam.cdf(x, lambda1, lambda2)
        plt.plot(x, y, color='k')

    # axes labels
    plt.xlabel('Line')
    plt.ylabel('Probability to cover line')
    plt.legend()
    finish()


@plot
def validate_one(line=0):
    """
    Validate convergence at one value of the line.

    """
    # construct a fake Poisson league
    league = PoissonLeague(10**5)
    now = datetime.today()

    # calculate ratings
    melo = Melo(
        league.times,
        league.labels1,
        league.labels2,
        league.values,
        lines=line,
        k=0.002
    )

    lambda1 = league.lambdas[0]
    ratings1 = melo.ratings[str(lambda1)]
    iterations = np.arange(ratings1.size)

    for lambda2 in league.lambdas[1:]:
        ratings2 = melo.ratings[str(lambda2)]

        # predicted cover probability
        ratings_diff = ratings1['over'] - ratings2['under']
        prob = melo.norm_cdf(ratings_diff)
        label = 'Skellam({}, {})'.format(lambda1, lambda2)
        plt.plot(iterations, prob, label=label)

        # true cover probability
        exact = 1 - skellam.cdf(line, lambda1, lambda2)
        plt.axhline(exact, color='k')

    # axes labels
    plt.xlabel('Iterations')
    plt.ylabel('Probability to cover line={}'.format(line))
    plt.ylim(0, 1)
    plt.legend(loc='lower left')
    finish()


@plot
def mean_predictions():
    """
    Scatterplot predicted vs known spread means.

    """
    # construct a fake Poisson league
    league = PoissonLeague(10**5)
    lines = np.linspace(-29.5, 30.5, 61)
    now = datetime.today()

    # calculate ratings
    melo = Melo(
        league.times,
        league.labels1,
        league.labels2,
        league.values,
        lines=lines,
        k=0.002
    )

    # mean-value predictions
    predictors = melo.predictors
    lambda1 = predictors['label1'].astype(float)
    lambda2 = predictors['label2'].astype(float)

    # compare predicted and true values
    pred = predictors['mean']
    true = lambda1 - lambda2

    # scatter plot predicted vs exact
    plt.plot(pred, true, 'o')
    plt.plot([-30, 30], [-30, 30], color='k', ls='dashed')

    # axes labels
    text = r'$\langle \mathrm{Skellam}(\lambda_i, \lambda_j) \rangle$'
    plt.xlabel(r'Predicted {}'.format(text))
    plt.ylabel(r'Analytic {}'.format(text))
    finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plots', nargs='*')
    args = parser.parse_args()

    if args.plots:
        for i in args.plots:
            if i.endswith('.pdf'):
                i = i[:-4]
            if i in plot_functions:
                plot_functions[i]()
            else:
                print('unknown plot:', i)
    else:
        for f in plot_functions.values():
            f()


if __name__ == "__main__":
    main()
