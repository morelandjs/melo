#!/usr/bin/env python3

from collections import OrderedDict
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, skellam

from melo import Melo

# font sizes
fontsize = dict(
    large=11,
    normal=10,
    small=9,
    tiny=8,
)

# new tableau colors
# https://www.tableau.com/about/blog/2016/7/colors-upgrade-tableau-10-56782
colors = OrderedDict([
    ('blue', '#4e79a7'),
    ('orange', '#f28e2b'),
    ('green', '#59a14f'),
    ('red', '#e15759'),
    ('cyan', '#76b7b2'),
    ('purple', '#b07aa1'),
    ('brown', '#9c755f'),
    ('yellow', '#edc948'),
    ('pink', '#ff9da7'),
    ('gray', '#bab0ac')
])

plt.rcdefaults()
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif'],
    'mathtext.fontset': 'cm',
    'mathtext.default': 'it',
    'mathtext.rm': 'sans',
    'mathtext.cal': 'sans',
    'font.size': fontsize['normal'],
    'legend.fontsize': fontsize['normal'],
    'axes.labelsize': fontsize['normal'],
    'axes.titlesize': fontsize['large'],
    'xtick.labelsize': fontsize['small'],
    'ytick.labelsize': fontsize['small'],
    'font.weight': 400,
    'axes.labelweight': 400,
    'axes.titleweight': 400,
    'axes.prop_cycle': plt.cycler('color', list(colors.values())),
    'lines.linewidth': .8,
    'lines.markersize': 3,
    'lines.markeredgewidth': 0,
    'patch.linewidth': .8,
    'axes.linewidth': .6,
    'xtick.major.width': .6,
    'ytick.major.width': .6,
    'xtick.minor.width': .4,
    'ytick.minor.width': .4,
    'xtick.major.size': 3.,
    'ytick.major.size': 3.,
    'xtick.minor.size': 2.,
    'ytick.minor.size': 2.,
    'xtick.major.pad': 3.5,
    'ytick.major.pad': 3.5,
    'axes.labelpad': 4.,
    'axes.formatter.limits': (-5, 5),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.color': 'black',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.frameon': False,
    'image.cmap': 'Blues',
    'image.interpolation': 'none',
})

plotdir = Path('figures')
plotdir.mkdir(exist_ok=True)

plot_functions = {}


def plot(f):
    """
    Plot function decorator.  Calls the function, does several generic tasks,
    and saves the figure as the function name.
    """
    def wrapper(*args, **kwargs):
        logging.info('generating plot: %s', f.__name__)
        f(*args, **kwargs)

        fig = plt.gcf()

        plotfile = plotdir / '{}.pdf'.format(f.__name__)
        fig.savefig(str(plotfile))
        logging.info('wrote %s', plotfile)
        plt.close(fig)

    plot_functions[f.__name__] = wrapper

    return wrapper


def figsize(relwidth=1, aspect=.618, refwidth=8):
    """
    Return figure dimensions from a relative width (to a reference width) and
    aspect ratio (default: 1/golden ratio).
    """
    width = relwidth * refwidth

    return width, width*aspect


def set_tight(fig=None, **kwargs):
    """
    Set tight_layout with a better default pad.
    """
    if fig is None:
        fig = plt.gcf()

    kwargs.setdefault('pad', .1)
    fig.set_tight_layout(kwargs)


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
        self.diff = []
        self.total = []

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
                self.diff.append(score1 - score2)
                self.total.append(score1 + score2)


@plot
def all_lines():
    """
    Test rating predictions at every value of the line.

    """
    fig, axes = plt.subplots(ncols=2, figsize=figsize(aspect=.309), sharey=True)

    # construct a fake Poisson league
    league = PoissonLeague(10**5)
    today = datetime.today()

    melo_args = [
        ('Fermi', np.arange(-29.5, 30.5), league.diff, 'scored $-$ allowed'),
        ('Bose', np.arange(10.5, 81.5), league.total, 'scored $+$ allowed'),
    ]

    for ax, (statistics, lines, values, xlabel) in zip(axes, melo_args):

        melo = Melo(
            league.times, league.labels1, league.labels2, values,
            lines=lines, statistics=statistics, k=1e-3
        )

        lambda1 = league.lambdas[0]

        for lambda2 in league.lambdas[1:]:

            # Elo predicted win probability
            lines, prob_to_cover = melo.predict_prob(
                today, str(lambda1), str(lambda2))
            ax.plot(lines, prob_to_cover, 'o',
                    label=r'$\lambda_2={}$'.format(lambda2))

            # true analytic win probability
            if ax.is_first_col():
                prob_to_cover = 1 - skellam.cdf(lines, lambda1, lambda2)
                ax.plot(lines, prob_to_cover, color='k', zorder=0)
            else:
                prob_to_cover = 1 - poisson.cdf(lines, lambda1 + lambda2)
                ax.plot(lines, prob_to_cover, color='k', zorder=0)

        # axes labels
        if ax.is_first_col():
            ax.set_ylabel('Probability to cover line')
        else:
            ax.legend(handletextpad=.2, loc=1)
        if ax.is_last_row():
            ax.set_xlabel(xlabel)

    set_tight()


@plot
def one_line():
    """
    Test rating convergence at one value of the line.

    """
    fig, axes = plt.subplots(ncols=2, figsize=figsize(aspect=.309), sharey=True)

    # construct a fake Poisson league
    league = PoissonLeague(10**5)
    today = datetime.today()

    melo_args = [
        ('Fermi', [-3.5, 3.5], league.diff, 'scored $-$ allowed'),
        ('Bose', [44.5], league.total, 'scored $+$ allowed'),
    ]

    for ax, (statistics, lines, values, xlabel) in zip(axes, melo_args):

        melo = Melo(
            league.times, league.labels1, league.labels2, values,
            lines=lines, statistics=statistics, k=1e-3
        )

        ratings = melo.ratings
        line = lines[-1]

        lambda1 = league.lambdas[0]
        ratings1 = ratings[str(lambda1)]['rating']

        for lambda2 in league.lambdas[1:]:

            # predicted cover probability
            ratings2 = ratings[str(lambda2)]['rating']
            dR = ratings1 + melo.conjugate(ratings2)
            prob = melo.prob_to_cover(dR[:,-1] if dR.ndim > 1 else dR)

            iterations = np.arange(len(ratings1))
            ax.plot(iterations, prob)

            # true analytic win probability
            if ax.is_first_col():
                prob_to_cover = 1 - skellam.cdf(line, lambda1, lambda2)
                ax.axhline(prob_to_cover, color='k')
            else:
                prob_to_cover = 1 - poisson.cdf(line, lambda1 + lambda2)
                ax.axhline(prob_to_cover, color='k')

        # axes labels
        if ax.is_last_row():
            ax.set_xlabel('Iterations')
        if ax.is_first_col():
            ax.set_ylabel('Probability to cover line')
            ax.set_title('Spread: line = {}'.format(line),
                         fontsize=fontsize['small'])
        else:
            ax.set_title('Total: line = {}'.format(line),
                         fontsize=fontsize['small'])

    set_tight()


@plot
def prior_rating():
    """
    Test that the unconditioned prior rating accurately reflects
    the population.

    """
    plt.figure(figsize=figsize(.6))

    # construct a fake Poisson league
    league = PoissonLeague(10**5)
    today = datetime.today()

    melo_args = [
        ('Fermi', np.arange(-29.5, 30.5), league.diff, 'scored $-$ allowed'),
        ('Bose', np.arange(10.5, 81.5), league.total, 'scored $+$ allowed'),
    ]

    for (statistics, lines, values, label) in melo_args:

        melo = Melo(
            league.times, league.labels1, league.labels2, values,
            lines=lines, statistics=statistics, k=1e-3
        )

        # true prior
        outcomes = melo.values[:, np.newaxis] > melo.lines
        outcomes = (outcomes if melo.dim > 1 else outcomes.ravel())
        prob_to_cover = np.sum(outcomes, axis=0) / np.size(outcomes, axis=0)
        plt.plot(lines, prob_to_cover, color='k', zorder=0)

        # constructed prior
        prob_to_cover = melo.prob_to_cover(2*melo.null_rtg)
        plt.plot(lines, prob_to_cover, 'o', label=label)

    plt.xlabel('Lines')
    plt.ylabel('Probability to cover line')
    plt.legend()
    set_tight()


def main():
    import argparse

    logging.basicConfig(
            format='[%(levelname)s][%(module)s] %(message)s',
            level=os.getenv('LOGLEVEL', 'info').upper()
        )

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
