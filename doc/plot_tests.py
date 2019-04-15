#!/usr/bin/env python3

from collections import OrderedDict
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, poisson, skellam

from melo import Melo


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

default_color = '#404040'
dashed_line = dict(color=default_color, linestyle='dashed')
font_size = 18

plt.rcdefaults()
plt.rcParams.update({
    'figure.figsize': (10, 6.18),
    'figure.dpi': 200,
    'figure.autolayout': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Lato'],
    'mathtext.fontset': 'custom',
    'mathtext.cal': 'sans',
    'font.size': font_size,
    'legend.fontsize': font_size,
    'axes.labelsize': font_size,
    'axes.titlesize': font_size,
    'axes.prop_cycle': plt.cycler('color', list(colors.values())),
    'xtick.labelsize': font_size - 2,
    'ytick.labelsize': font_size - 2,
    'lines.linewidth': 1.25,
    'lines.markeredgewidth': .1,
    'patch.linewidth': 1.25,
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.facecolor': '#eaeaf2',
    'axes.linewidth': 0,
    'grid.linestyle': '-',
    'grid.linewidth': 1,
    'grid.color': '#fcfcfc',
    'savefig.facecolor': '#fcfcfc',
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'xtick.minor.size': 0,
    'ytick.minor.size': 0,
    'xtick.major.pad': 7,
    'ytick.major.pad': 7,
    'text.color': default_color,
    'axes.edgecolor': default_color,
    'axes.labelcolor': default_color,
    'xtick.color': default_color,
    'ytick.color': default_color,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.frameon': False,
    'image.interpolation': 'none',
})

plotdir = Path('_static')
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

        plotfile = plotdir / '{}.png'.format(f.__name__)
        fig.savefig(str(plotfile))
        logging.info('wrote %s', plotfile)
        plt.close(fig)

    plot_functions[f.__name__] = wrapper

    return wrapper


def figsize(relwidth=1, aspect=.618, refwidth=10):
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


league = PoissonLeague(10**5)
today = datetime.today()

melo_spreads = Melo(
    league.times, league.labels1, league.labels2, league.diff,
    lines=np.arange(-29.5, 30.5), commutes=False, k=1e-3
)

melo_totals = Melo(
    league.times, league.labels1, league.labels2, league.total,
    lines=np.arange(10.5, 80.5), commutes=True, k=1e-3
)


@plot
def spread_prior():
    """
    Test rating predictions at every value of the line.

    """
    # reference label and lines
    lambda1 = league.lambdas[0]
    lines = melo_spreads.lines

    # target starting distribution
    outcomes = melo_spreads.values[:, np.newaxis] > lines
    prob_to_cover = np.mean(outcomes, axis=0)
    plt.plot(lines, prob_to_cover, color='k')

    for n, lambda2 in enumerate(league.lambdas[1:]):

        # target calibrated distributions
        prob_to_cover = skellam.sf(lines, lambda1, lambda2)
        plt.plot(lines, prob_to_cover, color='.6', zorder=1)

        # model starting distributions
        prob_to_cover = norm.cdf(2*melo_spreads.prior_rating)
        skip = len(league.lambdas[1:])
        plt.plot(lines[n::skip], prob_to_cover[n::skip], 'o',
                 label=r'$\lambda_2={}$'.format(lambda2), zorder=2)

        plt.xlabel('line $=$ scored $-$ allowed')
        plt.ylabel('probability to cover line')

        l = plt.legend(title=r'$\lambda_1 = {}$'.format(lambda1),
                    handletextpad=.2, loc=1)
        l._legend_box.align = 'right'

        # axes ticks
        l = np.floor(lines)
        plt.xticks(l[::10])
        plt.xlim(l.min(), l.max())

    set_tight()


@plot
def spread_calibrated():
    """
    Test rating predictions at every value of the line.

    """
    lambda1 = league.lambdas[0]

    for lambda2 in league.lambdas[1:]:

        # train margin-dependent Elo model
        lines, prob_to_cover = melo_spreads._predict(
            today, str(lambda1), str(lambda2))

        # Elo predicted win probability
        plt.plot(lines, prob_to_cover, 'o',
                 label=r'$\lambda_2={}$'.format(lambda2))

        # true analytic win probability
        prob_to_cover = skellam.sf(lines, lambda1, lambda2)
        plt.plot(lines, prob_to_cover, color='k')

        plt.xlabel('line $=$ scored $-$ allowed')
        plt.ylabel('probability to cover line')

        l = plt.legend(title=r'$\lambda_1 = {}$'.format(lambda1),
                    handletextpad=.2, loc=1)
        l._legend_box.align = 'right'

        # axes ticks
        l = np.floor(lines)
        plt.xticks(l[::10])
        plt.xlim(l.min(), l.max())

    set_tight()


@plot
def total_prior():
    """
    Test rating predictions at every value of the line.

    """
    # reference label and lines
    lambda1 = league.lambdas[0]
    lines = melo_totals.lines

    # target starting distribution
    outcomes = melo_totals.values[:, np.newaxis] > lines
    prob_to_cover = np.mean(outcomes, axis=0)
    plt.plot(lines, prob_to_cover, color='k')

    for n, lambda2 in enumerate(league.lambdas[1:]):

        # target calibrated distributions
        prob_to_cover = poisson.sf(lines, lambda1 + lambda2)
        plt.plot(lines, prob_to_cover, color='.6', zorder=1)

        # model starting distributions
        prob_to_cover = norm.cdf(2*melo_totals.prior_rating)
        skip = len(league.lambdas[1:])
        plt.plot(lines[n::skip], prob_to_cover[n::skip], 'o',
                 label=r'$\lambda_2={}$'.format(lambda2), zorder=2)

        plt.xlabel('line $=$ scored $+$ allowed')
        plt.ylabel('probability to cover line')

        l = plt.legend(title=r'$\lambda_1 = {}$'.format(lambda1),
                    handletextpad=.2, loc=1)
        l._legend_box.align = 'right'

        # axes ticks
        l = np.floor(lines)
        plt.xticks(l[::10])
        plt.xlim(l.min(), l.max())

    set_tight()


@plot
def total_calibrated():
    """
    Test rating predictions at every value of the line.

    """
    # reference label
    lambda1 = league.lambdas[0]

    for lambda2 in league.lambdas[1:]:

        # train margin-dependent Elo model
        lines, prob_to_cover = melo_totals._predict(
            today, str(lambda1), str(lambda2))

        # Elo predicted win probability
        plt.plot(lines, prob_to_cover, 'o',
                 label=r'$\lambda_2={}$'.format(lambda2))

        # true analytic win probability
        prob_to_cover = poisson.sf(lines, lambda1 + lambda2)
        plt.plot(lines, prob_to_cover, color='k')

        plt.xlabel('line $=$ scored $+$ allowed')
        plt.ylabel('probability to cover line')

        l = plt.legend(title=r'$\lambda_1 = {}$'.format(lambda1),
                    handletextpad=.2, loc=1)
        l._legend_box.align = 'right'

        # axes ticks
        l = np.floor(lines)
        plt.xticks(l[::10])
        plt.xlim(l.min(), l.max())

    set_tight()


@plot
def spread_convergence():
    """
    Test rating convergence at one value of the line.

    """
    fig, axes = plt.subplots(
        ncols=2, figsize=figsize(aspect=.4), sharey=True)

    melo_args = [
        (melo_spreads, [-3.5, 3.5], 'scored $-$ allowed'),
        (melo_totals, [42.5], 'scored $+$ allowed'),
    ]

    for ax, (melo, lines, xlabel) in zip(axes, melo_args):

        times = melo.ratings_history.time
        line = lines[-1]

        lambdas1 = np.full_like(times, league.lambdas[0], dtype=str)

        for lambda2 in league.lambdas[1:]:

            lambdas2 = np.full_like(times, lambda2, dtype=str)

            # predicted cover probability
            prob = melo.probability(times, lambdas1, lambdas2, lines=line)
            iterations = np.arange(len(times))
            ax.plot(iterations, prob)

            # true analytic win probability
            if ax.is_first_col():
                prob_to_cover = skellam.sf(line, lambda1, lambda2)
                ax.axhline(prob_to_cover, color='k')
            else:
                prob_to_cover = poisson.sf(line, lambda1 + lambda2)
                ax.axhline(prob_to_cover, color='k')

        # axes labels
        if ax.is_last_row():
            ax.set_xlabel('Iterations')
        if ax.is_first_col():
            ax.set_ylabel('Probability to cover line')
            ax.set_title('Spread: line = {}'.format(line),
                         fontsize=font_size)
        else:
            ax.set_title('Total: line = {}'.format(line),
                         fontsize=font_size)

    set_tight(w_pad=.5)


@plot
def one_line():
    """
    Test rating convergence at one value of the line.

    """
    fig, axes = plt.subplots(ncols=2, figsize=figsize(aspect=.4), sharey=True)

    # construct a fake Poisson league
    league = PoissonLeague(10**5)
    today = datetime.today()

    melo_args = [
        (False, [-3.5, 3.5], league.diff, 'scored $-$ allowed'),
        (True, [42.5], league.total, 'scored $+$ allowed'),
    ]

    for ax, (commutes, lines, values, xlabel) in zip(axes, melo_args):

        melo = Melo(
            league.times, league.labels1, league.labels2, values,
            lines=lines, commutes=commutes, k=1e-3
        )

        ratings = melo.ratings
        line = lines[-1]

        lambda1 = league.lambdas[0]
        ratings1 = ratings[str(lambda1)]['rating']

        for lambda2 in league.lambdas[1:]:

            # predicted cover probability
            ratings2 = ratings[str(lambda2)]['rating']
            dR = ratings1 + melo.conjugate(ratings2)
            prob = norm.cdf(dR[:,-1] if dR.ndim > 1 else dR)

            iterations = np.arange(len(ratings1))
            ax.plot(iterations, prob)

            # true analytic win probability
            if ax.is_first_col():
                prob_to_cover = skellam.sf(line, lambda1, lambda2)
                ax.axhline(prob_to_cover, color='k')
            else:
                prob_to_cover = poisson.sf(line, lambda1 + lambda2)
                ax.axhline(prob_to_cover, color='k')

        # axes labels
        if ax.is_last_row():
            ax.set_xlabel('Iterations')
        if ax.is_first_col():
            ax.set_ylabel('Probability to cover line')
            ax.set_title('Spread: line = {}'.format(line),
                         fontsize=font_size)
        else:
            ax.set_title('Total: line = {}'.format(line),
                         fontsize=font_size)

    set_tight(w_pad=.5)


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
        (False, np.arange(-29.5, 30.5), league.diff, 'scored $-$ allowed'),
        (True, np.arange(10.5, 81.5), league.total, 'scored $+$ allowed'),
    ]

    for (commutes, lines, values, label) in melo_args:

        melo = Melo(
            league.times, league.labels1, league.labels2, values,
            lines=lines, commutes=commutes, k=1e-3, bias=.5
        )

        # constructed prior
        prob_to_cover = melo.dist.cdf(2*melo.prior_rating)
        plt.plot(lines, prob_to_cover, 'o', label=label)

        # true prior
        outcomes = melo.values[:, np.newaxis] > melo.lines
        outcomes = (outcomes if melo.lines.size > 1 else outcomes.ravel())
        prob_to_cover = np.sum(outcomes, axis=0) / np.size(outcomes, axis=0)
        plt.plot(lines, prob_to_cover, color='k')

    plt.xlabel('Lines')
    plt.ylabel('Probability to cover line')
    plt.legend(loc='upper right', handletextpad=0)
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
