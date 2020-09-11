#!/usr/bin/env python3

from collections import OrderedDict
from itertools import combinations
import logging
import os
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, skellam

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


class League:
    """
    Create a toy-model league of Poisson random variables.

    """
    lambdas = [100, 90, 94, 98, 102, 106, 110]
    lambda1, *lambda2_list = lambdas

    def __init__(self, size=1000):

        self.times = np.arange(size).astype('datetime64[s]')
        lambdas1, lambdas2 = np.random.choice(self.lambdas, size=(2, size))

        self.spreads = skellam.rvs(mu1=lambdas1, mu2=lambdas2, size=size)
        self.totals = poisson.rvs(mu=(lambdas1 + lambdas2), size=size)

        self.labels1 = lambdas1.astype(str)
        self.labels2 = lambdas2.astype(str)


league = League(10**6)


@plot
def quickstart_example():
    """
    Create time series of comparison data by pairing and
    substracting 100 different Poisson distributions

    """
    mu_values = np.random.randint(80, 110, 100)
    mu1, mu2 = map(np.array, zip(*combinations(mu_values, 2)))
    labels1, labels2 = [mu.astype(str) for mu in [mu1, mu2]]
    spreads = skellam.rvs(mu1=mu1, mu2=mu2)
    times = np.arange(spreads.size).astype('datetime64[s]')

    # MELO class arguments (explained in docs)
    lines = np.arange(-59.5, 60.5)
    k = .15

    # train the model on the list of comparisons
    melo = Melo(lines=lines, k=k)
    melo.fit(times, labels1, labels2, spreads)

    # predicted and true (analytic) comparison values
    pred_times = np.repeat(melo.last_update, times.size)
    pred = melo.mean(pred_times, labels1, labels2)
    true = skellam.mean(mu1=mu1, mu2=mu2)

    # plot predicted means versus true means
    plt.scatter(pred, true)
    plt.plot([-20, 20], [-20, 20], color='k')
    plt.xlabel('predicted mean')
    plt.ylabel('true mean')


@plot
def validate_spreads():
    """
    Prior spread predictions at every value of the line.

    """
    fig, (ax_prior, ax2_post) = plt.subplots(
        nrows=2, figsize=figsize(aspect=1.2))

    # train margin-dependent Elo model
    melo = Melo(lines=np.arange(-49.5, 50.5), commutes=False, k=1e-4)
    melo.fit(league.times, league.labels1, league.labels2, league.spreads)

    # exact prior distribution
    outcomes = melo.training_data.value[:, np.newaxis] > melo.lines
    sf = np.mean(outcomes, axis=0)
    ax_prior.plot(melo.lines, sf, color='k')

    # label names
    label1 = str(league.lambda1)
    label2_list = [str(lambda2) for lambda2 in league.lambda2_list]

    plot_args = [
        (ax_prior, melo.first_update, 'prior'),
        (ax2_post, melo.last_update, 'posterior'),
    ]

    for ax, time, title in plot_args:
        for n, label2 in enumerate(label2_list):

            lines, sf = melo._predict(time, label1, label2)
            label = r'$\lambda_2={}$'.format(label2)

            if ax.is_first_row():
                ax.plot(lines[n::6], sf[n::6], 'o', zorder=2, label=label)

            if ax.is_last_row():
                ax.plot(lines, sf, 'o', zorder=2, label=label)

                sf = skellam.sf(melo.lines, int(label1), int(label2))
                ax.plot(melo.lines, sf, color='k')

            leg = ax.legend(title=r'$\lambda_1 = {}$'.format(label1),
                            handletextpad=.2, loc=1)
            leg._legend_box.align = 'right'

            lines = np.floor(lines)
            ax.set_xticks(lines[::10])
            ax.set_xlim(lines.min(), lines.max())

            if ax.is_last_row():
                ax.set_xlabel('line $=$ scored $-$ allowed')

            ax.set_ylabel('probability to cover line')

            ax.annotate(title, xy=(.05, .05),
                        xycoords='axes fraction', fontsize=24)

    set_tight(h_pad=1)


@plot
def validate_totals():
    """
    Prior spread predictions at every value of the line.

    """
    fig, (ax_prior, ax2_post) = plt.subplots(
        nrows=2, figsize=figsize(aspect=1.2))

    # train margin-dependent Elo model
    melo = Melo(lines=np.arange(149.5, 250.5), commutes=True, k=1e-4)
    melo.fit(league.times, league.labels1, league.labels2, league.totals)

    # exact prior distribution
    outcomes = melo.training_data.value[:, np.newaxis] > melo.lines
    sf = np.mean(outcomes, axis=0)
    ax_prior.plot(melo.lines, sf, color='k')

    # label names
    label1 = str(league.lambda1)
    label2_list = [str(lambda2) for lambda2 in league.lambda2_list]

    plot_args = [
        (ax_prior, melo.first_update, 'prior'),
        (ax2_post, melo.last_update, 'posterior'),
    ]

    for ax, time, title in plot_args:
        for n, label2 in enumerate(label2_list):

            lines, sf = melo._predict(time, label1, label2)
            label = r'$\lambda_2={}$'.format(label2)

            if ax.is_first_row():
                ax.plot(lines[n::6], sf[n::6], 'o', zorder=2, label=label)

            if ax.is_last_row():
                ax.plot(lines, sf, 'o', zorder=2, label=label)

                sf = poisson.sf(melo.lines, int(label1) + int(label2))
                ax.plot(melo.lines, sf, color='k')

            leg = ax.legend(title=r'$\lambda_1 = {}$'.format(label1),
                            handletextpad=.2, loc=1)
            leg._legend_box.align = 'right'

            lines = np.floor(lines)
            ax.set_xticks(lines[::10])
            ax.set_xlim(lines.min(), lines.max())

            if ax.is_last_row():
                ax.set_xlabel('line $=$ scored $+$ allowed')

            ax.set_ylabel('probability to cover line')

            ax.annotate(title, xy=(.05, .05),
                        xycoords='axes fraction', fontsize=24)

    set_tight(h_pad=1)


@plot
def convergence():
    """
    Test rating convergence at single value of the line.

    """
    fig, axes = plt.subplots(nrows=2, figsize=figsize(aspect=1.2))

    # label names
    label1 = str(league.lambda1)
    label2_list = [str(lambda2) for lambda2 in league.lambda2_list]

    # point spread and point total subplots
    subplots = [
        (False, [-0.5, 0.5], league.spreads, 'probability spread > 0.5'),
        (True, [200.5], league.totals, 'probability total > 200.5'),
    ]

    for ax, (commutes, lines, values, ylabel) in zip(axes, subplots):

        # train margin-dependent Elo model
        melo = Melo(lines=lines, commutes=commutes, k=1e-4)
        melo.fit(league.times, league.labels1, league.labels2, values)

        line = lines[-1]

        for label2 in label2_list:

            # evaluation times and labels
            times = np.arange(league.times.size)[::1000]
            labels1 = times.size * [label1]
            labels2 = times.size * [label2]

            # observed win probability
            prob = melo.probability(times, labels1, labels2, lines=line)
            ax.plot(times, prob)

            # true (analytic) win probability
            if ax.is_first_row():
                prob = skellam.sf(line, int(label1), int(label2))
                ax.axhline(prob, color='k')
            else:
                prob = poisson.sf(line, int(label1) + int(label2))
                ax.axhline(prob, color='k')

        # axes labels
        if ax.is_last_row():
            ax.set_xlabel('Iterations')
        ax.set_ylabel(ylabel)

    set_tight(w_pad=.5)


def main():
    import argparse

    logging.basicConfig(
            format='[%(levelname)s][%(module)s] %(message)s',
            level=os.getenv('LOGLEVEL', 'info').upper()
        )

    parser = argparse.ArgumentParser()
    parser.add_argument('plots', nargs='*')
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)

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
