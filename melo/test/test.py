# -*- coding: utf-8 -*-

from nose.tools import assert_almost_equal, assert_raises

import numpy as np
from scipy.stats import norm

from ..melo import Melo


def test_class_init():
    """
    Checking melo class constructor

    """
    # single entry
    time = np.datetime64('now')
    label1 = 'alpha'
    label2 = 'beta'
    value = np.random.uniform(-10, 10)
    lines = np.array(0)

    # construct class object
    melo = Melo(time, label1, label2, value, lines=lines)

    # assert commutes=False requires symmetric lines
    assert_raises(ValueError, Melo, time, label1, label2, value, [-1, 2])

    # multiple entries
    times = np.arange(100).astype('datetime64[s]')
    labels1 = np.repeat('alpha', 100)
    labels2 = np.repeat('beta', 100)
    values = np.random.normal(0, 10, size=100)

    # randomize times
    np.random.shuffle(times)
    melo = Melo(times, labels1, labels2, values, lines=0)
    comp = melo.comparisons

    # check comparison length
    assert comp.shape == (100,)

    # check that comparisons are sorted
    assert np.array_equal(np.sort(comp.time), comp.time)

    # check times
    assert melo.first_update == times.min()
    assert melo.last_update == times.max()


def test_prior_rating():
    """
    Checking prior (default) rating

    """
    times = np.arange(100).astype('datetime64[s]')
    labels1 = np.repeat('alpha', 100)
    labels2 = np.repeat('beta', 100)

    # test null rating for 50-50 outcome
    values = np.append(np.ones(50), -np.ones(50))
    melo = Melo(times, labels1, labels2, values, lines=0)
    prob = norm.cdf(2*melo.prior_rating)[0]
    assert_almost_equal(prob, .5)

    # test null rating for 10-90 outcome
    values = np.append(np.ones(10), -np.ones(90))
    melo = Melo(times, labels1, labels2, values, lines=0)
    prob = norm.cdf(2*melo.prior_rating)[0]
    assert_almost_equal(prob, .1)


def test_evolve():
    """
    Checking rating regression

    """
    decay_rate = np.random.rand()

    def regress(time):
        return 1 - np.exp(-decay_rate*time)

    melo = Melo(
        np.datetime64('now'), 'alpha', 'beta', 0,
        regress=regress, regress_unit='year'
    )

    melo.prior_rating = 0

    for years in range(4):
        nsec = int(years*melo.seconds['year'])
        seconds = np.timedelta64(nsec, 's')
        assert melo.evolve(1, seconds) == 1 - regress(years)


def test_query_rating():
    """
    Checking rating query function

    """
    # single entry
    time = np.datetime64('now')
    label1 = 'alpha'
    label2 = 'beta'
    value = np.random.uniform(-10, 10)

    # construct class object
    melo = Melo(time, label1, label2, value, lines=0)

    one_hour = np.timedelta64(1, 'h')
    melo.ratings_history['alpha'] = np.rec.array(
        [(time - one_hour, 1), (time, 2), (time + one_hour, 3)],
        dtype=[('time', 'datetime64[s]'), ('rating', 'float', 1)]
    )

    rating = melo.query_rating(time, 'alpha')
    assert_almost_equal(rating, 1)

    rating = melo.query_rating(time + one_hour, 'alpha')
    assert_almost_equal(rating, 2)


def test_rate():
    """
    Checking core rating function

    """
    # alpha wins, beta loses
    times = np.arange(2).astype('datetime64[s]')
    labels1 = np.repeat('alpha', 2)
    labels2 = np.repeat('beta', 2)
    values = [1, -1]

    # instantiate ratings
    melo = Melo(times, labels1, labels2, values, k=2)

    # rating_change = k * (obs - prior) = 2 * (1 - .5)
    rec = melo.ratings_history
    assert rec['alpha'].rating[0] == 1
    assert rec['beta'].rating[0] == -1
