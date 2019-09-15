# -*- coding: utf-8 -*-

from nose.tools import assert_almost_equal, assert_raises
import warnings

import numpy as np

from ..melo import Melo


def test_class_init():
    """
    Checking melo class constructor

    """
    # dummy class instance
    melo = Melo(0, lines=np.array(0))

    # assert commutes=False requires symmetric lines
    assert_raises(ValueError, Melo, 0, [-1, 2])

    # single comparison
    time = np.datetime64('now')
    label1 = 'alpha'
    label2 = 'beta'
    value = np.random.uniform(-10, 10)

    # fit to the training data
    melo.fit(time, label1, label2, value)

    # multiple comparisons
    times = np.arange(100).astype('datetime64[s]')
    labels1 = np.repeat('alpha', 100)
    labels2 = np.repeat('beta', 100)
    values = np.random.normal(0, 10, size=100)

    # randomize times
    np.random.shuffle(times)
    melo.fit(times, labels1, labels2, values)
    training_data = melo.training_data

    # check comparison length
    assert training_data.shape == (100,)

    # check that comparisons are sorted
    assert np.array_equal(
        np.sort(training_data.time), training_data.time)

    # check first and last times
    assert melo.first_update == times.min()
    assert melo.last_update == times.max()


def test_prior_rating():
    """
    Checking prior (default) rating

    """
    # dummy class instance
    melo = Melo(0)

    # generic comparison data
    times = np.arange(100).astype('datetime64[s]')
    labels1 = np.repeat('alpha', 100)
    labels2 = np.repeat('beta', 100)

    # test null rating for 50-50 outcome
    values = np.append(np.ones(50), -np.ones(50))
    melo.fit(times, labels1, labels2, values)
    prob = melo.probability(0, 'alpha', 'beta')
    assert_almost_equal(prob, .5)

    # test null rating for 10-90 outcome
    values = np.append(np.ones(10), -np.ones(90))
    melo.fit(times, labels1, labels2, values)
    prob = melo.probability(0, 'alpha', 'beta')
    assert_almost_equal(prob, .1)


def test_evolve():
    """
    Checking rating regression

    """
    # random rating decay
    rating_decay = np.random.rand()

    # create a typical regression function
    def regress(time):
        return 1 - np.exp(-rating_decay*time)

    # melo class instance
    melo = Melo(0, regress=regress, regress_unit='year')

    # fit the model to some data
    melo.fit(np.datetime64('now'), 'alpha', 'beta', 0)

    # fix the prior rating
    melo.prior_rating = 0

    # check that the ratings are properly regressed to the mean
    for years in range(4):
        nsec = int(years*melo.seconds['year'])
        seconds = np.timedelta64(nsec, 's')
        assert melo.evolve(1, melo.prior_rating, seconds) == 1 - regress(years)


def test_query_rating():
    """
    Checking rating query function

    """
    # dummy class instance
    melo = Melo(0)

    # single entry
    time = np.datetime64('now')
    label1 = 'alpha'
    label2 = 'beta'
    value = np.random.uniform(-10, 10)

    # train the model
    melo.fit(time, label1, label2, value)

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
    # dummy class instance
    melo = Melo(2)

    # alpha wins, beta loses
    times = np.arange(2).astype('datetime64[s]')
    labels1 = np.repeat('alpha', 2)
    labels2 = np.repeat('beta', 2)
    values = [1, -1]

    # instantiate ratings
    melo.fit(times, labels1, labels2, values)

    # rating_change = k * (obs - prior) = 2 * (1 - .5)
    rec = melo.ratings_history
    assert rec['alpha'].rating[0] == 1
    assert rec['beta'].rating[0] == -1
