# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from nose.tools import  assert_almost_equal, assert_raises
import string

import numpy as np
from scipy.stats import norm

#from ..melo import Melo
from melo import Melo


time = datetime.today()


def test_class_init():
    """
    Checking melo class constructor

    """
    # single entry
    label1 = 'alpha'
    label2 = 'beta'
    value = np.random.uniform(-10, 10)
    lines = np.array(0)

    # construct class object
    melo = Melo(time, label1, label2, value, lines=lines)

    # Fermi mode requires symmetric lines
    assert_raises(ValueError, Melo, time, label1, label2, value, [-1, 2], 'Fermi')

    # Only Fermi and Bose modes are defined
    assert_raises(ValueError, Melo, time, label1, label2, value, [-1, 1], 'Other')

    # multiple entries
    times = [time + timedelta(seconds=s) for s in range(100)]
    labels1 = np.random.choice(list(string.ascii_lowercase), size=100)
    labels2 = np.random.choice(list(string.ascii_lowercase), size=100)
    values = np.random.normal(0, 10, size=100)
    lines = np.array(0)

    # randomize times
    np.random.shuffle(times)
    melo = Melo(times, labels1, labels2, values, lines=lines)
    comp = melo.comparisons

    # check comparison length
    assert comp.shape == (100,)

    # check that comparisons are sorted
    assert np.array_equal(np.sort(comp.time), comp.time)

    # check oldest time
    assert melo.oldest == time

    # check that outcomes are correct
    assert np.array_equal(comp.value > 0, comp.outcome)


def test_null_rating():
    """
    Checking null (default) rating

    """
    # multiple
    times = [time + timedelta(seconds=s) for s in range(100)]
    labels1 = np.random.choice(list(string.ascii_lowercase), size=100)
    labels2 = np.random.choice(list(string.ascii_lowercase), size=100)
    lines = np.array(0)

    # test null rating for 50-50 outcome
    values = np.append(np.ones(50), -np.ones(50))
    melo = Melo(times, labels1, labels2, values, lines=lines)
    null_rtg = melo.null_rating(melo.comparisons.outcome)
    assert_almost_equal(norm.cdf(2*null_rtg), .5)

    # test null rating for 10-90 outcome
    values = np.append(np.ones(10), -np.ones(90))
    melo = Melo(times, labels1, labels2, values, lines=lines)
    null_rtg = melo.null_rating(melo.comparisons.outcome)
    assert_almost_equal(norm.cdf(2*null_rtg), .1)


def test_regress():
    """
    Checking rating regression

    """
    label1 = 'alpha'
    label2 = 'beta'
    value = np.random.uniform(-10, 10)
    lines = np.array(0)

    melo = Melo(time, label1, label2, value, lines=lines,
                decay=lambda t: .5 if t > timedelta(days=.5) else 1)

    melo.null_rtg = 0
    rating = 1

    for p in range(1, 5):
        rating = melo.regress(rating, timedelta(days=1))
        assert rating == .5**p


def test_query_rating():
    """
    Checking rating query function

    """
    # single entry
    label1 = 'alpha'
    label2 = 'beta'
    value = np.random.uniform(-10, 10)
    lines = np.array(0)

    # construct class object
    melo = Melo(time, label1, label2, value, lines=lines)

    one_hour = timedelta(hours=1)
    melo.ratings['alpha'] = np.array(
        [(time - one_hour, 1), (time, 2), (time + one_hour, 3)],
        dtype=[('time', 'M8[us]'), ('rating', 'f8')]
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
    times = [time, time + timedelta(hours=1)]
    labels1 = ['alpha', 'alpha']
    labels2 = ['beta', 'beta']
    values = [1, -1]
    lines = 0

    # instantiate ratings
    melo = Melo(times, labels1, labels2, values, lines=lines, k=2)
    null_rtg = melo.null_rating(melo.comparisons.outcome)

    # rating_change = k * (obs - prior) = 2 * (1 - .5)
    assert melo.ratings['alpha']['rating'][0] == 1
    assert melo.ratings['beta']['rating'][0] == -1


def test_predict():
    """
    Checking comparison prediction function

    """

